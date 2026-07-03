#!/usr/bin/env python3
"""Self-scheduling worker for the 12-run astlm training matrix.

Run the exact same command on every node/terminal:
    python3 scripts/orchestrator.py

Each worker claims the incomplete setting with the LOWEST current progress
(min-heap by %done); ties are broken by preferring ft types (2/15/full) with
the fewest settings claimed so far, so parallel workers spread across
architectures instead of exhausting one ft type before touching the others.
Trains the claimed setting to its target step count, then loops to claim the
next-lowest. Progress is read directly from checkpoints/<name>/ on disk
(ground truth); orchestrator_state.csv is only used to avoid two workers
claiming the same setting.

While a setting trains, a background poller appends a "checkpoint" row to
orchestrator_state.csv every time a new checkpoint-N dir appears on disk
(checked every 20s), so progress across all workers is visible from one file
without tailing each logs/<setting>.log separately.

Crash recovery is manual: if a worker dies mid-run its last row ("started" or
"checkpoint") has no matching "finished"/"failed" row, which blocks that
setting from being re-claimed. If a worker crashes, delete/edit its stale
row in orchestrator_state.csv before restarting a worker.

GPU count per node is fixed at NPROC_PER_NODE (not env-configurable): the
target step count (see run_matrix.py) assumes a constant effective batch size
across every resume of a given setting. If nodes had different GPU counts,
the same run could get resumed under a different global batch size at
different points in its history, making "steps" incomparable. Every worker
must run on a node with exactly this many GPUs.
"""
import os, sys, csv, time, socket, subprocess, fcntl, threading
from pathlib import Path

REPO_ROOT       = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
LOGS_DIR        = REPO_ROOT / "logs"
NPROC_PER_NODE  = 4
STATE_CSV       = REPO_ROOT / "scripts" / "orchestrator_state.csv"
LOCK_FILE       = str(STATE_CSV) + ".lock"
STATE_FIELDS    = ["timestamp", "hostname", "setting", "status", "pct_at_claim"]

sys.path.insert(0, os.path.dirname(__file__))
from run_matrix import RUNS, setting_name as _setting_name  # noqa: E402

# train_multigpu.py's --no-boundary-tokens flag needs one of these per run; the
# rest of RUNS (model/ft/max_steps/batch_size) comes straight from run_matrix.py,
# the same module train_multigpu.py reads its defaults from — single source of
# truth, see run_matrix.py's docstring for why this was split out.


def setting_name(r):
    return _setting_name(r["model"], r["ft"], r["nob"])


SETTING_TO_FT = {setting_name(r): r["ft"] for r in RUNS}


def current_step(name):
    d = CHECKPOINTS_DIR / name
    if not d.is_dir():
        return 0
    steps = [
        int(p.name.split("-")[-1])
        for p in d.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    return max(steps) if steps else 0


def pct_done(r):
    step = current_step(setting_name(r))
    return round(100 * step / r["max_steps"], 2)


def read_state():
    if not STATE_CSV.exists():
        return []
    with open(STATE_CSV) as f:
        return list(csv.DictReader(f))


def append_state_row(row):
    exists = STATE_CSV.exists()
    with open(STATE_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATE_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def append_state(hostname, setting, status, pct):
    LOCK_FILE_DIR = Path(LOCK_FILE).parent
    LOCK_FILE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            append_state_row({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hostname": hostname, "setting": setting,
                "status": status, "pct_at_claim": pct,
            })
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def claim_next(hostname):
    Path(LOCK_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            last_status = {}
            ft_attempts = {}  # ft type -> count of settings with any history (claimed at least once)
            for row in read_state():
                last_status[row["setting"]] = row["status"]  # later rows overwrite
                ft = SETTING_TO_FT.get(row["setting"])
                if ft is not None:
                    ft_attempts.setdefault(ft, set()).add(row["setting"])

            candidates = []
            for r in RUNS:
                name = setting_name(r)
                pct = pct_done(r)
                if pct >= 100:
                    continue
                if last_status.get(name) in ("started", "checkpoint"):
                    continue  # actively claimed by someone else (checkpoint = still running)
                ft_used = len(ft_attempts.get(r["ft"], ()))
                candidates.append((pct, ft_used, RUNS.index(r), name, r))

            if not candidates:
                return None

            # min-heap by pct, then prefer ft types with the fewest settings claimed so far
            # (spreads workers across ft2/ft15/full instead of exhausting one type first),
            # then matrix order as final tie-break
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            pct, _, _, name, run = candidates[0]

            append_state_row({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hostname": hostname, "setting": name,
                "status": "started", "pct_at_claim": pct,
            })
            return run
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def watch_checkpoints(hostname, run, stop_event, poll_seconds=20):
    name = setting_name(run)
    seen_step = current_step(name)
    while not stop_event.wait(poll_seconds):
        step = current_step(name)
        if step > seen_step:
            seen_step = step
            append_state(hostname, name, "checkpoint", pct_done(run))


def run_training(hostname, run):
    name = setting_name(run)
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"{name}.log"

    cmd = [
        "torchrun", f"--nproc_per_node={NPROC_PER_NODE}",
        str(REPO_ROOT / "scripts" / "train_multigpu.py"),
        "--model", run["model"], "--ft", run["ft"],
        "--batch-size", str(run["batch_size"]),
        "--grad-accum", str(run["grad_accum"]),
        "--max-steps", str(run["max_steps"]),
        "--optim", run["optim"],
    ]
    if run["nob"]:
        cmd.append("--no-boundary-tokens")

    print(f"[orchestrator] launching {name}: {' '.join(cmd)}")
    print(f"[orchestrator] log: {log_path}")

    stop_event = threading.Event()
    watcher = threading.Thread(target=watch_checkpoints, args=(hostname, run, stop_event), daemon=True)
    watcher.start()
    try:
        shell_cmd = "set -o pipefail; " + " ".join(cmd) + f" 2>&1 | tee -a {log_path}"
        proc = subprocess.run(["bash", "-c", shell_cmd], cwd=str(REPO_ROOT))
        return proc.returncode == 0
    finally:
        stop_event.set()
        watcher.join()


def main():
    hostname = socket.gethostname()
    while True:
        run = claim_next(hostname)
        if run is None:
            print("[orchestrator] no incomplete, unclaimed settings left — exiting")
            break

        name = setting_name(run)
        print(f"[orchestrator] {hostname} claimed {name}")
        try:
            ok = run_training(hostname, run)
        except (KeyboardInterrupt, Exception):
            # mark it failed even on Ctrl+C / unexpected crash so the "started" row
            # doesn't dangle and block re-claiming — see manual crash recovery note above
            append_state(hostname, name, "failed", pct_done(run))
            print(f"[orchestrator] {name} interrupted/crashed — marked failed")
            raise
        status = "finished" if ok else "failed"
        append_state(hostname, name, status, pct_done(run))
        print(f"[orchestrator] {name} -> {status}")

        if not ok:
            print(f"[orchestrator] {name} failed — check logs/{name}.log, exiting worker")
            break


if __name__ == "__main__":
    main()
