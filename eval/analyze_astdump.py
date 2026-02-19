import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

IN_PATH = os.path.join(REPO_ROOT, "data", "processed", "astdump.jsonl")
MAX_SEQ_LEN = 2048
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def approx_token_len(text: str) -> int:
    # Fast approximation: split into word-like chunks and punctuation symbols.
    return len(TOKEN_RE.findall(text))


def percentile(sorted_values, p: float) -> float:
    if not sorted_values:
        return 0.0

    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])

    k = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return float(sorted_values[lo])

    frac = k - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def main():
    lengths = []
    skipped = 0

    with open(IN_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                skipped += 1
                continue

            obj = json.loads(line)
            dump_str = obj.get("ast_dump", "")
            if not dump_str:
                skipped += 1
                continue

            lengths.append(approx_token_len(dump_str))

            if i % 50000 == 0:
                print(f"[INFO] Processed {i:,} lines...")

    if not lengths:
        print("[WARN] No valid samples found.")
        return

    sorted_lengths = sorted(lengths)
    samples = len(sorted_lengths)
    mean_len = sum(sorted_lengths) / samples

    print("========== AST DUMP APPROX TOKEN LENGTH STATS ==========")
    print("Samples:", samples)
    print("Skipped:", skipped)
    print("Min:", sorted_lengths[0])
    print("Max:", sorted_lengths[-1])
    print("Mean:", round(mean_len, 2))
    print("Median:", round(percentile(sorted_lengths, 50), 2))
    print("P90:", round(percentile(sorted_lengths, 90), 2))
    print("P95:", round(percentile(sorted_lengths, 95), 2))
    print("P99:", round(percentile(sorted_lengths, 99), 2))

    waste = [max(0, MAX_SEQ_LEN - x) for x in sorted_lengths]
    used = [min(MAX_SEQ_LEN, x) for x in sorted_lengths]
    avg_waste = sum(waste) / samples
    avg_util = (sum(used) / samples) / MAX_SEQ_LEN * 100.0

    print("\n========== Padding Waste if using 2048 fixed ==========")
    print("Avg wasted tokens:", round(avg_waste, 2))
    print("Avg utilization:", f"{avg_util:.2f}%")

if __name__ == "__main__":
    main()
