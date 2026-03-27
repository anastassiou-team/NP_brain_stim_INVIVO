#!/usr/bin/env python3
"""
Regenerate every figure in the paper.

Usage
-----
    python run_all.py            # run all figure scripts sequentially
    python run_all.py fig2       # run only Figure 2
"""
import subprocess
import sys
import time

SCRIPTS = [
    ("Figure 2  – entrainment",           "scripts/fig2_entrainment.py"),
    ("Figure 3  – row 1 (scatter + bubble)",  "scripts/figure3_row1.py"),
    ("Figure 3  – row 2 (frequency bars)",    "scripts/figure3_row2.py"),
    ("Figure 3  – row 3 (waveform bars)",     "scripts/figure3_row3.py"),
    ("Figure S4 – MMR entrainment",       "scripts/figS4_mmr_entrainment.py"),
    ("Figure S  – VL vs MMR correlation", "scripts/figS_vl_mmr_corr.py"),
]


def run(label, script):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {script}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  -> {status}  ({elapsed:.1f}s)\n")
    return result.returncode


def main():
    # Optional filter: python run_all.py fig2 fig3
    filters = [a.lower() for a in sys.argv[1:]]

    results = {}
    for label, script in SCRIPTS:
        if filters and not any(f in script.lower() for f in filters):
            continue
        results[label] = run(label, script)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for label, rc in results.items():
        mark = "OK" if rc == 0 else "FAIL"
        print(f"  [{mark}]  {label}")
    print()

    if any(rc != 0 for rc in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
