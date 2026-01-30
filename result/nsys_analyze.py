#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------
def parse_int(x: str) -> int:
    return int(x.replace(",", "").strip())

def parse_float(x: str) -> float:
    return float(x.replace(",", "").strip())

def gemm_flops(M, N, K):
    # standard GEMM FLOPs
    return 2.0 * float(M) * float(N) * float(K)

# Parse your benchmark table row:
# 32768  32768   4096 |  128  256   64       8 |     8      3    1   0    0       0 |    10       2   12.3655   711.34
BENCH_ROW_RE = re.compile(
    r"^\s*(?P<M>\d+)\s+(?P<N>\d+)\s+(?P<K>\d+)\s*\|\s*"
    r"(?P<BM>\d+)\s+(?P<BN>\d+)\s+(?P<BK>\d+)\s+(?P<GM>\d+)\s*\|\s*"
    r"(?P<WARP>\d+)\s+(?P<STG>\d+)\s+(?P<CTA>\d+)\s+(?P<WS>\d+)\s+(?P<FLATTEN>\d+)\s+(?P<SUB>\d+)\s*\|\s*"
    r"(?P<iters>\d+)\s+(?P<warmup>\d+)\s+(?P<ms>[\d.]+)\s+(?P<tflops>[\d.]+)\s*$"
)

# Find the nsys section header
NSYS_SECTION_START_RE = re.compile(r"^\s*\[\d+/\d+\]\s+Executing\s+'cuda_gpu_kern_sum'\s+stats\s+report\s*$")

# Parse nsys cuda_gpu_kern_sum row (only matmul kernels)
# 25.1  277,197,278  24  11,549,886.6  11,707,847.0  10,613,427  12,413,344  698,511.8  matmul_kernel_tma__BM128...
NSYS_ROW_RE = re.compile(
    r"^\s*(?P<time_pct>[\d.]+)\s+"
    r"(?P<total_ns>[\d,]+)\s+"
    r"(?P<instances>\d+)\s+"
    r"(?P<avg_ns>[\d,\.]+)\s+"
    r"(?P<med_ns>[\d,\.]+)\s+"
    r"(?P<min_ns>[\d,]+)\s+"
    r"(?P<max_ns>[\d,]+)\s+"
    r"(?P<std_ns>[\d,\.]+)\s+"
    r"(?P<name>matmul_kernel_tma__\S+)\s*$"
)

# Parse params from kernel name
KERNEL_PARAM_RE = re.compile(
    r"BM(?P<BM>\d+)_BN(?P<BN>\d+)_BK(?P<BK>\d+)_GM(?P<GM>\d+)"
    r"_WARP(?P<WARP>\d+)_STG(?P<STG>\d+)_CTA(?P<CTA>\d+)"
    r"_WS(?P<WS>\d+)_FLATTEN(?P<FLATTEN>\d+)_SUB(?P<SUB>\d+)"
)

KEY_COLS = ["BM","BN","BK","GM","WARP","STG","CTA","WS","FLATTEN","SUB"]

# -----------------------------
# Parsing
# -----------------------------
def parse_bench_table(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        m = BENCH_ROW_RE.match(line)
        if not m:
            continue
        d = {k: int(m.group(k)) for k in ["M","N","K","BM","BN","BK","GM","WARP","STG","CTA","WS","FLATTEN","SUB","iters","warmup"]}
        d["bench_ms"] = float(m.group("ms"))
        d["bench_tflops"] = float(m.group("tflops"))
        rows.append(d)
    return pd.DataFrame(rows)

def parse_nsys_cuda_gpu_kern_sum(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    in_sec = False
    rows = []
    for i, line in enumerate(lines):
        if NSYS_SECTION_START_RE.match(line):
            in_sec = True
            continue
        if in_sec:
            # Stop when next [x/y] section begins
            if re.match(r"^\s*\[\d+/\d+\]\s+Executing\s+'", line):
                break
            m = NSYS_ROW_RE.match(line)
            if not m:
                continue
            name = m.group("name")
            pm = KERNEL_PARAM_RE.search(name)
            if not pm:
                continue
            d = {k: int(pm.group(k)) for k in KEY_COLS}
            d.update({
                "nsys_time_pct": float(m.group("time_pct")),
                "nsys_total_ns": parse_int(m.group("total_ns")),
                "nsys_instances": int(m.group("instances")),
                "nsys_avg_ns": parse_float(m.group("avg_ns")),
                "nsys_med_ns": parse_float(m.group("med_ns")),
                "nsys_min_ns": parse_int(m.group("min_ns")),
                "nsys_max_ns": parse_int(m.group("max_ns")),
                "nsys_std_ns": parse_float(m.group("std_ns")),
                "kernel_name": name,
            })
            rows.append(d)
    return pd.DataFrame(rows)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_txt", help="the whole stdout text you pasted (saved to a file)")
    ap.add_argument("--csv", default=None, help="output CSV path")
    args = ap.parse_args()
    if args.csv is None:
        args.csv = args.out_txt + ".csv"
    text = Path(args.out_txt).read_text(errors="ignore")

    df_bench = parse_bench_table(text)
    df_nsys = parse_nsys_cuda_gpu_kern_sum(text)

    if df_bench.empty:
        raise SystemExit("ERROR: did not find benchmark table rows (M N K | ...).")
    if df_nsys.empty:
        raise SystemExit("ERROR: did not find cuda_gpu_kern_sum matmul rows.")

    # Assume one problem size (like your example). If multiple, we keep M/N/K per bench row.
    # For TFLOPs from nsys, we need M/N/K. We'll join nsys to bench first; if missing, we'll fallback to first M/N/K.
    df = pd.merge(df_bench, df_nsys, on=KEY_COLS, how="outer", indicator=True)

    # Fallback M/N/K if some kernels appear only in nsys (e.g., extra STG variants):
    # pick first seen M/N/K from bench
    M0, N0, K0 = int(df_bench.iloc[0]["M"]), int(df_bench.iloc[0]["N"]), int(df_bench.iloc[0]["K"])
    df["M"] = df["M"].fillna(M0).astype(int)
    df["N"] = df["N"].fillna(N0).astype(int)
    df["K"] = df["K"].fillna(K0).astype(int)

    flops = gemm_flops(M0, N0, K0)

    # compute nsys TFLOPs (avg/med)
    df["nsys_avg_ms"] = df["nsys_avg_ns"] / 1e6
    df["nsys_med_ms"] = df["nsys_med_ns"] / 1e6
    df["nsys_avg_tflops"] = (flops / (df["nsys_avg_ns"] * 1e-9)) / 1e12
    df["nsys_med_tflops"] = (flops / (df["nsys_med_ns"] * 1e-9)) / 1e12
    df["nsys_cv"] = df["nsys_std_ns"] / df["nsys_avg_ns"]

    # nice ordering
    show_cols = KEY_COLS + [
        "bench_ms","bench_tflops",
        "nsys_instances","nsys_avg_ms","nsys_med_ms","nsys_avg_tflops","nsys_med_tflops","nsys_cv",
        "_merge"
        # ,"kernel_name"
    ]
    for c in show_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # sort by nsys_med_tflops desc if available, else bench_tflops desc
    df_sorted = df.sort_values(
        by=["nsys_med_tflops","bench_tflops"],
        ascending=[False, False],
        na_position="last"
    )

    # print a readable table
    pd.set_option("display.max_colwidth", 120)
    print(f"Parsed problem size (fallback): M={M0}, N={N0}, K={K0}")
    print(df_sorted[show_cols].to_string(index=False))

    # write csv
    df_sorted.to_csv(args.csv, index=False)
    print(f"\nWrote: {args.csv}")

if __name__ == "__main__":
    main()
