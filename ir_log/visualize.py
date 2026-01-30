#!/usr/bin/env python3
"""
Convert all *.mlir files in the current directory to:
  - <name>.dot (from triton-opt -view-op-graph stderr)
  - <name>.png (rendered by graphviz dot)

Example (what we run per file):
  triton-opt before.mlir -allow-unregistered-dialect -view-op-graph -o /dev/null 2> before.dot
  dot -Tpng before.dot -o before.png
"""

import subprocess
from pathlib import Path
import shutil
import sys
import os

TRITON_OPT = "triton-opt"  # or set to an absolute path if not in PATH
DOT_BIN = "dot"            # graphviz

def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{p.stdout}\n"
            f"--- stderr ---\n{p.stderr}\n"
        )

def main() -> int:
    here = Path(".").resolve()

    if shutil.which(TRITON_OPT) is None:
        print(f"[ERROR] '{TRITON_OPT}' not found in PATH. "
              f"Set TRITON_OPT to an absolute path or add it to PATH.",
              file=sys.stderr)
        return 2

    if shutil.which(DOT_BIN) is None:
        print(f"[ERROR] '{DOT_BIN}' (Graphviz) not found in PATH. "
              f"Install graphviz or add 'dot' to PATH.",
              file=sys.stderr)
        return 3

    mlir_files = sorted(here.glob("*.mlir"))
    if not mlir_files:
        print("[INFO] No .mlir files found in current directory.")
        return 0
    cwd = os.getcwd()
    os.makedirs(os.path.join(cwd, "dot"), exist_ok=True)
    os.makedirs(os.path.join(cwd, "png"), exist_ok=True)
    for mlir in mlir_files:
        base = mlir.with_suffix("")  # remove .mlir
        base_name = base.name
        dot_path = os.path.join(cwd, "dot", base_name + ".dot")
        png_path = os.path.join(cwd, "png", base_name + ".png")

        print(dot_path, png_path)
        # assert 0
        print(f"[*] {mlir.name} -> {dot_path}, {png_path}")

        # 1) Generate .dot from triton-opt stderr
        proc = subprocess.run(
            [TRITON_OPT, str(mlir),
             "-allow-unregistered-dialect",
             "-view-op-graph",
             "-o", "/dev/null"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            print(f"[ERROR] triton-opt failed on {mlir.name}\n{proc.stderr}",
                  file=sys.stderr)
            return proc.returncode

        with open(dot_path, "w", encoding="utf-8") as f:
            f.write(proc.stderr)
        # 2) Render .png using graphviz
        run([DOT_BIN, "-Tpng", str(dot_path), "-o", str(png_path)])

    print("[DONE] Converted all .mlir files.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
