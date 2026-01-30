import os
BEST = {}  # key -> dict(ms=..., params=...)

def record_best(key: str, ms: float, params: dict):
    cur = BEST.get(key)
    if cur is None or ms < cur["ms"]:
        BEST[key] = {"ms": ms, "params": params}

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not BEST:
        return
    terminalreporter.write("\n\n===== Triton best configs =====\n")
    for key, v in sorted(BEST.items()):
        terminalreporter.write(f"{key}: {v['ms']:.3f} ms  params={v['params']}\n")