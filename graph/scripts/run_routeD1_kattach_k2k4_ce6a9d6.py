from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROUTE_RUNNER = PROJECT_ROOT / "graph" / "routes" / "routeK_topk" / "scripts" / "run_routeD1_kattach_k2k4_ce6a9d6.py"


def main() -> None:
    runpy.run_path(str(ROUTE_RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
