from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REFERENCE_D1_DIR = PROJECT_ROOT / "graph" / "outputs" / "routeD_tns_guided_logic_egat_20260504_200855" / "D1_EGAT_Base_LogicAE_CB"
REPRO_PACK_DIR = PROJECT_ROOT / "graph" / "outputs" / "D1_REPRO_PACK"


def main() -> None:
    reference = json.loads((REFERENCE_D1_DIR / "run_summary.json").read_text(encoding="utf-8"))["best_graph_model"]
    repro_frozen = json.loads((REPRO_PACK_DIR / "config_frozen.json").read_text(encoding="utf-8")) if (REPRO_PACK_DIR / "config_frozen.json").exists() else None
    status = {
        "reference_auc": reference.get("auc"),
        "reference_ap": reference.get("ap"),
        "reference_f1": reference.get("f1"),
        "repro_pack_exists": bool(repro_frozen is not None),
        "smoke_gate_rule": {
            "abs_auc_delta_max": 0.003,
            "abs_ap_delta_max": 0.003,
        },
        "note": "This smoke file is a governance gate description plus artifact existence check. Full K0 execution should use the frozen D1_REPRO_PACK.",
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
