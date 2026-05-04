from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "graph" / "outputs"


def _latest_dirs(prefix: str) -> list[Path]:
    return sorted(OUT.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True)


def _read_route_summary(route_dir: Path) -> tuple[list[dict], dict]:
    route_csv = route_dir / "route_summary.csv"
    route_json = route_dir / "run_summary.json"
    rows = pd.read_csv(route_csv).to_dict(orient="records") if route_csv.exists() else []
    meta = json.loads(route_json.read_text(encoding="utf-8")) if route_json.exists() else {}
    return rows, meta


def _format_compare(title: str, rows: list[dict], keys: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.append("No completed outputs found.")
        lines.append("")
        return lines
    df = pd.DataFrame(rows)
    keep = [key for key in keys if key in df.columns]
    lines.append(df[keep].to_csv(index=False).strip())
    lines.append("")
    return lines


def main() -> None:
    route_map = {
        "RouteA": _latest_dirs("routeA_current_topk_egat"),
        "RouteB": _latest_dirs("routeB_senior_exact_plus_logic_edges"),
        "RouteC": _latest_dirs("routeC_abnormal_reliability_gate"),
        "RouteD": _latest_dirs("routeD_tns_guided_logic_egat"),
    }

    all_rows = []
    route_payloads: dict[str, tuple[list[dict], dict]] = {}
    for route_name, dirs in route_map.items():
        if not dirs:
            continue
        rows, meta = _read_route_summary(dirs[0])
        route_payloads[route_name] = (rows, meta)
        all_rows.extend(rows)

    if not all_rows:
        print("No route outputs found.")
        return

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT / f"three_routes_summary_{ts}.csv"
    md_path = OUT / f"three_routes_summary_{ts}.md"
    df.to_csv(csv_path, index=False)

    lines = ["# Three Routes Summary", ""]
    lines.extend(
        _format_compare(
            "Route A: old current top-k vs current EGAT",
            route_payloads.get("RouteA", ([], {}))[0],
            ["route", "output_dir", "edge_set", "backbone", "relation_model", "AUC", "AP", "F1", "Recall", "Precision"],
        )
    )
    lines.extend(
        _format_compare(
            "Route B: SeniorBaseExact vs SeniorBaseExact + LogicAE_CB vs SeniorBaseExact + Full",
            route_payloads.get("RouteB", ([], {}))[0],
            ["route", "output_dir", "edge_set", "backbone", "relation_model", "AUC", "AP", "F1", "Recall", "Precision", "notes"],
        )
    )
    lines.extend(
        _format_compare(
            "Route C: abnormal reliability gate on Base_CB",
            route_payloads.get("RouteC", ([], {}))[0],
            [
                "route",
                "output_dir",
                "edge_set",
                "backbone",
                "relation_model",
                "use_abnormal_edge_weight",
                "use_abnormal_gate",
                "use_abnormal_value_gate",
                "use_abnormal_attention_bias",
                "abnormal_score_source",
                "AUC",
                "AP",
                "F1",
                "Recall",
                "Precision",
            ],
        )
    )
    lines.extend(
        _format_compare(
            "Route D: TNS-guided LogicAE on current EGAT",
            route_payloads.get("RouteD", ([], {}))[0],
            [
                "route",
                "output_dir",
                "edge_set",
                "backbone",
                "relation_model",
                "use_tns_guided_logic",
                "tns_phi_days",
                "tns_logic_mode",
                "tns_logic_lambda",
                "logic_tns_topk",
                "use_node_gat",
                "AUC",
                "AP",
                "F1",
                "Recall",
                "Precision",
            ],
        )
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
