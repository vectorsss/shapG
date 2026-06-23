"""Config loading, result saving and loading."""

import json

import pandas as pd
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_previous_results(output_dir):
    """Load explainer results and KPI caches from a previous run."""
    results = {}
    sv_path = output_dir / "shapley_values.csv"
    timing_path = output_dir / "timing.json"

    if not sv_path.exists() or not timing_path.exists():
        return results, {}, {}, None

    df = pd.read_csv(sv_path)
    with open(timing_path) as f:
        timing = json.load(f)

    for method in df["method"].unique():
        sub = df[df["method"] == method]
        values = {}
        for _, row in sub.iterrows():
            node = row["node"]
            try:
                node = int(node)
            except (ValueError, TypeError):
                pass
            values[node] = row["shapley_value"]
        results[method] = {
            "values": values,
            "time": timing.get(method, 0.0),
            "explainer": None,  # no live explainer instance
        }

    # Load cached KPI results
    prev_bwd = {}
    bwd_path = output_dir / "kpi_backward.json"
    if bwd_path.exists():
        with open(bwd_path) as f:
            prev_bwd = json.load(f)

    prev_fwd = {}
    prev_full_metric = None
    fwd_path = output_dir / "kpi_forward.json"
    if fwd_path.exists():
        with open(fwd_path) as f:
            prev_fwd = json.load(f)
        meta = prev_fwd.pop("_meta", {})
        prev_full_metric = meta.get("full_metric")

    print(f"  Loaded {len(results)} methods from {output_dir}")
    return results, prev_bwd, prev_fwd, prev_full_metric


def save_results(
    output_dir,
    explainer_results,
    kpi_results,
    forward_kpi_results,
    cfg,
    full_metric=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Shapley values + timing ---
    rows = []
    for label, res in explainer_results.items():
        for node, val in res["values"].items():
            rows.append({"method": label, "node": node, "shapley_value": val})
    pd.DataFrame(rows).to_csv(output_dir / "shapley_values.csv", index=False)

    timing = {label: res["time"] for label, res in explainer_results.items()}
    with open(output_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    # --- Backward KPI ---
    if kpi_results:
        kpi_ser = {}
        for method, data in kpi_results.items():
            kpi_ser[method] = {
                "metrics": [float(m) for m in data["metrics"]],
                "slope": data["slope"],
                "features_dropped": data["features_dropped"],
            }
        with open(output_dir / "kpi_backward.json", "w") as f:
            json.dump(kpi_ser, f, indent=2)

    # --- Forward KPI ---
    if forward_kpi_results:
        fwd_ser = {"_meta": {}}
        if full_metric is not None:
            fwd_ser["_meta"]["full_metric"] = float(full_metric)
        for method, data in forward_kpi_results.items():
            entry = {
                "metrics": [float(m) for m in data["metrics"]],
                "slope": data["slope"],
            }
            if "time" in data:
                entry["time"] = float(data["time"])
            fwd_ser[method] = entry
        with open(output_dir / "kpi_forward.json", "w") as f:
            json.dump(fwd_ser, f, indent=2)

    # --- Copy config ---
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\nResults saved to {output_dir}/")
