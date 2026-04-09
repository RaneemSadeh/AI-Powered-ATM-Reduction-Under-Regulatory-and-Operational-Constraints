"""
run.py -- CLI entry point for the ATM Network Optimisation pipeline.

Usage
-----
  python run.py                          # all scenarios, 50% default
  python run.py --arch sage --epochs 400
  python run.py --radius 3.0
"""
import sys, io
# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import argparse, os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

from data_pipeline   import load_and_preprocess
from graph_builder   import build_graph, graph_summary
from gnn_trainer     import train_and_score
from scenario_runner import run_all_scenarios, SCENARIOS


def main():
    parser = argparse.ArgumentParser(description="ATM Network Optimisation — Al-Safwa Bank")
    parser.add_argument("--arch",   default="gcn",  choices=["gcn", "sage"],
                        help="GNN architecture: gcn (default) or sage")
    parser.add_argument("--epochs", type=int, default=300,
                        help="GNN training epochs (default: 300)")
    parser.add_argument("--radius", type=float, default=5.0,
                        help="Graph edge radius in km (default: 5.0)")
    args = parser.parse_args()

    data_path = os.path.join(ROOT, "Sim_Data", "ATM_Simulated_Dataset_Safwa.xlsx")

    print("\n" + "═"*62)
    print("  🏦  Al-Safwa Bank — ATM Network Optimisation")
    print("═"*62)

    print("\n📂 Loading data …")
    _, df, df_norm, _ = load_and_preprocess(data_path)
    print(f"   ✓  {len(df)} ATMs across {df['Region'].nunique()} governorates")

    print(f"\n🕸️  Building spatial graph  (radius = {args.radius} km) …")
    graph = build_graph(df, radius_km=args.radius)
    print("  ", graph_summary(graph, df).replace("\n", "\n   "))

    print(f"\n🧠 Training {args.arch.upper()} GNN  ({args.epochs} epochs) …")
    utility_scores, _ = train_and_score(df, df_norm, graph,
                                        arch=args.arch, epochs=args.epochs, verbose=True)

    print("\n⚙️  Running ILP optimisation across all scenarios …")
    results = run_all_scenarios(df, utility_scores, graph)

    print("\n" + "═"*62)
    print("  📊  RESULTS SUMMARY")
    print("═"*62)

    for key, res in results.items():
        meta = SCENARIOS[key]
        status = "✅ Optimal" if res["feasible"] else "⚠️  Feasible"
        print(f"\n  {meta['label_en']}")
        print(f"   Status       : {status}")
        print(f"   ATMs kept    : {res['n_kept']} / {len(df)}")
        print(f"   Coverage     : {res['coverage_pct']:.1f} %")
        print(f"   Revenue/day  : {res['revenue_retained']:,.0f} JOD")
        print(f"   Customers    : {res['customers_served']:,}")
        print(f"   Kept         : {', '.join(res['kept_names'])}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir = os.path.join(ROOT, "outputs")
    summary = {}
    for key, res in results.items():
        summary[key] = {
            "scenario":         SCENARIOS[key]["label_en"],
            "atms_kept":        res["n_kept"],
            "coverage_pct":     res["coverage_pct"],
            "revenue_jod_day":  res["revenue_retained"],
            "customers_served": res["customers_served"],
            "kept_atms":        res["kept_names"],
            "removed_atms":     res["removed_names"],
        }

    with open(os.path.join(output_dir, "scenario_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    utility_dict = {
        df.iloc[i]["Branch Name"]: round(float(utility_scores[i]), 4)
        for i in range(len(df))
    }
    with open(os.path.join(output_dir, "utility_scores.json"), "w",
              encoding="utf-8") as f:
        json.dump(utility_dict, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Results saved to: {output_dir}")
    print("\n🌐 Launch dashboard:  streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
