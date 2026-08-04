#!/usr/bin/env python3
"""qualkit-action runner: executes the assurance checks defined by a user
config module and gates CI on the results.

The config module (user's repo) must define:
    get_reference_model() -> torch.nn.Module      # FP32 / pre-compression
    get_deploy_model()    -> torch.nn.Module      # the artifact to ship
    get_cal_loader()      -> DataLoader[(x, y)]   # labeled calibration data
    get_probe_loader()    -> DataLoader (optional, enables SEU + timing)

Gates (any failure -> exit 1):
    coverage : forecast coverage floor under stale calibration >= min-coverage-floor
    setsize  : mean recalibrated prediction-set size <= max-mean-setsize
"""
import argparse
import importlib.util
import json
import os
import sys


def load_config(path):
    spec = importlib.util.spec_from_file_location("qualkit_user_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--min-coverage-floor", type=float, default=0.85)
    ap.add_argument("--max-mean-setsize", type=float, default=1.5)
    ap.add_argument("--seu-bits", type=int, default=8)
    ap.add_argument("--out", default="qualkit_report")
    args = ap.parse_args()

    from qualkit import report

    cfg = load_config(args.config)
    ref = cfg.get_reference_model()
    dep = cfg.get_deploy_model()
    cal = cfg.get_cal_loader()
    probe = cfg.get_probe_loader() if hasattr(cfg, "get_probe_loader") else None

    r = report.run_all(ref, dep, cal, probe, alpha=args.alpha,
                       seu_bits=args.seu_bits)
    r["setsize_acceptance"]["threshold"] = args.max_mean_setsize
    r["setsize_acceptance"]["verdict"] = (
        "PASS" if r["setsize_acceptance"]["mean_set_size"] <= args.max_mean_setsize
        else "FAIL")

    with open(f"{args.out}.json", "w") as f:
        json.dump(r, f, indent=2)
    md = report.to_markdown(r)
    with open(f"{args.out}.md", "w") as f:
        f.write(md)

    failures = []
    floor = r["coverage_forecast"]["forecast_coverage_floor"]
    if floor < args.min_coverage_floor:
        failures.append(
            f"coverage gate: forecast floor {floor:.3f} < required "
            f"{args.min_coverage_floor:.3f} (TV distance "
            f"{r['coverage_forecast']['tv_distance']:.4f} - recalibrate on the "
            f"deployed artifact or reduce compression)")
    if r["setsize_acceptance"]["verdict"] == "FAIL":
        failures.append(
            f"set-size gate: mean set size "
            f"{r['setsize_acceptance']['mean_set_size']:.2f} > "
            f"{args.max_mean_setsize} (valid but uninformative at this "
            f"precision)")

    verdict = "PASS" if not failures else "FAIL"
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(f"# qualkit gate: {verdict}\n\n")
            for fail in failures:
                f.write(f"- FAILED {fail}\n")
            f.write("\n" + md + "\n")

    print(f"qualkit gate: {verdict}")
    for fail in failures:
        print(f"  FAILED {fail}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
