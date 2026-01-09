#!/usr/bin/env python3
"""
train_all.py

Run the full pipeline:
 - preprocess
 - forecasting
 - segmentation
 - churn
 - sentiment
 - recommender

Creates logs and a JSON summary in data/outputs/.
Place this file in the project root (next to README.md) and run:
    python train_all.py
"""
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime
import shlex

ROOT = Path(__file__).resolve().parents[0]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OUTPUTS = ROOT / "data" / "outputs"
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)

PIPELINE_LOG = LOGS / "pipeline_log.txt"

# Steps: (name, command list (for subprocess.run), expected_output_files list)
# Commands assume python is available in current venv
# We'll attempt to find model folder either at 'models' or 'src/models'
def find_models_folder():
    cand1 = ROOT / "models"
    cand2 = ROOT / "src" / "models"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None

models_folder = find_models_folder()
if models_folder is None:
    print("Warning: no models folder found at 'models/' or 'src/models/'. Please ensure model scripts exist.")
    # We'll still attempt to run src models where applicable
    models_folder = ROOT / "src" / "models"

print(f"Using models folder: {models_folder}")

steps = [
    ("preprocess", [sys.executable, str(ROOT / "src" / "preprocess.py")], [
        str(DATA_PROCESSED / "olist_full_orders.csv"),
        str(DATA_PROCESSED / "daily_revenue.csv"),
        str(DATA_PROCESSED / "customer_metrics.csv"),
    ]),
    ("forecasting", [sys.executable, str(models_folder / "forecasting_model.py")], [
        str(DATA_PROCESSED / "revenue_forecast_sarima.csv")
    ]),
    ("segmentation", [sys.executable, str(models_folder / "customer_segmentation.py")], [
        str(DATA_PROCESSED / "customer_segments.csv")
    ]),
    ("churn", [sys.executable, str(models_folder / "churn_model.py")], [
        str(DATA_PROCESSED / "customer_churn_predictions.csv")
    ]),
    ("sentiment", [sys.executable, str(models_folder / "sentiment_model.py")], [
        str(DATA_PROCESSED / "review_sentiments.csv"),
        str(DATA_PROCESSED / "sentiment_model_pipeline.joblib")
    ]),
    ("recommender", [sys.executable, str(models_folder / "recommender_model.py")], [
        str(DATA_PROCESSED / "product_recommendations.csv")
    ]),
]

def log(msg):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with PIPELINE_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_command(cmd, cwd=None, timeout=None):
    """Run a command list, return (returncode, stdout, stderr)."""
    try:
        log(f"RUN: {' '.join([shlex.quote(str(c)) for c in cmd])}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd, timeout=timeout)
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return 9999, "", f"Exception running command: {e}"

def check_files_exist(paths):
    missing = [p for p in paths if not Path(p).exists()]
    return missing

def main():
    summary = {
        "run_id": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "started_at": datetime.utcnow().isoformat(),
        "steps": []
    }

    for name, cmd, expected in steps:
        step_info = {
            "name": name,
            "command": cmd,
            "started_at": datetime.utcnow().isoformat(),
            "returncode": None,
            "stdout_snippet": None,
            "stderr_snippet": None,
            "missing_outputs": None,
            "finished_at": None
        }

        # If script file not present, skip with note
        cmd_path = Path(cmd[1]) if len(cmd) > 1 else None
        if cmd_path and not cmd_path.exists():
            msg = f"Script not found: {cmd_path} -- skipping step '{name}'"
            log(msg)
            step_info["returncode"] = -1
            step_info["stderr_snippet"] = msg
            step_info["finished_at"] = datetime.utcnow().isoformat()
            summary["steps"].append(step_info)
            continue

        rc, out, err = run_command(cmd)
        step_info["returncode"] = rc
        step_info["stdout_snippet"] = (out or "").strip()[:4000]
        step_info["stderr_snippet"] = (err or "").strip()[:4000]
        step_info["finished_at"] = datetime.utcnow().isoformat()

        # write full logs for this step to pipeline log
        log(f"STEP {name} finished with code {rc}")
        if out:
            log(f"STEP {name} STDOUT (first 400 chars):\n{out[:400]}")
        if err:
            log(f"STEP {name} STDERR (first 400 chars):\n{err[:400]}")

        # Check expected output files
        missing = check_files_exist(expected)
        step_info["missing_outputs"] = missing
        if missing:
            log(f"WARNING: step '{name}' missing expected outputs: {missing}")
        else:
            log(f"Step '{name}' produced expected outputs.")

        summary["steps"].append(step_info)

    summary["finished_at"] = datetime.utcnow().isoformat()
    # Save summary JSON
    out_file = DATA_OUTPUTS / f"train_summary_{summary['run_id']}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved pipeline summary to {out_file}")

    # Print a short summary to console
    print("\n=== PIPELINE SUMMARY ===")
    for s in summary["steps"]:
        status = "OK" if s["returncode"] == 0 and not s["missing_outputs"] else "ISSUE"
        print(f"- {s['name']}: {status} (rc={s['returncode']}) missing_outputs={len(s['missing_outputs']) if s['missing_outputs'] else 0}")

    print(f"\nDetailed JSON summary written to: {out_file}")
    print(f"Full logs appended to: {PIPELINE_LOG}")

if __name__ == "__main__":
    main()
