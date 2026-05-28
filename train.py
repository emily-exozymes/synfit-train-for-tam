#!/usr/bin/env python
"""
SynFit-Train: wraps SynFit's two-stage training (train_single_branches.py +
train_joint_shared_module.py) so it runs as one Tamarind job from a user CSV.

Strategy: SynFit's scripts hardcode the CSV path to
  dataset/multi_fitness_data/{protein_name}.csv
relative to the repo root, so we stage the user's CSV there before invoking
them as subprocesses. SynFit writes Stage-1 checkpoints to
  results/{protein}/{metric}/fold_{fold}/seed_{seed}/best_model.pth
and Stage-2 to
  joint_results/{protein}/fold_{fold}/seed_{seed}/best_model.pth
We zip everything to out/ at the end.
"""

import os
import re
import sys
import glob
import shutil
import subprocess
import pandas as pd

# ---------- Inputs ----------
PROTEIN     = re.sub(r"[^A-Za-z0-9_]", "_", os.environ.get("protein_name", "MY_PROTEIN").strip()) or "MY_PROTEIN"
NUM_FOLDS   = int(os.environ.get("num_folds", "1"))
SEED        = int(os.environ.get("seed", "42"))
EPOCHS_S1   = int(os.environ.get("max_epochs_stage1", "80"))
EPOCHS_S2   = int(os.environ.get("max_epochs_stage2", "60"))

NUM_FOLDS = max(1, min(NUM_FOLDS, 5))

# ---------- Locate input CSV ----------
csv_candidates = sorted(glob.glob("inputs/*.csv"))
if not csv_candidates:
    sys.exit("ERROR: no training CSV found in inputs/")
input_csv = csv_candidates[0]
print(f"Training CSV: {input_csv}")

df = pd.read_csv(input_csv)
required_cols = {"mutant", "mutated_sequence"}
missing = required_cols - set(df.columns)
if missing:
    sys.exit(f"ERROR: CSV missing required columns: {missing}")
dms_cols = [c for c in df.columns if c.startswith("DMS_score_")]
if len(dms_cols) != 2:
    sys.exit(
        f"ERROR: SynFit's joint model is hardcoded to 2 heads, but the CSV has "
        f"{len(dms_cols)} DMS_score_* columns ({dms_cols}). Provide exactly 2."
    )
print(f"Found {len(df)} variants and {len(dms_cols)} objectives: {dms_cols}")

# ---------- Stage the CSV where SynFit expects it ----------
# Repo root is /app at runtime. SynFit scripts look in dataset/multi_fitness_data/.
data_dir = "/app/dataset/multi_fitness_data"
os.makedirs(data_dir, exist_ok=True)
staged = os.path.join(data_dir, f"{PROTEIN}.csv")
shutil.copy(input_csv, staged)
print(f"Staged CSV at {staged}")

# ---------- GPU check ----------
import torch
if not torch.cuda.is_available():
    sys.exit("ERROR: no CUDA GPU available")
n_gpus = torch.cuda.device_count()
print(f"GPUs available: {n_gpus} | {torch.cuda.get_device_name(0)}")

# ---------- Patch SynFit's hardcoded epoch counts before running ----------
# SynFit's scripts use `for epoch in range(80):` and `range(60):` literally.
# We rewrite them in-place so the env-var settings take effect.
def patch_epochs(path: str, old: str, new_n: int):
    with open(path, "r") as f:
        src = f.read()
    patched = src.replace(old, f"for epoch in range({new_n}):", 1)
    if patched != src:
        with open(path, "w") as f:
            f.write(patched)
        print(f"Patched epochs in {path}: '{old.strip()}' -> range({new_n})")

patch_epochs("/app/SynFit/train_single_branches.py", "for epoch in range(80):", EPOCHS_S1)
patch_epochs("/app/SynFit/train_joint_shared_module.py", "for epoch in range(60):", EPOCHS_S2)

# ---------- Run Stage 1 then Stage 2 for each fold ----------
NUM_METRICS = len(dms_cols)
env = os.environ.copy()
env["PYTHONPATH"] = "/app:/app/SynFit:" + env.get("PYTHONPATH", "")
env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n_gpus))

for fold in range(NUM_FOLDS):
    print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")

    # Stage 1: one baseline per DMS_score column
    for idx in range(NUM_METRICS):
        print(f"\n--- Stage 1: fold={fold} baseline_idx={idx} ---")
        rc = subprocess.call(
            [
                "python", "/app/SynFit/train_single_branches.py",
                "--protein", PROTEIN,
                "--baseline_idx", str(idx),
                "--fold", str(fold),
                "--seed", str(SEED),
                "--device", "cuda:0",
            ],
            cwd="/app",
            env=env,
        )
        if rc != 0:
            sys.exit(f"ERROR: Stage 1 failed (fold={fold}, baseline_idx={idx}, rc={rc})")

    # Stage 2: joint
    print(f"\n--- Stage 2: fold={fold} ---")
    gpu_arg = ",".join(str(i) for i in range(n_gpus))
    rc = subprocess.call(
        [
            "python", "/app/SynFit/train_joint_shared_module.py",
            "--protein", PROTEIN,
            "--fold", str(fold),
            "--seed", str(SEED),
            "--gpus", gpu_arg,
        ],
        cwd="/app",
        env=env,
    )
    if rc != 0:
        sys.exit(f"ERROR: Stage 2 failed (fold={fold}, rc={rc})")

# ---------- Collect outputs ----------
os.makedirs("out", exist_ok=True)
print("\nCollecting outputs...")

bundle_root = f"out/synfit-train-{PROTEIN}"
os.makedirs(bundle_root, exist_ok=True)

# Copy Stage-1 + Stage-2 results into the bundle
for src_subdir in ("results", "joint_results"):
    src = f"/app/{src_subdir}"
    if os.path.isdir(src):
        dst = os.path.join(bundle_root, src_subdir)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"  copied {src} -> {dst}")
    else:
        print(f"  WARNING: {src} not found")

# Also surface the Stage-2 joint checkpoint(s) at the top level of out/
# for convenience - that's what SynFit-Predict consumes.
for fold in range(NUM_FOLDS):
    src = f"/app/joint_results/{PROTEIN}/fold_{fold}/seed_{SEED}/best_model.pth"
    if os.path.exists(src):
        dst = f"out/joint_model_fold{fold}.pth"
        shutil.copy(src, dst)
        print(f"  joint checkpoint: {dst}")
    else:
        print(f"  WARNING: expected joint checkpoint not found: {src}")

# Zip the full bundle
archive_path = shutil.make_archive(bundle_root, "zip", root_dir=bundle_root)
print(f"  zipped: {archive_path}")
# Remove the unzipped tree to keep the output listing tidy
shutil.rmtree(bundle_root)

print("\nDone.")
print("Outputs in out/:")
for p in sorted(os.listdir("out")):
    size = os.path.getsize(os.path.join("out", p))
    print(f"  {p}  ({size/1e6:.1f} MB)")
