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
# SynFit's utils.py (at /app/utils.py) computes the data dir as
# os.path.dirname(__file__) + '/multi_fitness_data/<protein>.csv'
# which resolves to /app/multi_fitness_data/, NOT /app/dataset/multi_fitness_data/.
# Stage to /app/multi_fitness_data/ so the GeneralMultiFitnessDataset finds it.
# Also stage a copy at the cwd-relative path as a belt-and-suspenders fallback.
data_dir = "/app/multi_fitness_data"
os.makedirs(data_dir, exist_ok=True)
staged = os.path.join(data_dir, f"{PROTEIN}.csv")
shutil.copy(input_csv, staged)
print(f"Staged CSV at {staged}")
# Also stage at cwd in case the search order changes upstream.
shutil.copy(input_csv, f"/app/{PROTEIN}.csv")

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
# Tamarind has a soft cap on output upload size. The Stage-1 baseline
# checkpoints are ~2.6 GB EACH (one per metric per fold), and the Stage-2
# joint checkpoint is another 2.6 GB. Bundling all of them blew past
# the upload limit and got the job flagged "Internal Error" even though
# training succeeded. So:
#   - Only the Stage-2 joint checkpoint(s) ship as deliverables (Predict
#     consumes these).
#   - The bundle zip ships ONLY the per-epoch result logs (.txt files),
#     not the checkpoints. Logs are tiny; useful for debugging/reporting.
#   - Stage-1 baselines are intermediates that can be regenerated.
os.makedirs("out", exist_ok=True)
print("\nCollecting outputs...")

bundle_root = f"out/synfit-train-{PROTEIN}-logs"
os.makedirs(bundle_root, exist_ok=True)

# Copy only the result.txt files (per-epoch Spearman logs), no checkpoints.
import fnmatch
for src_subdir in ("results", "joint_results"):
    src_root = f"/app/{src_subdir}"
    if not os.path.isdir(src_root):
        print(f"  WARNING: {src_root} not found")
        continue
    for dirpath, dirnames, filenames in os.walk(src_root):
        for fname in filenames:
            if fnmatch.fnmatch(fname, "*.txt"):
                src_file = os.path.join(dirpath, fname)
                rel = os.path.relpath(src_file, "/app")
                dst_file = os.path.join(bundle_root, rel)
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy(src_file, dst_file)
    print(f"  copied result logs from {src_root}")

# Surface the Stage-2 joint checkpoint(s) - this is what Predict consumes.
# The raw checkpoint is the full ESM2-650M state dict in fp32 (~2.6 GB),
# which is over Tamarind's per-file output-upload cap and gets the job
# flagged "Internal Error" even though training finished cleanly. Re-save
# in fp16 to halve the file to ~1.3 GB. Only floating-point tensors are
# cast; integer/bool buffers (position ids, masks) are left untouched. On
# reload, load_state_dict upcasts fp16 -> the model's fp32 params, so the
# keys are identical and SynFit-Predict loads it unchanged (only the stored
# precision differs, which is negligible for inference ranking).
for fold in range(NUM_FOLDS):
    src = f"/app/joint_results/{PROTEIN}/fold_{fold}/seed_{SEED}/best_model.pth"
    if os.path.exists(src):
        dst = f"out/joint_model_fold{fold}.pth"
        sd = torch.load(src, map_location="cpu")
        sd = {
            k: (v.half() if torch.is_tensor(v) and torch.is_floating_point(v) else v)
            for k, v in sd.items()
        }
        torch.save(sd, dst)
        print(f"  joint checkpoint (fp16): {dst}  ({os.path.getsize(dst)/1e9:.2f} GB)")
    else:
        print(f"  WARNING: expected joint checkpoint not found: {src}")

# Tiny zip of just the logs.
archive_path = shutil.make_archive(bundle_root, "zip", root_dir=bundle_root)
print(f"  zipped logs: {archive_path}  ({os.path.getsize(archive_path)/1e6:.2f} MB)")
shutil.rmtree(bundle_root)

print("\nDone.")
print("Outputs in out/:")
for p in sorted(os.listdir("out")):
    size = os.path.getsize(os.path.join("out", p))
    print(f"  {p}  ({size/1e6:.1f} MB)")
