# ============================================================
# INDRA LightGBM Classifier v3.8 — VSCode / Local
# Changes from v3.7:
#   - Two-tier training: hard species get
#     n_estimators=600, lr=0.03, pos_weight cap=80
#   - Normal species: 400 trees, lr=0.05, cap=40
#   - HARD_SPECIES_KW matches generator v3.8
#   - Output files named v38
# ============================================================

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, average_precision_score,
                             precision_recall_curve)
import joblib, time, re, psutil
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════
INDRA_DIR    = "c:/Users/sahil/Downloads/"
X_FILE       = INDRA_DIR + "indra_X_v38_135k_20260404_0525.npy"
Y_FILE       = INDRA_DIR + "indra_Y_v38_135k_20260404_0525.npy"
LABEL_FILE   = INDRA_DIR + "indra_labels_v38_135k_20260404_0525.csv"
CLUSTER_FILE = INDRA_DIR + "indra_cluster_map_v38_135k_20260404_0525.csv"
SAVE_DIR     = INDRA_DIR

PRESENCE_THRESHOLD = 0.01
RECALL_TARGET      = 0.90

HARD_SPECIES_KW = [
    'SO2', 'H2CO', 'CH2O', 'CH3OCHO',
    'CH3CH2OH', 'C2H5OH', 'OCN',
    'CH3COOH', 'HCOOH', 'CH3CHO',
]

pipeline_start = time.time()
def elapsed():
    return (time.time() - pipeline_start) / 60

# ══════════════════════════════════════════════════════════
# 0. SYSTEM CHECK
# ══════════════════════════════════════════════════════════
ram_available = psutil.virtual_memory().available / 1e9
ram_total     = psutil.virtual_memory().total / 1e9
cpu_threads   = psutil.cpu_count(logical=True)
LGBM_JOBS     = min(cpu_threads, 8)

print("="*60)
print("  INDRA v3.8 — LightGBM — Two-tier Hard Species")
print("="*60)
print(f"\n  RAM available : {ram_available:.1f} GB / "
      f"{ram_total:.1f} GB")
print(f"  CPU threads   : {cpu_threads}")
print(f"  LightGBM jobs : {LGBM_JOBS}")
print(f"  Recall target : {RECALL_TARGET*100:.0f}%")
print(f"\n  Hard species  : {len(HARD_SPECIES_KW)} keywords")
print(f"    → 600 trees, lr=0.03, cap=80")
print(f"  Normal species:")
print(f"    → 400 trees, lr=0.05, cap=40")

if ram_available < 5.0:
    print(f"  ⚠  Low RAM ({ram_available:.1f} GB) "
          f"— close other apps")
else:
    print(f"  ✔  RAM sufficient")

# ══════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════
print("\n[1/7] Loading data...")
X_raw  = np.load(X_FILE,  mmap_mode='r').astype(np.float32)
Y_data = np.load(Y_FILE,  mmap_mode='r').astype(np.float32)

labels_df = pd.read_csv(LABEL_FILE)
labels    = labels_df['label'].astype(str).tolist()
n_species = len(labels)

print(f"  X raw    : {X_raw.shape}")
print(f"  Y        : {Y_data.shape}")
print(f"  Species  : {n_species} cluster labels")
print(f"  X range  : {X_raw.min():.3f}–{X_raw.max():.3f}"
      f"  (should be 0–6.2 tau)")

# ══════════════════════════════════════════════════════════
# 2. PARSE Y
# ══════════════════════════════════════════════════════════
print("\n[2/7] Parsing Y matrix...")
Y_frac     = Y_data[:, 1 : n_species + 1]
frac_names = labels
Y_binary   = (Y_frac > PRESENCE_THRESHOLD).astype(np.int8)
presence_rate = Y_binary.mean(axis=0)
active_per    = Y_binary.sum(axis=1)

print(f"  Species              : {n_species}")
print(f"  Overall presence rate: {Y_binary.mean()*100:.1f}%")
print(f"  Active per spectrum  : "
      f"mean={active_per.mean():.1f}  "
      f"min={active_per.min()}  "
      f"max={active_per.max()}")

def is_hard(name):
    n = name.upper()
    return any(kw.upper() in n for kw in HARD_SPECIES_KW)

hard_indices = [i for i in range(n_species)
                if is_hard(frac_names[i])]
print(f"\n  Hard species in label space: {len(hard_indices)}")
print(f"\n  Hard species presence rates:")
for i in hard_indices:
    rate = Y_binary[:, i].mean() * 100
    ok   = "✔" if rate > 3.0 else "⚠ low"
    print(f"    {ok} {frac_names[i][:50]:<50}: {rate:.1f}%")

# ══════════════════════════════════════════════════════════
# 3. TAU — DIRECT, NO TRANSFORM
# ══════════════════════════════════════════════════════════
print("\n[3/7] Preparing tau input (direct, no transform)...")
X_tau = np.maximum(X_raw, 0.0).astype(np.float32)
assert np.all(np.isfinite(X_tau)), "Non-finite in X_tau"
print(f"  τ range : {X_tau.min():.3f} → {X_tau.max():.3f}")
print(f"  No log-transform applied  ✔")

del X_raw, Y_data
ram_now = psutil.virtual_memory().available / 1e9
print(f"  RAM after load: {ram_now:.1f} GB free")

# ══════════════════════════════════════════════════════════
# 4. SPLIT 80 / 10 val / 10 test
# ══════════════════════════════════════════════════════════
print("\n[4/7] Splitting 80 / 10 / 10...")
X_trainval, X_te, Yb_trainval, Yb_te = train_test_split(
    X_tau, Y_binary, test_size=0.10, random_state=42)
X_tr, X_val, Yb_tr, Yb_val = train_test_split(
    X_trainval, Yb_trainval,
    test_size=0.111, random_state=42)

print(f"  Train : {X_tr.shape[0]:,}")
print(f"  Val   : {X_val.shape[0]:,}")
print(f"  Test  : {X_te.shape[0]:,}")

del X_tau, Y_binary, X_trainval, Yb_trainval
ram_now = psutil.virtual_memory().available / 1e9
print(f"  RAM after split: {ram_now:.1f} GB free")

# ══════════════════════════════════════════════════════════
# 5. TRAIN — two-tier by species difficulty
# ══════════════════════════════════════════════════════════
print(f"\n[5/7] Training {n_species} classifiers "
      f"(two-tier)...")
print(f"  {X_tr.shape[0]:,} spectra × "
      f"{X_tr.shape[1]:,} features")

def make_lgbm(pos_weight, hard=False):
    return LGBMClassifier(
        n_estimators     = 600  if hard else 400,
        max_depth        = 7    if hard else 6,
        learning_rate    = 0.03 if hard else 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.7  if hard else 0.6,
        scale_pos_weight = pos_weight,
        min_child_samples= 5    if hard else 10,
        n_jobs           = LGBM_JOBS,
        random_state     = 42,
        verbose          = -1,
    )

def train_one(i, X_train, y_train):
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    hard  = is_hard(frac_names[i])
    if n_pos == 0:
        pos_weight = 15.0
    else:
        cap        = 80.0 if hard else 40.0
        pos_weight = float(
            np.clip((n_neg / n_pos) * 0.5, 5.0, cap))
    clf = make_lgbm(pos_weight, hard=hard)
    clf.fit(X_train, y_train)
    return clf

t0             = time.time()
classifiers    = []
print_every    = 20
n_hard_trained = 0

for i in range(n_species):
    clf = train_one(i, X_tr, Yb_tr[:, i])
    classifiers.append(clf)
    if is_hard(frac_names[i]):
        n_hard_trained += 1

    if (i + 1) % print_every == 0 or i == n_species - 1:
        elapsed_min = (time.time() - t0) / 60
        rate        = (i + 1) / max(elapsed_min, 0.01)
        eta         = (n_species - i - 1) / max(rate, 0.01)
        ram_now     = psutil.virtual_memory().available / 1e9
        tier        = "★hard" if is_hard(frac_names[i]) \
                      else " norm"
        print(f"  [{i+1:>3}/{n_species}] {tier}  "
              f"{elapsed_min:5.1f} min  "
              f"~{eta:4.0f} min rem  "
              f"RAM {ram_now:.1f}GB")

train_elapsed = (time.time() - t0) / 60
print(f"\n  ✔ Trained {n_species} classifiers "
      f"in {train_elapsed:.1f} min")
print(f"  Hard species trained : {n_hard_trained}")
print(f"  Normal species       : "
      f"{n_species - n_hard_trained}")

joblib.dump(classifiers,
            SAVE_DIR + "lgbm_classifiers_v38.pkl")
joblib.dump(frac_names,
            SAVE_DIR + "lgbm_species_names_v38.pkl")
print(f"  💾 lgbm_classifiers_v38.pkl")
print(f"  💾 lgbm_species_names_v38.pkl")

# ══════════════════════════════════════════════════════════
# 6. RECALL-FIRST THRESHOLD TUNING
# ══════════════════════════════════════════════════════════
print(f"\n[6/7] Recall-first threshold tuning "
      f"(target {RECALL_TARGET*100:.0f}%)...")

Y_val_proba = np.column_stack([
    clf.predict_proba(X_val)[:, 1]
    for clf in classifiers])

thresholds       = np.full(n_species, 0.5)
achieved_recalls = np.zeros(n_species)
achieved_precs   = np.zeros(n_species)
unreachable      = 0

for i in range(n_species):
    y_true  = Yb_val[:, i].astype(int)
    y_score = Y_val_proba[:, i]
    if y_true.sum() == 0:
        thresholds[i] = 0.5
        continue
    precs, recs, thresh_vals = precision_recall_curve(
        y_true, y_score)
    valid = recs[:-1] >= RECALL_TARGET
    if valid.any():
        thresholds[i] = float(thresh_vals[valid][-1])
    else:
        thresholds[i] = float(thresh_vals[0])
        unreachable  += 1
    pred = (y_score >= thresholds[i]).astype(int)
    achieved_recalls[i] = recall_score(
        y_true, pred, zero_division=0)
    achieved_precs[i]   = precision_score(
        y_true, pred, zero_division=0)

has_data   = Yb_val.sum(axis=0) > 0
y_val_flat = Yb_val[:, has_data].flatten().astype(int)
rec_after  = recall_score(
    y_val_flat,
    (Y_val_proba[:, has_data] >=
     thresholds[has_data]).flatten(),
    zero_division=0)

print(f"  Species with val examples : {has_data.sum()}")
print(f"  Unreachable               : {unreachable}")
print(f"  Mean achieved recall      : "
      f"{achieved_recalls[has_data].mean()*100:.1f}%")
print(f"  Mean achieved precision   : "
      f"{achieved_precs[has_data].mean()*100:.1f}%")
print(f"  Recall after tuning       : "
      f"{rec_after*100:.1f}%")

print(f"\n  Hard species recall after tuning:")
for i in hard_indices:
    if Yb_val[:, i].sum() > 0:
        ok = "✔" if achieved_recalls[i] >= 0.9 else "⚠"
        print(f"    {ok} {frac_names[i][:50]:<50}: "
              f"{achieved_recalls[i]*100:.1f}%")

joblib.dump(thresholds,
            SAVE_DIR + "lgbm_thresholds_recall90_v38.pkl")
print(f"\n  💾 lgbm_thresholds_recall90_v38.pkl")

# ══════════════════════════════════════════════════════════
# 7. EVALUATE ON TEST SET
# ══════════════════════════════════════════════════════════
print(f"\n[7/7] Evaluating on {X_te.shape[0]:,} "
      f"test spectra...")

Y_pred_proba_te = np.column_stack([
    clf.predict_proba(X_te)[:, 1]
    for clf in classifiers])

Y_pred_std   = (Y_pred_proba_te > 0.50).astype(int)
Y_pred_tuned = (Y_pred_proba_te >= thresholds).astype(int)

def report_metrics(y_true_mat, y_pred_mat,
                   y_score_mat, label):
    yt    = y_true_mat.flatten().astype(int)
    yp    = y_pred_mat.flatten()
    ys    = y_score_mat.flatten()
    prec  = precision_score(yt, yp, zero_division=0)
    rec   = recall_score(yt, yp, zero_division=0)
    f1    = f1_score(yt, yp, zero_division=0)
    prauc = average_precision_score(yt, ys)
    fp    = int(((yp==1)&(yt==0)).sum())
    tn    = int(((yp==0)&(yt==0)).sum())
    fpr   = fp / (fp + tn + 1e-10)
    avg_d = y_pred_mat.sum(axis=1).mean()
    print(f"\n  {'─'*50}")
    print(f"  {label}")
    print(f"  {'─'*50}")
    print(f"  PR-AUC    : {prauc*100:.1f}%")
    print(f"  Precision : {prec*100:.1f}%")
    print(f"  Recall    : {rec*100:.1f}%")
    print(f"  F1        : {f1*100:.1f}%")
    print(f"  FPR       : {fpr*100:.1f}%")
    print(f"  Avg predicted/spectrum : {avg_d:.1f}")
    return prauc, prec, rec, f1, fpr

print("\n" + "★"*60)
print("  OVERALL TEST RESULTS — v3.8")
print("★"*60)
report_metrics(Yb_te, Y_pred_std,
               Y_pred_proba_te, "Standard (0.50)")
report_metrics(Yb_te, Y_pred_tuned,
               Y_pred_proba_te,
               f"Recall-first ({RECALL_TARGET*100:.0f}%)")

# Per-species metrics
per_prauc        = []
per_recall_tuned = []
per_prec_tuned   = []
for i in range(n_species):
    yt  = Yb_te[:, i].astype(int)
    ys  = Y_pred_proba_te[:, i]
    ypt = Y_pred_tuned[:, i]
    if yt.sum() == 0:
        per_prauc.append(0.0)
        per_recall_tuned.append(0.0)
        per_prec_tuned.append(0.0)
    else:
        per_prauc.append(
            average_precision_score(yt, ys))
        per_recall_tuned.append(
            recall_score(yt, ypt, zero_division=0))
        per_prec_tuned.append(
            precision_score(yt, ypt, zero_division=0))

per_prauc        = np.array(per_prauc)
per_recall_tuned = np.array(per_recall_tuned)

print(f"\n{'═'*60}")
print(f"  PR-AUC DISTRIBUTION — {n_species} SPECIES")
print(f"{'═'*60}")
print(f"  Mean   : {per_prauc.mean()*100:.1f}%")
print(f"  Median : {np.median(per_prauc)*100:.1f}%")
print(f"  > 0.8  : {(per_prauc>0.8).sum():>4} / {n_species}")
print(f"  > 0.6  : {(per_prauc>0.6).sum():>4} / {n_species}")
print(f"  < 0.1  : {(per_prauc<0.1).sum():>4} / {n_species}")

print(f"\n  HARD SPECIES RESULTS (key improvement target):")
print(f"  {'Label':<50} {'Freq%':>5} "
      f"{'PRAUC':>6} {'Rec_tun':>8}")
print(f"  {'-'*50} {'-'*5} {'-'*6} {'-'*8}")
for i in hard_indices:
    freq = Yb_te[:, i].mean()
    ok   = "✔" if per_recall_tuned[i] >= 0.9 else "⚠"
    print(f"  {ok} {frac_names[i][:48]:<48} "
          f"{freq*100:4.1f}%  "
          f"{per_prauc[i]*100:5.1f}%  "
          f"{per_recall_tuned[i]*100:7.1f}%")

print(f"\n  TOP 10 BEST:")
print(f"  {'Label':<44} {'Freq%':>5} "
      f"{'PRAUC':>6} {'Rec':>6}")
print(f"  {'-'*44} {'-'*5} {'-'*6} {'-'*6}")
for i in np.argsort(per_prauc)[::-1][:10]:
    freq = Yb_te[:, i].mean()
    print(f"  {frac_names[i][:44]:<44} "
          f"{freq*100:4.1f}%  "
          f"{per_prauc[i]*100:5.1f}%  "
          f"{per_recall_tuned[i]*100:5.1f}%")

print(f"\n  BOTTOM 10 HARDEST:")
print(f"  {'Label':<44} {'Freq%':>5} "
      f"{'PRAUC':>6} {'Rec':>8}")
print(f"  {'-'*44} {'-'*5} {'-'*6} {'-'*8}")
for i in np.argsort(per_prauc)[:10]:
    freq = Yb_te[:, i].mean()
    print(f"  {frac_names[i][:44]:<44} "
          f"{freq*100:4.1f}%  "
          f"{per_prauc[i]*100:5.1f}%  "
          f"{per_recall_tuned[i]*100:7.1f}%")

# Version comparison
print(f"\n{'═'*60}")
print(f"  VERSION COMPARISON  v3.7 → v3.8")
print(f"{'═'*60}")
yt_f = Yb_te.flatten().astype(int)
yp_s = Y_pred_std.flatten()
yp_t = Y_pred_tuned.flatten()
ys_f = Y_pred_proba_te.flatten()
fp_s = int(((yp_s==1)&(yt_f==0)).sum())
tn_s = int(((yp_s==0)&(yt_f==0)).sum())
fp_t = int(((yp_t==1)&(yt_f==0)).sum())
tn_t = int(((yp_t==0)&(yt_f==0)).sum())

curr_prauc    = average_precision_score(yt_f, ys_f)*100
curr_rec_std  = recall_score(
    yt_f, yp_s, zero_division=0)*100
curr_rec_tun  = recall_score(
    yt_f, yp_t, zero_division=0)*100
curr_prec_std = precision_score(
    yt_f, yp_s, zero_division=0)*100
curr_fpr      = fp_t/(fp_t+tn_t+1e-10)*100

prev = {'PR-AUC':       63.6,
        'Precision':    57.4,
        'Recall std':   57.8,
        'Recall tuned': 89.6,
        'FPR':           2.0}
curr = {'PR-AUC':       curr_prauc,
        'Precision':    curr_prec_std,
        'Recall std':   curr_rec_std,
        'Recall tuned': curr_rec_tun,
        'FPR':          curr_fpr}

print(f"  {'Metric':<24} {'v3.7':>8} {'v3.8':>8}")
print(f"  {'-'*24} {'-'*8} {'-'*8}")
for metric in ['PR-AUC', 'Precision', 'Recall std',
               'Recall tuned', 'FPR']:
    delta = curr[metric] - prev[metric]
    arrow = "↑" if delta > 0 else "↓"
    print(f"  {metric:<24} "
          f"{prev[metric]:>7.1f}%  "
          f"{curr[metric]:>7.1f}%  "
          f"{arrow}{abs(delta):.1f}%")

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "★"*60)
print("  FINAL SUMMARY — v3.8")
print("★"*60)
print(f"  Pipeline time     : {elapsed():.1f} min")
print(f"  Label space       : {n_species}")
print(f"  PR-AUC            : {curr_prauc:.1f}%")
print(f"  Recall std (0.50) : {curr_rec_std:.1f}%")
print(f"  Recall tuned (90%): {curr_rec_tun:.1f}%")
print(f"  Hard species ≥90% recall:")
for i in hard_indices:
    if Yb_te[:, i].sum() > 0:
        ok = "✔" if per_recall_tuned[i] >= 0.9 else "⚠"
        print(f"    {ok} {frac_names[i][:52]}")
print(f"\n  Saved to {SAVE_DIR}:")
print(f"    lgbm_classifiers_v38.pkl")
print(f"    lgbm_thresholds_recall90_v38.pkl")
print(f"    lgbm_species_names_v38.pkl")
print("★"*60)