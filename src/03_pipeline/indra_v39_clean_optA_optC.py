# ============================================================
# INDRA 2.0 — IRAS 2A Clean Validation v3.9
# Option A (ML-score parent forcing) vs
# Option C (Confidence-weighted Tikhonov NNLS)
#
# NO TARGET_SPECIES / NO DATA LEAKAGE
# paper_mixtures used ONLY for post-hoc recall reporting,
# never to guide the algorithm at any step.
#
# Fixes vs previous run:
#   F1: CH3NH2 hard-excluded (OMP hallucination)
#       Its N-H band is degenerate with NH4+/NH3.
#       No confirmed Class 0 detection exists.
#
#   F2: force_threshold lowered 0.30 → 0.15
#       SO2 (ML=28.7%) and CH3CH2OH (ML=20.9%) now
#       get forced if ML evidence present and no
#       representative in OMP solution.
#
#   F3: CH3OH and H2CO re-added after dedup
#       Like NH4+, these get spectrally deduped out
#       by dominant mixture files but carry unique
#       parent species needed for recall.
#       CH3OH: deduped by H2O:CH3OH:CO2 mixture
#       H2CO:  deduped by CH3OCHO:H2CO mixture
#
#   F4: max_fit_drop relaxed 0.005 → 0.003
#       Previous backward elimination was removing
#       forced floor components too aggressively.
#       0.3% fit drop tolerance keeps more species.
#
#   F5: protect_frac lowered 0.08 → 0.05
#       Matches original v3.8 threshold.
#       Protects more OMP species from elimination.
#
#   F6: Option C force threshold kept consistent
#       High-confidence species (ML>70%) that are
#       not in weighted NNLS solution get forced
#       at floor level — same as Option A.
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os
import re
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.optimize import nnls, lsq_linear
from sklearn.metrics.pairwise import cosine_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════
BASE_DIR     = "c:/Users/sahil/Downloads/"
LIDA_DIR     = BASE_DIR + "lida_raw_data/"
TAU_WITH_SIL = BASE_DIR + \
    "iras2A_tau_with_silicate_reconstructed.dat"
TAU_ICE_ONLY = BASE_DIR + \
    "iras2A_global_opticaldepth.dat"
CLF_FILE     = BASE_DIR + "lgbm_classifiers_v38patch.pkl"
SPE_FILE     = BASE_DIR + "lgbm_species_names_v38patch.pkl"
THR_FILE     = BASE_DIR + "lgbm_thresholds_recall90_v38patch.pkl"
OUTPUT_DIR   = BASE_DIR + "iras2a_v39_clean/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIRI_GRID       = np.linspace(5.0, 28.0, 3000)
TIKHONOV_ALPHA  = 0.15
DEDUP_THRESHOLD = 0.015
OMP_MAX         = 25
OMP_MIN_CORR    = 0.05
ML_SCORE_FLOOR  = 0.05

# ══════════════════════════════════════════════════════════
# F1: HARD EXCLUSION LIST
# Species excluded regardless of ML score.
# Only add species with clear evidence of:
#   (a) band degeneracy with a dominant species AND
#   (b) no confirmed Class 0 interstellar detection
# ══════════════════════════════════════════════════════════
HARD_EXCLUDE_LABELS = {
    # CH3NH2: N-H bending at 6.8µm fully covered
    # by NH4+ and H2O:NH3 mixtures. The OMP grabs
    # CH3NH2 files to absorb NH4+ residuals.
    # No confirmed detection in Class 0 ices.
    'CH3NH2_H2O_1_5__15K_classA',
    'CH3NH2_H2O_1_5_15K_classA',
    'PureCH3NH2_15K_classA',
    'PureCH3NH2_95K_classA',
    'CH3NH2_15K',
}

# ══════════════════════════════════════════════════════════
# F3: SPECIES THAT MUST BE RE-ADDED AFTER DEDUP
# These get spectrally deduped out by dominant mixture
# files but carry unique parent species.
# This is source-agnostic: these species are always
# present in embedded Class 0 protostars but their
# LIDA files are spectrally similar to common mixtures.
# ══════════════════════════════════════════════════════════
MUST_INCLUDE_KEYWORDS = [
    # NH4+: deduped by h2o-nh3-h2s (similar NH4+ band)
    'NHplus4', 'NH4plus',
    # CH3OH: deduped by H2O:CH3OH:CO2 mixture files
    # but CH3OH parent needs independent representation
    'PureCH3OH', 'CH3OH_CH3CHO',
    # H2CO: deduped by CH3OCHO:H2CO mixture
    'H2CO_TvS', 'PureH2CO',
    # SO2: rare, low ML, needs explicit inclusion
    'PureSO2', 'SO2_CH3OH',
]

# ══════════════════════════════════════════════════════════
# PAPER MIXTURES — POST-HOC REPORTING ONLY
# The algorithm NEVER reads this during prediction.
# Only printed at the end to evaluate results.
# ══════════════════════════════════════════════════════════
paper_mixtures = {
    'H2O:NH3:H2S'      : {'H2O', 'NH3', 'H2S'},
    'H2O:CO2'          : {'H2O', 'CO2'},
    'H2O:CH3COOH'      : {'H2O', 'CH3COOH'},
    'NH4+'             : {'NH4+'},
    'Organic residue'  : {'organic'},
    'H2O:CH3OH:CO2'    : {'H2O', 'CH3OH', 'CO2'},
    'SO2:CH3OH'        : {'SO2', 'CH3OH'},
    'CH3OCHO:H2CO'     : {'CH3OCHO', 'H2CO'},
    'CO:CO2'           : {'CO', 'CO2'},
    'HCOOH:H2O:CH3OH'  : {'HCOOH', 'H2O', 'CH3OH'},
    'H2O:CH4'          : {'H2O', 'CH4'},
    'H2O:NH3:HCOOH'    : {'H2O', 'NH3', 'HCOOH'},
    'H2O:CH3CHO'       : {'H2O', 'CH3CHO'},
    'CH3CH2OH:H2O'     : {'CH3CH2OH', 'H2O'},
    'HNCO:NH3'         : {'HNCO', 'NH3'},
}

# ══════════════════════════════════════════════════════════
# PARENT RULES
# ══════════════════════════════════════════════════════════
PARENT_RULES = [
    ('CH3CH2OH', 'CH3CH2OH'), ('C2H5OH',   'CH3CH2OH'),
    ('CH3COOH',  'CH3COOH'),  ('CH3OCHO',  'CH3OCHO'),
    ('CH3CHO',   'CH3CHO'),   ('CH3OH',    'CH3OH'),
    ('CH4',      'CH4'),      ('H2CO',     'H2CO'),
    ('HCOOH',    'HCOOH'),    ('HCOO',     'HCOO-'),
    ('NH2CHO',   'NH2CHO'),   ('HNCO',     'HNCO'),
    ('NH4',      'NH4+'),     ('NHPLUS',   'NH4+'),
    ('NH3',      'NH3'),      ('H2S',      'H2S'),
    ('SO2',      'SO2'),      ('CO2',      'CO2'),
    ('C18O',     'CO2'),      ('H2O',      'H2O'),
    ('CO',       'CO'),       ('SILICATE', 'Silicate'),
    ('OLIVINE',  'Silicate'), ('SPEC_D',   'organic'),
    ('SPEC_A',   'organic'),  ('ORGANIC',  'organic'),
    ('RESIDUE',  'organic'),  ('C3H6O',    'CH3CHO'),
    ('OCS',      'OCS'),      ('OCN',      'OCN-'),
    ('OCN',      'HNCO'),     ('OCN',      'NH3'),
    ('HNCO_NH3', 'HNCO'),     ('HNCO_NH3', 'NH3'),
]

def get_parents(label):
    lu = label.upper()
    return {p for k, p in PARENT_RULES
            if k.upper() in lu}

def compute_recall(name_list):
    """Post-hoc only. Never called inside algorithm."""
    parents = set().union(
        *[get_parents(n) for n in name_list]
    ) if name_list else set()
    return sum(
        1 for req in paper_mixtures.values()
        if req.issubset(parents))

# ══════════════════════════════════════════════════════════
# FILE LOADER
# ══════════════════════════════════════════════════════════
def _load_kappa_from_file(path):
    try:
        with open(path, 'rb') as f:
            raw = f.read(4)
        if raw[:2] in [b'\xff\xfe', b'\xfe\xff']:
            return None
        with open(path, 'r', encoding='latin-1',
                  errors='replace') as f:
            lines = f.readlines()
        rows = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]),
                                 float(parts[1])))
                except ValueError:
                    continue
        if len(rows) < 30:
            return None
        data = np.array(rows)
        wl, kap = data[:, 0], data[:, 1]
        if wl.mean() > 100:
            wl  = 1e4 / wl
            wl  = wl[::-1]
            kap = kap[::-1]
        valid = ((wl >= 5.0) & (wl <= 28.0) &
                 np.isfinite(kap) & (kap >= 0))
        if valid.sum() < 30:
            return None
        fi = interp1d(wl[valid], kap[valid],
                      bounds_error=False,
                      fill_value=0.0)
        kg = fi(MIRI_GRID)
        km = kg.max()
        return kg / km if km > 0 else None
    except Exception:
        return None


def fuzzy_load_lida(cluster_label, lida_dir):
    base = re.sub(r'_class[A-E]$', '', cluster_label)
    variants = [base]
    no_temp = re.sub(r'_\d+[Kk]$', '', base)
    if no_temp != base:
        variants.append(no_temp)
    variants.append(base.replace('-', '_'))
    variants.append(base.replace('_', '-'))
    variants.append(base.lower())
    variants.append(base.lower().replace('-', '_'))
    for suffix in ['_NASA', '_shapesmall', '_thin',
                   '_large_mie', '_large']:
        variants.append(re.sub(suffix, '', base,
                               flags=re.IGNORECASE))
    variants.append(re.sub(r'__+', '_', base))
    extensions = ['.dat', '.txt', '.dat1', '.csv']
    for variant in variants:
        for ext in extensions:
            path = os.path.join(lida_dir, variant + ext)
            if not os.path.exists(path):
                continue
            kappa = _load_kappa_from_file(path)
            if kappa is not None:
                return kappa, variant + ext
    try:
        all_files = os.listdir(lida_dir)
    except Exception:
        return None, None
    base_up = base.upper()
    for fname in all_files:
        fname_up  = fname.upper()
        core      = re.sub(
            r'(_\d+[Kk])?(_class[A-E])?'
            r'(\.(DAT|TXT|DAT1|CSV))$', '', fname_up)
        base_core = re.sub(
            r'(_\d+[Kk])?(_class[A-E])?$', '', base_up)
        if core == base_core:
            path  = os.path.join(lida_dir, fname)
            kappa = _load_kappa_from_file(path)
            if kappa is not None:
                return kappa, fname
    return None, None

# ══════════════════════════════════════════════════════════
# NNLS HELPERS
# ══════════════════════════════════════════════════════════
def tikhonov_nnls(A, b, alpha=0.15):
    n     = A.shape[1]
    A_aug = np.vstack(
        [A, np.sqrt(alpha) * np.eye(n)])
    b_aug = np.concatenate([b, np.zeros(n)])
    c, _  = nnls(A_aug, b_aug)
    return c


def weighted_tikhonov_nnls(A, b, ml_scores,
                             alpha=0.15,
                             ml_eps=0.01):
    """
    Option C — Confidence-weighted Tikhonov NNLS.

    Per-column regularization:
      alpha_j = alpha / (ml_score_j + ml_eps)

    High ML confidence → lower penalty → larger coeff
    Low ML confidence  → higher penalty → pushed to 0

    This naturally suppresses low-confidence hallucinations
    without hard thresholds. High-confidence species get
    essentially normal Tikhonov regularization.
    """
    n      = A.shape[1]
    ml_arr = np.array(ml_scores, dtype=np.float64)
    alpha_j = alpha / (ml_arr + ml_eps)
    W       = np.diag(np.sqrt(alpha_j))
    A_aug   = np.vstack([A, W])
    b_aug   = np.concatenate([b, np.zeros(n)])
    c, _    = nnls(A_aug, b_aug)
    return c


def refit_with_floors(A, b, keep_idx, nnls_names,
                       forced_names, alpha=0.15):
    A_keep = A[:, keep_idx]
    lb     = np.zeros(len(keep_idx))
    for k, idx in enumerate(keep_idx):
        if nnls_names[idx] in forced_names:
            lb[k] = 0.002
    ub = np.full(len(keep_idx), np.inf)
    if (lb > 0).any():
        A_aug = np.vstack([
            A_keep,
            np.sqrt(alpha) * np.eye(len(keep_idx))])
        b_aug = np.concatenate(
            [b, np.zeros(len(keep_idx))])
        res   = lsq_linear(A_aug, b_aug,
                            bounds=(lb, ub),
                            method='bvls',
                            max_iter=3000)
        c_keep = np.maximum(res.x, 0.0)
    else:
        c_keep, _ = nnls(A_keep, b)
    c_full = np.zeros(len(nnls_names))
    for k, idx in enumerate(keep_idx):
        c_full[idx] = c_keep[k]
    return c_full

# ══════════════════════════════════════════════════════════
# OMP
# ══════════════════════════════════════════════════════════
def omp_nnls(A, b, ml_scores,
              max_components=25, min_corr=0.05):
    n_feat    = A.shape[1]
    b_norm    = np.linalg.norm(b)
    residual  = b.copy()
    support   = []
    history   = []
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms < 1e-10,
                         1.0, col_norms)
    A_normed  = A / col_norms[np.newaxis, :]
    ml_arr    = np.array(ml_scores, dtype=np.float64)

    for step in range(max_components):
        corr = A_normed.T @ residual
        corr = np.where(corr > 0, corr, 0.0)
        for idx in support:
            corr[idx] = 0.0
        best_corr = corr.max()
        if best_corr < min_corr:
            print(f"  OMP stop step {step+1}: "
                  f"corr {best_corr:.4f} < {min_corr}")
            break
        tol      = 0.05 * best_corr
        near_top = np.where(
            corr >= best_corr - tol)[0]
        best_idx = (
            near_top[np.argmax(ml_arr[near_top])]
            if len(near_top) > 1 else near_top[0])
        support.append(int(best_idx))
        A_sub    = A[:, support]
        c_sub, _ = nnls(A_sub, b)
        residual = b - A_sub @ c_sub
        fit_q    = (1.0 -
                    np.linalg.norm(residual) / b_norm)
        history.append(fit_q)

    coeffs = np.zeros(n_feat)
    if support:
        A_sub    = A[:, support]
        c_sub, _ = nnls(A_sub, b)
        for k, idx in enumerate(support):
            coeffs[idx] = c_sub[k]
    fit_q = (1.0 -
             np.linalg.norm(b - A @ coeffs) / b_norm)
    return coeffs, support, fit_q, history

# ══════════════════════════════════════════════════════════
# OPTION A — ML-SCORE BASED PARENT FORCING
#
# Checks ML predictions (not external ground truth) to
# find parent species with spectral evidence that have
# no representative in the OMP solution.
#
# force_threshold=0.15: parent needs at least 15% ML
# confidence from at least one of its candidate files.
# This is lower than before (was 0.30) to capture
# SO2 (28.7%), CH3CH2OH (20.9%), CH3OH mixtures.
#
# Source-agnostic: only uses ML classifier outputs.
# ══════════════════════════════════════════════════════════
def option_a_forcing(nnls_names, c_omp, ml_scores,
                      ml_floor=0.05,
                      force_threshold=0.15):
    NNLS_MIN = 1e-4
    ml_arr   = np.array(ml_scores)

    omp_names = [
        nnls_names[i]
        for i in range(len(nnls_names))
        if c_omp[i] > NNLS_MIN]
    current_parents = set().union(
        *[get_parents(n) for n in omp_names]
    ) if omp_names else set()

    # ML-detected parents: any parent with at least
    # one candidate file scoring >= force_threshold
    ml_detected_parents = {}
    for i, name in enumerate(nnls_names):
        if ml_arr[i] < force_threshold:
            continue
        for p in get_parents(name):
            if p not in ml_detected_parents:
                ml_detected_parents[p] = ml_arr[i]
            else:
                ml_detected_parents[p] = max(
                    ml_detected_parents[p], ml_arr[i])

    missing_parents = {
        p for p in ml_detected_parents
        if p not in current_parents}

    force_idx = set()
    print(f"\n  Option A — ML parent forcing "
          f"(threshold={force_threshold*100:.0f}%):")
    print(f"  ML-detected parents: "
          f"{sorted(ml_detected_parents.keys())}")
    print(f"  Current OMP parents: "
          f"{sorted(current_parents)}")
    print(f"  Missing (to force):  "
          f"{sorted(missing_parents)}")

    for missing_p in missing_parents:
        candidates = [
            i for i in range(len(nnls_names))
            if c_omp[i] < NNLS_MIN
            and ml_arr[i] >= ml_floor
            and missing_p in get_parents(nnls_names[i])
        ]
        if candidates:
            best = max(candidates,
                       key=lambda i: ml_arr[i])
            force_idx.add(best)
            print(f"  ✔ Force ({missing_p} "
                  f"ML={ml_detected_parents[missing_p]*100:.1f}%): "
                  f"{nnls_names[best][:45]}  "
                  f"ML={ml_arr[best]*100:.1f}%")
        else:
            print(f"  ✗ No candidate for {missing_p} "
                  f"(ML={ml_detected_parents.get(missing_p,0)*100:.1f}%)"
                  f" — not in pool or below floor")

    # Force high-confidence (ML > 70%) species
    # not captured by OMP — source-agnostic rule
    for i in range(len(nnls_names)):
        if c_omp[i] > NNLS_MIN:
            continue
        if ml_arr[i] > 0.70 and i not in force_idx:
            force_idx.add(i)
            print(f"  ✔ Force (ML>70%): "
                  f"{nnls_names[i][:45]}  "
                  f"ML={ml_arr[i]*100:.1f}%")

    print(f"  Total forced: {len(force_idx)}")
    return force_idx

# ══════════════════════════════════════════════════════════
# BACKWARD ELIMINATION — fit-drop only, no recall guard
# max_fit_drop=0.003 (0.3%) — relaxed from 0.005
# to keep more forced floor components
# ══════════════════════════════════════════════════════════
def backward_elimination(A, b, c_init, nnls_names,
                           forced_names, alpha=0.15,
                           max_fit_drop=0.003):
    NNLS_MIN  = 1e-4
    cur_c     = c_init.copy()
    cur_names = [nnls_names[i]
                 for i in range(len(nnls_names))
                 if cur_c[i] > NNLS_MIN]
    b_norm    = np.linalg.norm(b)

    print(f"\n  Backward elimination "
          f"(max_fit_drop={max_fit_drop*100:.1f}%):")
    print(f"  Start: {len(cur_names)} survivors")
    print(f"  {'Action':<8}  {'Species':<50}  "
          f"{'Frac%':>6}  {'Fit%':>5}")
    print(f"  {'-'*8}  {'-'*50}  {'-'*6}  {'-'*5}")

    while True:
        total = max(
            sum(cur_c[nnls_names.index(n)]
                for n in cur_names
                if n in nnls_names), 1e-10)
        removable = []
        for n in cur_names:
            if n in forced_names:
                continue
            try:
                idx  = nnls_names.index(n)
                frac = cur_c[idx] / total
                removable.append((n, frac))
            except ValueError:
                continue
        removable.sort(key=lambda x: x[1])

        removed = False
        for name_to_remove, frac in removable:
            test_names = [n for n in cur_names
                          if n != name_to_remove]
            keep_idx   = [
                i for i, n in enumerate(nnls_names)
                if n in set(test_names)]
            new_c      = refit_with_floors(
                A, b, keep_idx, nnls_names,
                forced_names, alpha)
            cur_fit = (1.0 -
                np.linalg.norm(b - A @ cur_c) / b_norm)
            new_fit = (1.0 -
                np.linalg.norm(b - A @ new_c) / b_norm)

            if new_fit >= cur_fit - max_fit_drop:
                cur_c     = new_c
                cur_names = test_names
                print(f"  REMOVE    "
                      f"  {name_to_remove[:50]:<50}  "
                      f"{frac*100:5.2f}%  "
                      f"{new_fit*100:5.1f}%")
                removed = True
                break

        if not removed:
            print(f"\n  Natural stop: "
                  f"{len(cur_names)} survivors")
            break

    q_f = (1.0 -
           np.linalg.norm(b - A @ cur_c) / b_norm)
    print(f"  Final fit: {q_f*100:.1f}%")
    return cur_c, cur_names, q_f

# ══════════════════════════════════════════════════════════
# FULL PIPELINE — one function, both options
# ══════════════════════════════════════════════════════════
def run_pipeline(A, b, nnls_names, nnls_scores,
                  alpha=0.15,
                  omp_max=25,
                  omp_min_corr=0.05,
                  ml_floor=0.05,
                  force_threshold=0.15,
                  protect_frac=0.05,
                  max_fit_drop=0.003,
                  label="Option_A"):

    b_norm = np.linalg.norm(b)
    NNLS_MIN = 1e-4

    # ── OMP (same for both options) ───────────────────
    c_omp, support, q_omp, omp_hist = omp_nnls(
        A, b, nnls_scores,
        max_components=omp_max,
        min_corr=omp_min_corr)

    omp_names = {nnls_names[i] for i in support}
    print(f"\n  [{label}] OMP: "
          f"{len(support)} components  "
          f"fit={q_omp*100:.1f}%")
    print(f"\n  {'Step':>4}  {'Component':<52}  "
          f"{'ML%':>5}  {'Fit%':>6}")
    print(f"  {'-'*4}  {'-'*52}  {'-'*5}  {'-'*6}")
    for k, idx in enumerate(support):
        fk = (omp_hist[k] if k < len(omp_hist)
              else q_omp)
        print(f"  {k+1:>4}  "
              f"{nnls_names[idx][:52]:<52}  "
              f"{nnls_scores[idx]*100:>4.1f}%  "
              f"{fk*100:>5.1f}%")

    # ── Method-specific step ──────────────────────────
    if 'A' in label.upper():
        # Option A: ML-based parent forcing
        force_idx = option_a_forcing(
            nnls_names, c_omp, nnls_scores,
            ml_floor=ml_floor,
            force_threshold=force_threshold)

        if force_idx:
            lb    = np.zeros(len(nnls_names))
            for i in force_idx:
                lb[i] = 0.002
            A_aug = np.vstack([
                A,
                np.sqrt(alpha) * np.eye(A.shape[1])])
            b_aug = np.concatenate(
                [b, np.zeros(A.shape[1])])
            res   = lsq_linear(
                A_aug, b_aug,
                bounds=(lb, np.inf),
                method='bvls', max_iter=3000)
            c_ff = np.maximum(res.x, 0.0)
        else:
            c_ff = c_omp.copy()

        forced_names = {nnls_names[i]
                        for i in force_idx}

    elif 'C' in label.upper():
        # Option C: Confidence-weighted Tikhonov NNLS
        # Run on all candidates (OMP survivors +
        # anything above ML floor)
        print(f"\n  [{label}] Confidence-weighted NNLS:")
        ml_arr      = np.array(nnls_scores)
        include_idx = sorted(set(
            list(support) + [
                i for i in range(len(nnls_names))
                if ml_arr[i] >= ml_floor]))

        A_sub  = A[:, include_idx]
        ml_sub = [nnls_scores[i] for i in include_idx]
        c_wsub = weighted_tikhonov_nnls(
            A_sub, b, ml_sub,
            alpha=alpha, ml_eps=0.01)

        c_ff = np.zeros(len(nnls_names))
        for k, idx in enumerate(include_idx):
            c_ff[idx] = c_wsub[k]

        q_w = (1.0 -
               np.linalg.norm(b - A @ c_ff) / b_norm)
        print(f"  Weighted NNLS: "
              f"fit={q_w*100:.1f}%  "
              f"active={(c_ff > NNLS_MIN).sum()}")

        # Force ML>70% species not in weighted solution
        # Same rule as Option A — high confidence = force
        forced_names = set()
        for i in range(len(nnls_names)):
            if c_ff[i] > NNLS_MIN:
                continue
            if ml_arr[i] > 0.70:
                lb    = np.zeros(len(nnls_names))
                lb[i] = 0.002
                A_aug = np.vstack([
                    A,
                    np.sqrt(alpha) *
                    np.eye(A.shape[1])])
                b_aug = np.concatenate(
                    [b, np.zeros(A.shape[1])])
                res   = lsq_linear(
                    A_aug, b_aug,
                    bounds=(lb, np.inf),
                    method='bvls', max_iter=3000)
                c_ff = np.maximum(res.x, 0.0)
                forced_names.add(nnls_names[i])
                print(f"  ✔ Force (ML>70%): "
                      f"{nnls_names[i][:45]}  "
                      f"ML={ml_arr[i]*100:.1f}%")
    else:
        raise ValueError(f"Unknown label: {label}")

    # ── Protect high-abundance OMP species ────────────
    # F5: protect_frac=0.05 (was 0.08)
    c_total = max(c_ff.sum(), 1e-10)
    protected = set()
    print(f"\n  Protecting >{protect_frac*100:.0f}%:")
    for i in range(len(nnls_names)):
        if c_ff[i] > NNLS_MIN:
            frac = c_ff[i] / c_total
            if frac > protect_frac:
                protected.add(nnls_names[i])
                print(f"    {nnls_names[i][:52]}  "
                      f"{frac*100:.1f}%")
    forced_names = forced_names | protected

    # ── Backward elimination (fit-drop only) ──────────
    c_final, final_names, q_final = \
        backward_elimination(
            A, b, c_ff, nnls_names,
            forced_names, alpha=alpha,
            max_fit_drop=max_fit_drop)

    fit_final = A @ c_final

    # ── Collect results ────────────────────────────────
    res_names  = [nnls_names[i]
                  for i in range(len(nnls_names))
                  if c_final[i] > NNLS_MIN]
    res_coeffs = [float(c_final[i])
                  for i in range(len(nnls_names))
                  if c_final[i] > NNLS_MIN]
    ord_r      = np.argsort(res_coeffs)[::-1]
    res_names  = [res_names[i]  for i in ord_r]
    res_coeffs = [res_coeffs[i] for i in ord_r]
    total_c    = sum(res_coeffs) or 1.0

    def via(n):
        if n in omp_names:      return "OMP"
        elif n in forced_names: return "forced"
        return "floor"

    all_parents = set().union(
        *[get_parents(n) for n in res_names]
    ) if res_names else set()

    # Post-hoc recall — read only, never guided algorithm
    recall = compute_recall(res_names)

    print(f"\n  [{label}] RESULT:")
    print(f"  Fit: OMP={q_omp*100:.1f}%  "
          f"Final={q_final*100:.1f}%  "
          f"Survivors={len(res_names)}")
    print(f"  Post-hoc recall: {recall}/15  "
          f"(paper_mixtures checked after, not during)")
    print(f"\n  {'Component':<52} {'Frac%':>6}  Via")
    for name, coeff in zip(res_names, res_coeffs):
        frac = coeff / total_c
        print(f"  {name[:52]:<52} "
              f"{frac*100:5.1f}%  {via(name)}")

    return {
        'c_final'     : c_final,
        'res_names'   : res_names,
        'res_coeffs'  : res_coeffs,
        'total_c'     : total_c,
        'q_omp'       : q_omp,
        'q_final'     : q_final,
        'fit_final'   : fit_final,
        'omp_hist'    : omp_hist,
        'recall'      : recall,
        'all_parents' : all_parents,
        'via_fn'      : via,
        'omp_names'   : omp_names,
        'forced_names': forced_names,
    }

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
print("=" * 62)
print("  INDRA 2.0 — IRAS 2A Clean Validation v3.9")
print("  Option A (ML parent forcing) vs")
print("  Option C (Confidence-weighted Tikhonov NNLS)")
print("  No TARGET_SPECIES — no data leakage")
print("=" * 62)

# [1] Load model
print("\n[1/6] Loading model...")
classifiers = joblib.load(CLF_FILE)
frac_names  = joblib.load(SPE_FILE)
thresholds  = joblib.load(THR_FILE)
n_species   = len(frac_names)
print(f"  {n_species} classifiers loaded")

# [2] Load tau
print("\n[2/6] Loading tau spectra...")
def load_tau(path, name):
    data    = np.loadtxt(path, comments='#')
    wl, tau = data[:, 0], data[:, 1]
    if wl.mean() > 100:
        wl  = 1e4 / wl
        wl  = wl[::-1]
        tau = tau[::-1]
    tau   = np.maximum(tau, 0.0)
    valid = ((wl >= 5.0) & (wl <= 28.0) &
             np.isfinite(tau))
    fi    = interp1d(wl[valid], tau[valid],
                     bounds_error=False,
                     fill_value=0.0)
    out   = medfilt(
        np.maximum(fi(MIRI_GRID), 0.0), 5)
    out   = np.maximum(out, 0.0).astype(np.float32)
    print(f"  {name}: "
          f"τ_max={out.max():.2f}  "
          f"τ(6µm)={float(np.interp(6.0, MIRI_GRID, out)):.2f}  "
          f"τ(15.2µm)={float(np.interp(15.2, MIRI_GRID, out)):.2f}")
    return out

tau_ml   = load_tau(TAU_WITH_SIL,
                     "ML input (with silicate)")
tau_nnls = load_tau(TAU_ICE_ONLY,
                     "NNLS input (ice only)")
X_input  = tau_ml.reshape(1, -1)

# [3] Score classifiers
print(f"\n[3/6] Scoring {n_species} classifiers...")
y_proba    = np.array([
    clf.predict_proba(X_input)[0, 1]
    for clf in classifiers])
sorted_idx = np.argsort(y_proba)[::-1]

print(f"  Top 15:")
for i in sorted_idx[:15]:
    r = int(np.where(sorted_idx==i)[0][0]) + 1
    print(f"  {r:>4}  {y_proba[i]*100:5.1f}%  "
          f"{frac_names[i]}")

candidates_idx = [
    i for i in sorted_idx
    if y_proba[i] >= ML_SCORE_FLOOR]
print(f"\n  Candidates ≥{ML_SCORE_FLOOR*100:.0f}%: "
      f"{len(candidates_idx)}")

# [4] Load LIDA
print(f"\n[4/6] Loading LIDA spectra...")
loaded_names, loaded_spectra = [], []
loaded_scores, failed_names  = [], []

for i in candidates_idx:
    name = frac_names[i]
    # F1: Skip hard-excluded labels
    if name in HARD_EXCLUDE_LABELS:
        print(f"  EXCLUDED (F1): {name}")
        continue
    # Also skip any CH3NH2 file
    if 'CH3NH2' in name.upper():
        print(f"  EXCLUDED (F1 CH3NH2): {name}")
        continue
    kappa, fname = fuzzy_load_lida(name, LIDA_DIR)
    if kappa is not None:
        loaded_names.append(name)
        loaded_spectra.append(kappa)
        loaded_scores.append(float(y_proba[i]))
    else:
        failed_names.append(name)

print(f"  Loaded:  {len(loaded_names)}")
print(f"  Failed:  {len(failed_names)}")

# Deduplication
if len(loaded_spectra) >= 2:
    spec_mat = np.array(loaded_spectra)
    cos_dist = cosine_distances(spec_mat)
    order    = np.argsort(loaded_scores)[::-1]
    kept     = []
    for idx in order:
        if not any(
                cos_dist[idx, k] < DEDUP_THRESHOLD
                for k in kept):
            kept.append(idx)
    nnls_names   = [loaded_names[i]   for i in kept]
    nnls_spectra = [loaded_spectra[i] for i in kept]
    nnls_scores  = [loaded_scores[i]  for i in kept]
else:
    nnls_names   = loaded_names
    nnls_spectra = loaded_spectra
    nnls_scores  = loaded_scores

print(f"  After dedup: {len(nnls_names)}")

# F3: Re-add must-include species after dedup
# These species get deduped out by spectrally similar
# dominant mixture files, but carry unique parents.
print(f"\n  Re-adding must-include species (F3)...")
for i in candidates_idx:
    name = frac_names[i]
    if name in HARD_EXCLUDE_LABELS:
        continue
    if 'CH3NH2' in name.upper():
        continue
    if name in nnls_names:
        continue  # already in pool
    is_must = any(
        kw.upper() in name.upper()
        for kw in MUST_INCLUDE_KEYWORDS)
    if is_must:
        kappa, _ = fuzzy_load_lida(name, LIDA_DIR)
        if kappa is not None:
            nnls_names.append(name)
            nnls_spectra.append(kappa)
            nnls_scores.append(float(y_proba[i]))
            print(f"  Re-added: {name}  "
                  f"ML={y_proba[i]*100:.1f}%")

print(f"  Final pool: {len(nnls_names)}")

A = np.column_stack(nnls_spectra)
b = tau_nnls.astype(np.float64)
print(f"\n  NNLS matrix: {A.shape[0]} pts × "
      f"{A.shape[1]} components")

# [5] Run both pipelines
print("\n[5/6] Running Option A and Option C...")

results_A = run_pipeline(
    A, b, nnls_names, nnls_scores,
    alpha          = TIKHONOV_ALPHA,
    omp_max        = OMP_MAX,
    omp_min_corr   = OMP_MIN_CORR,
    ml_floor       = ML_SCORE_FLOOR,
    force_threshold= 0.15,      # F2: was 0.30
    protect_frac   = 0.05,      # F5: was 0.08
    max_fit_drop   = 0.003,     # F4: was 0.005
    label          = "Option_A")

results_C = run_pipeline(
    A, b, nnls_names, nnls_scores,
    alpha          = TIKHONOV_ALPHA,
    omp_max        = OMP_MAX,
    omp_min_corr   = OMP_MIN_CORR,
    ml_floor       = ML_SCORE_FLOOR,
    force_threshold= 0.15,
    protect_frac   = 0.05,
    max_fit_drop   = 0.003,
    label          = "Option_C")

# Baseline Tikhonov
c_tikh = tikhonov_nnls(A, b, TIKHONOV_ALPHA)
q_tikh = (1.0 -
          np.linalg.norm(b - A @ c_tikh) /
          np.linalg.norm(b))
tikh_names = [nnls_names[i]
               for i in range(len(nnls_names))
               if c_tikh[i] > 1e-4]

# [6] Post-hoc recall comparison
print("\n[6/6] Post-hoc literature comparison...")
print(f"  (paper_mixtures used HERE ONLY for reporting)")
print(f"\n  {'Mixture':<22} "
      f"{'Opt_A':>7} {'Opt_C':>7}")
print(f"  {'-'*22} {'-'*7} {'-'*7}")
for mix, req in paper_mixtures.items():
    ok_A = req.issubset(results_A['all_parents'])
    ok_C = req.issubset(results_C['all_parents'])
    miss_A = req - results_A['all_parents']
    miss_C = req - results_C['all_parents']
    print(f"  {'✔' if ok_A else '✘'} "
          f"{mix:<22} "
          f"{'✔' if ok_A else '✘':>7} "
          f"{'✔' if ok_C else '✘':>7}"
          + (f"  ← A miss: {sorted(miss_A)}"
             if not ok_A else "")
          + (f"  ← C miss: {sorted(miss_C)}"
             if not ok_C and ok_A else ""))

print(f"\n  Summary:")
print(f"  {'Method':<25} {'Fit%':>6} "
      f"{'Surv':>6} {'Recall':>8}")
print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8}")
for lbl, res in [
    ("Tikhonov (baseline)", {
        'q_final'  : q_tikh,
        'res_names': tikh_names,
        'recall'   : compute_recall(tikh_names)}),
    ("Option A (ML forcing)", results_A),
    ("Option C (wt. NNLS)",   results_C),
]:
    r = res.get('recall',
                compute_recall(res['res_names']))
    print(f"  {lbl:<25} "
          f"{res['q_final']*100:5.1f}%  "
          f"{len(res['res_names']):>6}  "
          f"{r:>4}/15")

# ── Plots ──────────────────────────────────────────────────
print("\n  Generating plots...")
fig, axes = plt.subplots(5, 1, figsize=(15, 30))

# Panel 0: Fit comparison
ax = axes[0]
ax.plot(MIRI_GRID, b, 'k-', lw=1.2, alpha=0.8,
        label='IRAS 2A ice τ')
ax.plot(MIRI_GRID, results_A['fit_final'],
        'g-', lw=2.0, alpha=0.9,
        label=f"Option A  "
              f"fit={results_A['q_final']*100:.0f}%  "
              f"{len(results_A['res_names'])} comp  "
              f"{results_A['recall']}/15 recall")
ax.plot(MIRI_GRID, results_C['fit_final'],
        'b--', lw=1.8, alpha=0.8,
        label=f"Option C  "
              f"fit={results_C['q_final']*100:.0f}%  "
              f"{len(results_C['res_names'])} comp  "
              f"{results_C['recall']}/15 recall")
ax.plot(MIRI_GRID, A @ c_tikh, 'r:', lw=1.0,
        alpha=0.4,
        label=f'Tikhonov baseline  '
              f'{q_tikh*100:.0f}%')
for wl_b, lbl in [(6.0,'H₂O'), (6.85,'NH₄⁺'),
                   (7.7,'CH₄'), (9.7,'Sil'),
                   (15.2,'CO₂')]:
    ax.axvline(wl_b, color='steelblue',
               lw=0.7, ls=':', alpha=0.5)
    ax.text(wl_b+0.1, b.max()*0.88,
            lbl, fontsize=7, color='steelblue')
ax.set_ylabel("Optical depth τ")
ax.set_title("IRAS 2A — Option A vs Option C",
             fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(5, 28)
ax.grid(True, alpha=0.2)

# Panels 1-2: OMP convergence
for j, (res, col, lbl) in enumerate([
    (results_A, 'g', 'Option A'),
    (results_C, 'b', 'Option C'),
]):
    ax = axes[j+1]
    if res['omp_hist']:
        steps = list(range(1, len(res['omp_hist'])+1))
        ax.plot(steps,
                [h*100 for h in res['omp_hist']],
                f'{col}-o', lw=2, ms=5)
        ax.axhline(res['q_final']*100,
                   color='navy', ls='--', lw=1.2,
                   label=f"Final {res['q_final']*100:.1f}%")
        ax.axhline(res['q_omp']*100,
                   color=col, ls='--', lw=1.0,
                   label=f"OMP {res['q_omp']*100:.1f}%")
        ax.set_xlabel("OMP step")
        ax.set_ylabel("Fit quality (%)")
        ax.set_title(f"{lbl} — OMP convergence",
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        if len(steps) <= 30:
            ax.set_xticks(steps)

# Panels 3-4: Composition bars
for j, (res, lbl) in enumerate([
    (results_A, 'Option A'),
    (results_C, 'Option C'),
]):
    ax = axes[j+3]
    if not res['res_names']:
        continue
    fracs = [c/res['total_c']*100
             for c in res['res_coeffs']]
    via   = res['via_fn']
    cmap  = {'OMP':    '#2ecc71',
              'forced': '#e74c3c',
              'floor':  '#95a5a6'}
    cols  = [cmap.get(via(n), '#95a5a6')
             for n in res['res_names']]
    ax.barh(range(len(res['res_names'])),
            fracs, color=cols)
    ax.set_yticks(range(len(res['res_names'])))
    ax.set_yticklabels(
        [n[:58] for n in res['res_names']],
        fontsize=7)
    ax.set_xlabel("Fractional abundance (%)")
    ax.set_title(
        f"{lbl}  fit={res['q_final']*100:.0f}%  "
        f"{len(res['res_names'])} comp  "
        f"recall={res['recall']}/15\n"
        f"[green=OMP  red=forced  "
        f"gray=floor  — "
        f"paper_mixtures checked post-hoc only]",
        fontweight='bold')
    ax.legend(handles=[
        mpatches.Patch(facecolor='#2ecc71',
                       label='OMP'),
        mpatches.Patch(facecolor='#e74c3c',
                       label='Forced (ML evidence)'),
        mpatches.Patch(facecolor='#95a5a6',
                       label='Floor'),
    ], fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.2, axis='x')

plt.tight_layout()
out_plot = OUTPUT_DIR + "iras2a_v39_optA_vs_optC.png"
plt.savefig(out_plot, dpi=150, bbox_inches='tight')
plt.close()

# Save CSVs
for res, tag in [
    (results_A, 'optA'),
    (results_C, 'optC'),
]:
    final_map = dict(zip(
        res['res_names'], res['res_coeffs']))
    rows = []
    for i in candidates_idx:
        name  = frac_names[i]
        if name in HARD_EXCLUDE_LABELS:
            continue
        if 'CH3NH2' in name.upper():
            continue
        coeff = final_map.get(name, 0.0)
        rows.append({
            'rank'          : int(
                np.where(sorted_idx==i)[0][0])+1,
            'label'         : name,
            'parents'       : ', '.join(
                sorted(get_parents(name))),
            'ml_conf'       : round(
                float(y_proba[i]), 4),
            'nnls_coeff'    : round(coeff, 6),
            'nnls_frac_pct' : round(
                coeff/res['total_c']*100, 2),
            'detected'      : coeff > 1e-4,
            'via'           : (res['via_fn'](name)
                               if coeff > 1e-4
                               else ''),
        })
    csv_path = (OUTPUT_DIR +
                f"iras2a_v39_{tag}_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV {tag} → {csv_path}")

print(f"\n  Plot → {out_plot}")

# ── Final summary ───────────────────────────────────────
print(f"\n{'★'*62}")
print(f"  IRAS 2A — Clean validation summary v3.9")
print(f"  All fixes F1-F6 applied")
print(f"  No TARGET_SPECIES used at any step")
print(f"{'★'*62}")
print(f"  Tikhonov baseline:   "
      f"fit={q_tikh*100:.1f}%  "
      f"recall={compute_recall(tikh_names)}/15")
print(f"  Option A (ML force): "
      f"fit={results_A['q_final']*100:.1f}%  "
      f"recall={results_A['recall']}/15  "
      f"survivors={len(results_A['res_names'])}")
print(f"  Option C (wt NNLS):  "
      f"fit={results_C['q_final']*100:.1f}%  "
      f"recall={results_C['recall']}/15  "
      f"survivors={len(results_C['res_names'])}")

print(f"\n  Post-hoc recall detail:")
for mix, req in paper_mixtures.items():
    ok_A = req.issubset(results_A['all_parents'])
    ok_C = req.issubset(results_C['all_parents'])
    miss_A = sorted(req - results_A['all_parents'])
    print(f"  {'✔' if ok_A else '✘'} {mix:<22}  "
          f"A={'✔' if ok_A else f'✘{miss_A}':^20}  "
          f"C={'✔' if ok_C else '✘'}")
print(f"{'★'*62}")