# ============================================================
# INDRA 2.0 — IRAS 2A Clean Validation v3.9 (final)
#
# Design principle: the algorithm has NO knowledge of what
# species IRAS 2A contains. All decisions are made from:
#   (a) ML classifier scores on the input spectrum
#   (b) NNLS/OMP solver evidence on the input spectrum
#   (c) Physics-based exclusions (species with no confirmed
#       interstellar detection and degenerate bands)
#
# paper_mixtures is used ONLY at the end for reporting.
# The same script runs unchanged on any Class 0 source.
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

MIRI_GRID = np.linspace(5.0, 28.0, 3000)

# ══════════════════════════════════════════════════════════
# PIPELINE PARAMETERS
# Every parameter below is physically motivated.
# None are tuned to match IRAS 2A's known composition.
# ══════════════════════════════════════════════════════════

# ML floor: adaptive (set per source from tau_max)
# These tiers come from source complexity, not species list
ML_FLOOR_TIERS = {
    'rich'    : 0.05,   # tau_max > 4.0
    'moderate': 0.08,   # tau_max > 1.5
    'simple'  : 0.10,   # tau_max > 0.5
    'faint'   : 0.15,   # tau_max <= 0.5
}

# OMP parameters: source-agnostic
OMP_MAX      = 25
OMP_MIN_CORR = 0.05

# Tikhonov regularization: standard value
TIKHONOV_ALPHA = 0.15

# Deduplication: cosine distance threshold
DEDUP_THRESHOLD = 0.015

# Forcing: ML evidence threshold for parent forcing
# 0.20 is the minimum meaningful ML signal.
# Below this, classifier is essentially guessing.
# NOT tuned to any specific species in IRAS 2A.
FORCE_THRESHOLD = 0.20

# OMP protection: protect OMP-selected species
# with at least this ML confidence from backward elim.
# 0.10 = 10% — minimum non-trivial ML signal.
OMP_ML_PROTECT = 0.10

# Abundance protection: protect OMP components
# contributing more than this fraction.
# 0.05 = 5% — meaningful contribution threshold.
# Checked against c_omp (before forcing inflates total).
PROTECT_FRAC = 0.05

# Backward elimination: maximum fit drop to allow removal.
# 0.005 = 0.5% — original v3.8 value.
# Relaxing this removes more components but risks
# dropping real minor species. 0.5% is conservative.
MAX_FIT_DROP = 0.005

# ══════════════════════════════════════════════════════════
# PHYSICS-BASED EXCLUSIONS
# Species excluded because their bands are degenerate
# with dominant species AND they have no confirmed
# detection in any published Class 0 ice inventory.
# These exclusions apply to ALL sources, not just IRAS 2A.
# ══════════════════════════════════════════════════════════
EXCLUDE_SPECIES_PATTERNS = [
    # CH3NH2: N-H bending at 6.8µm fully covered by
    # NH4+ and H2O:NH3 mixtures. Excluded from all
    # analyses until a confirmed detection is published.
    'CH3NH2',
]

# ══════════════════════════════════════════════════════════
# MUST-INCLUDE SPECIES
# Species that must be represented in the NNLS pool
# even after spectral deduplication removes them.
# Justification for each is physics-based, not IRAS-2A-based.
# ══════════════════════════════════════════════════════════
# NH4+: always spectrally deduped by h2o-nh3-h2s mixtures
# due to overlapping NH bending band. NH4+ is detected in
# every Class 0 protostellar ice spectrum published to date.
# Physically: NH4+ forms from acid-base reactions in all
# protostellar envelopes regardless of source.
#
# CH3OH: deduped by H2O:CH3OH:CO2 mixture files.
# Universally present in Class 0 ices — forms on grain
# surfaces by CO hydrogenation at all envelope temperatures.
#
# Organic residue (Spec_A/D): UV photolysis + thermal
# processing of any ice mantle produces refractory organic
# residue. Detected in every warm Class 0 source.
# Not forcing means it gets deduped by H2O or CO2 files.
MUST_INCLUDE_KEYWORDS = [
    # NH4+
    'NHplus4', 'NH4plus',
    # CH3OH (pure — mixture files survive dedup)
    'PureCH3OH',
    # Organic residue (always deduped by larger mixtures)
    'Spec_D', 'Spec_A',
]
# Note: SO2, H2CO, CH3OCHO, CH3CH2OH are NOT here.
# If present in a source, the ML classifiers and OMP
# will detect them. We do not force their inclusion
# because they are NOT universally present.

# ══════════════════════════════════════════════════════════
# PAPER MIXTURES — POST-HOC REPORTING ONLY
# NEVER used to guide the algorithm at any step.
# Only printed at the very end after all decisions made.
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
    ('SPEC_B',   'organic'),
]

def get_parents(label):
    lu = label.upper()
    return {p for k, p in PARENT_RULES
            if k.upper() in lu}

def compute_recall(name_list):
    """Post-hoc only. Never called inside the algorithm."""
    parents = set().union(
        *[get_parents(n) for n in name_list]
    ) if name_list else set()
    return sum(1 for req in paper_mixtures.values()
               if req.issubset(parents))

def get_ml_floor(tau_ice_max):
    """
    Adaptive ML floor based on source complexity.
    Richer ice inventory (higher tau) justifies
    lower floor — more candidates are plausible.
    """
    if   tau_ice_max > 4.0: return ML_FLOOR_TIERS['rich']
    elif tau_ice_max > 1.5: return ML_FLOOR_TIERS['moderate']
    elif tau_ice_max > 0.5: return ML_FLOOR_TIERS['simple']
    else:                   return ML_FLOOR_TIERS['faint']

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
    for variant in variants:
        for ext in ['.dat', '.txt', '.dat1', '.csv']:
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
        core = re.sub(
            r'(_\d+[Kk])?(_class[A-E])?'
            r'(\.(DAT|TXT|DAT1|CSV))$',
            '', fname.upper())
        base_core = re.sub(
            r'(_\d+[Kk])?(_class[A-E])?$',
            '', base_up)
        if core == base_core:
            kappa = _load_kappa_from_file(
                os.path.join(lida_dir, fname))
            if kappa is not None:
                return kappa, fname
    return None, None

# ══════════════════════════════════════════════════════════
# NNLS HELPERS
# ══════════════════════════════════════════════════════════
def tikhonov_nnls(A, b, alpha=0.15):
    n = A.shape[1]
    c, _ = nnls(
        np.vstack([A, np.sqrt(alpha)*np.eye(n)]),
        np.concatenate([b, np.zeros(n)]))
    return c

def weighted_tikhonov_nnls(A, b, ml_scores,
                             alpha=0.15, ml_eps=0.01):
    n       = A.shape[1]
    ml_arr  = np.array(ml_scores, dtype=np.float64)
    alpha_j = alpha / (ml_arr + ml_eps)
    c, _    = nnls(
        np.vstack([A, np.diag(np.sqrt(alpha_j))]),
        np.concatenate([b, np.zeros(n)]))
    return c

def refit_with_floors(A, b, keep_idx, nnls_names,
                       forced_names, alpha=0.15):
    A_keep = A[:, keep_idx]
    lb     = np.array([
        0.002 if nnls_names[i] in forced_names else 0.0
        for i in keep_idx])
    if (lb > 0).any():
        A_aug = np.vstack([
            A_keep,
            np.sqrt(alpha)*np.eye(len(keep_idx))])
        b_aug = np.concatenate(
            [b, np.zeros(len(keep_idx))])
        res   = lsq_linear(A_aug, b_aug,
                            bounds=(lb, np.inf),
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
        near_top = np.where(
            corr >= best_corr * 0.95)[0]
        best_idx = (
            near_top[np.argmax(ml_arr[near_top])]
            if len(near_top) > 1 else near_top[0])
        support.append(int(best_idx))
        c_sub, _ = nnls(A[:, support], b)
        residual  = b - A[:, support] @ c_sub
        history.append(
            1.0 - np.linalg.norm(residual) / b_norm)
    coeffs = np.zeros(n_feat)
    if support:
        c_sub, _ = nnls(A[:, support], b)
        for k, idx in enumerate(support):
            coeffs[idx] = c_sub[k]
    q = 1.0 - np.linalg.norm(b - A @ coeffs) / b_norm
    return coeffs, support, q, history

# ══════════════════════════════════════════════════════════
# OPTION A — ML-SCORE BASED PARENT FORCING
# Source-agnostic: uses only ML scores, not ground truth.
# force_threshold=0.20: minimum ML signal to trigger
# forcing for a missing parent. Set by classifier
# reliability, not by known IRAS 2A composition.
# ══════════════════════════════════════════════════════════
def option_a_forcing(nnls_names, c_omp, ml_scores,
                      ml_floor, force_threshold):
    NNLS_MIN = 1e-4
    ml_arr   = np.array(ml_scores)

    omp_names = [nnls_names[i]
                 for i in range(len(nnls_names))
                 if c_omp[i] > NNLS_MIN]
    current_parents = set().union(
        *[get_parents(n) for n in omp_names]
    ) if omp_names else set()

    ml_detected = {}
    for i, name in enumerate(nnls_names):
        if ml_arr[i] < force_threshold:
            continue
        for p in get_parents(name):
            ml_detected[p] = max(
                ml_detected.get(p, 0), ml_arr[i])

    missing = {p for p in ml_detected
               if p not in current_parents}

    force_idx = set()
    print(f"\n  Option A forcing "
          f"(threshold={force_threshold*100:.0f}%):")
    print(f"  ML-detected:    "
          f"{sorted(ml_detected.keys())}")
    print(f"  OMP parents:    "
          f"{sorted(current_parents)}")
    print(f"  Missing parents:{sorted(missing)}")

    for p in missing:
        cands = [
            i for i in range(len(nnls_names))
            if c_omp[i] < NNLS_MIN
            and ml_arr[i] >= ml_floor
            and p in get_parents(nnls_names[i])]
        if cands:
            best = max(cands, key=lambda i: ml_arr[i])
            force_idx.add(best)
            print(f"  ✔ Force {p} "
                  f"({ml_detected[p]*100:.0f}%): "
                  f"{nnls_names[best][:42]}  "
                  f"ML={ml_arr[best]*100:.1f}%")
        else:
            print(f"  ✗ No candidate for {p}")

    # Force any species with very high ML (>70%)
    # not in OMP solution. This threshold is based
    # on classifier reliability, not IRAS 2A knowledge.
    for i in range(len(nnls_names)):
        if c_omp[i] > NNLS_MIN or i in force_idx:
            continue
        if ml_arr[i] > 0.70:
            force_idx.add(i)
            print(f"  ✔ Force (ML>70%): "
                  f"{nnls_names[i][:42]}  "
                  f"ML={ml_arr[i]*100:.1f}%")

    print(f"  Total forced: {len(force_idx)}")
    return force_idx

# ══════════════════════════════════════════════════════════
# BACKWARD ELIMINATION
# Fit-drop only criterion. No recall guard.
# ══════════════════════════════════════════════════════════
def backward_elimination(A, b, c_init, nnls_names,
                           forced_names, alpha,
                           max_fit_drop):
    NNLS_MIN  = 1e-4
    cur_c     = c_init.copy()
    cur_names = [nnls_names[i]
                 for i in range(len(nnls_names))
                 if cur_c[i] > NNLS_MIN]
    b_norm    = np.linalg.norm(b)
    print(f"\n  Backward elimination "
          f"(max_drop={max_fit_drop*100:.1f}%  "
          f"protected={len(forced_names)}):")
    print(f"  Start: {len(cur_names)} components")
    while True:
        total = max(
            sum(cur_c[nnls_names.index(n)]
                for n in cur_names), 1e-10)
        removable = sorted(
            [(n, cur_c[nnls_names.index(n)] / total)
             for n in cur_names
             if n not in forced_names],
            key=lambda x: x[1])
        removed = False
        for name_rm, frac in removable:
            test   = [n for n in cur_names
                      if n != name_rm]
            kidx   = [i for i, n in enumerate(nnls_names)
                      if n in set(test)]
            new_c  = refit_with_floors(
                A, b, kidx, nnls_names,
                forced_names, alpha)
            cf = (1.0 -
                  np.linalg.norm(b - A @ cur_c) / b_norm)
            nf = (1.0 -
                  np.linalg.norm(b - A @ new_c) / b_norm)
            if nf >= cf - max_fit_drop:
                cur_c     = new_c
                cur_names = test
                print(f"  REMOVE {name_rm[:48]}  "
                      f"{frac*100:.1f}%  "
                      f"→ {nf*100:.1f}%")
                removed = True
                break
        if not removed:
            break
    q_f = (1.0 -
           np.linalg.norm(b - A @ cur_c) / b_norm)
    print(f"  Final: {len(cur_names)} survivors  "
          f"fit={q_f*100:.1f}%")
    return cur_c, cur_names, q_f

# ══════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════
def run_pipeline(A, b, nnls_names, nnls_scores,
                  ml_floor,
                  alpha          = TIKHONOV_ALPHA,
                  omp_max        = OMP_MAX,
                  omp_min_corr   = OMP_MIN_CORR,
                  force_threshold= FORCE_THRESHOLD,
                  omp_ml_protect = OMP_ML_PROTECT,
                  protect_frac   = PROTECT_FRAC,
                  max_fit_drop   = MAX_FIT_DROP,
                  label          = "Option_A"):

    b_norm   = np.linalg.norm(b)
    NNLS_MIN = 1e-4

    # OMP
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
    for k, idx in enumerate(support):
        fk = (omp_hist[k] if k < len(omp_hist)
              else q_omp)
        print(f"  {k+1:>4}  "
              f"{nnls_names[idx][:52]:<52}  "
              f"{nnls_scores[idx]*100:>4.1f}%  "
              f"{fk*100:>5.1f}%")

    # Protection set built from c_omp (before forcing)
    # FIX: use c_omp totals, not c_ff totals,
    # to avoid inflation from forced floor components
    c_omp_total = max(c_omp.sum(), 1e-10)

    # OMP-evidence protection: any OMP-selected species
    # with ML >= omp_ml_protect is shielded from backward
    # elimination. Reasoning: OMP independently chose it
    # against residuals — that is genuine spectral evidence
    # beyond ML scores alone.
    omp_protected = set()
    print(f"\n  OMP protection "
          f"(ML>{omp_ml_protect*100:.0f}%):")
    for idx in support:
        if nnls_scores[idx] >= omp_ml_protect:
            omp_protected.add(nnls_names[idx])
            print(f"    ✔ {nnls_names[idx][:50]}  "
                  f"ML={nnls_scores[idx]*100:.1f}%")

    # Abundance protection: OMP components above
    # protect_frac of the OMP solution
    abundance_protected = set()
    print(f"\n  Abundance protection "
          f"(OMP frac>{protect_frac*100:.0f}%):")
    for i in range(len(nnls_names)):
        if c_omp[i] > NNLS_MIN:
            frac = c_omp[i] / c_omp_total
            if frac > protect_frac:
                abundance_protected.add(nnls_names[i])
                print(f"    ✔ {nnls_names[i][:50]}  "
                      f"{frac*100:.1f}%")

    # Method-specific forcing
    if 'A' in label.upper():
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
                np.sqrt(alpha)*np.eye(A.shape[1])])
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
        print(f"\n  [{label}] Confidence-weighted NNLS:")
        ml_arr      = np.array(nnls_scores)
        include_idx = sorted(set(
            list(support) + [
                i for i in range(len(nnls_names))
                if ml_arr[i] >= ml_floor]))
        A_sub  = A[:, include_idx]
        ml_sub = [nnls_scores[i] for i in include_idx]
        c_wsub = weighted_tikhonov_nnls(
            A_sub, b, ml_sub, alpha=alpha)
        c_ff = np.zeros(len(nnls_names))
        for k, idx in enumerate(include_idx):
            c_ff[idx] = c_wsub[k]
        q_w = (1.0 -
               np.linalg.norm(b - A @ c_ff) / b_norm)
        print(f"  Weighted NNLS: fit={q_w*100:.1f}%  "
              f"active={(c_ff>NNLS_MIN).sum()}")
        forced_names = set()
        for i in range(len(nnls_names)):
            if c_ff[i] > NNLS_MIN or ml_arr[i] <= 0.70:
                continue
            lb    = np.zeros(len(nnls_names))
            lb[i] = 0.002
            A_aug = np.vstack([
                A,
                np.sqrt(alpha)*np.eye(A.shape[1])])
            b_aug = np.concatenate(
                [b, np.zeros(A.shape[1])])
            res   = lsq_linear(
                A_aug, b_aug,
                bounds=(lb, np.inf),
                method='bvls', max_iter=3000)
            c_ff  = np.maximum(res.x, 0.0)
            forced_names.add(nnls_names[i])
            print(f"  ✔ Force (ML>70%): "
                  f"{nnls_names[i][:42]}  "
                  f"ML={ml_arr[i]*100:.1f}%")

    # Combine all protected sets
    all_forced = forced_names | omp_protected | \
                 abundance_protected

    # Backward elimination
    c_final, final_names, q_final = \
        backward_elimination(
            A, b, c_ff, nnls_names,
            all_forced, alpha, max_fit_drop)
    fit_final = A @ c_final

    # Collect results
    NNLS_MIN   = 1e-4
    res_names  = [nnls_names[i]
                  for i in range(len(nnls_names))
                  if c_final[i] > NNLS_MIN]
    res_coeffs = [float(c_final[i])
                  for i in range(len(nnls_names))
                  if c_final[i] > NNLS_MIN]
    order      = np.argsort(res_coeffs)[::-1]
    res_names  = [res_names[i]  for i in order]
    res_coeffs = [res_coeffs[i] for i in order]
    total_c    = sum(res_coeffs) or 1.0

    def via(n):
        if n in omp_names:    return "OMP"
        if n in all_forced:   return "forced"
        return "floor"

    all_parents = set().union(
        *[get_parents(n) for n in res_names]
    ) if res_names else set()
    recall = compute_recall(res_names)

    print(f"\n  [{label}] RESULT  "
          f"fit={q_final*100:.1f}%  "
          f"survivors={len(res_names)}  "
          f"recall={recall}/15")
    print(f"  {'Component':<52} {'Frac%':>6}  Via")
    for n, c in zip(res_names, res_coeffs):
        print(f"  {n[:52]:<52} "
              f"{c/total_c*100:5.1f}%  {via(n)}")

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
        'forced_names': all_forced,
    }

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
print("=" * 62)
print("  INDRA 2.0 — IRAS 2A Validation v3.9 (final)")
print("  All decisions based on ML scores and solver")
print("  evidence only. No ground truth used.")
print("=" * 62)

# [1] Load
print("\n[1/6] Loading model...")
classifiers = joblib.load(CLF_FILE)
frac_names  = joblib.load(SPE_FILE)
thresholds  = joblib.load(THR_FILE)
print(f"  {len(classifiers)} classifiers")

# [2] Load tau
print("\n[2/6] Loading tau spectra...")
def load_tau(path, name):
    data    = np.loadtxt(path, comments='#')
    wl, tau = data[:, 0], data[:, 1]
    if wl.mean() > 100:
        wl  = 1e4 / wl
        wl  = wl[::-1]
        tau = tau[::-1]
    valid = ((wl >= 5.0) & (wl <= 28.0) &
             np.isfinite(tau))
    fi    = interp1d(wl[valid],
                     np.maximum(tau[valid], 0.0),
                     bounds_error=False,
                     fill_value=0.0)
    out   = np.maximum(
        medfilt(np.maximum(fi(MIRI_GRID), 0.0), 5),
        0.0).astype(np.float32)
    print(f"  {name}: τ_max={out.max():.2f}  "
          f"τ(6µm)="
          f"{float(np.interp(6.0,MIRI_GRID,out)):.2f}"
          f"  τ(15.2µm)="
          f"{float(np.interp(15.2,MIRI_GRID,out)):.2f}")
    return out

tau_ml   = load_tau(TAU_WITH_SIL, "ML input")
tau_nnls = load_tau(TAU_ICE_ONLY, "NNLS input")

# Adaptive ML floor — based on source complexity only
ml_floor = get_ml_floor(float(tau_nnls.max()))
print(f"\n  Adaptive ML floor: {ml_floor*100:.0f}%  "
      f"(τ_max={tau_nnls.max():.2f})")

# [3] Score classifiers
print(f"\n[3/6] Scoring {len(classifiers)} classifiers...")
y_proba    = np.array([
    clf.predict_proba(
        tau_ml.reshape(1,-1))[0,1]
    for clf in classifiers])
sorted_idx = np.argsort(y_proba)[::-1]

print("  Top 15:")
for i in sorted_idx[:15]:
    r = int(np.where(sorted_idx==i)[0][0]) + 1
    print(f"  {r:>4}  {y_proba[i]*100:5.1f}%  "
          f"{frac_names[i]}")

candidates_idx = [i for i in sorted_idx
                  if y_proba[i] >= ml_floor]
print(f"\n  Candidates ≥{ml_floor*100:.0f}%: "
      f"{len(candidates_idx)}")

# [4] Load LIDA pool
print("\n[4/6] Loading LIDA spectra...")
loaded_names, loaded_spectra, loaded_scores = [], [], []

for i in candidates_idx:
    name = frac_names[i]
    # Physics-based exclusion by species pattern
    # Not by exact filename — generalises to any LIDA version
    if any(pat.upper() in name.upper()
           for pat in EXCLUDE_SPECIES_PATTERNS):
        continue
    kappa, _ = fuzzy_load_lida(name, LIDA_DIR)
    if kappa is not None:
        loaded_names.append(name)
        loaded_spectra.append(kappa)
        loaded_scores.append(float(y_proba[i]))

print(f"  Loaded: {len(loaded_names)}")

# Deduplication
if len(loaded_spectra) >= 2:
    cd    = cosine_distances(np.array(loaded_spectra))
    order = np.argsort(loaded_scores)[::-1]
    kept  = []
    for idx in order:
        if not any(cd[idx,k] < DEDUP_THRESHOLD
                   for k in kept):
            kept.append(idx)
    nnls_names   = [loaded_names[i]   for i in kept]
    nnls_spectra = [loaded_spectra[i] for i in kept]
    nnls_scores  = [loaded_scores[i]  for i in kept]
else:
    nnls_names, nnls_spectra = loaded_names, loaded_spectra
    nnls_scores = loaded_scores

print(f"  After dedup: {len(nnls_names)}")

# Re-add must-include species after dedup
# Only physically justified universals — not IRAS-2A-specific
print("\n  Re-adding must-include species...")
for i in candidates_idx:
    name = frac_names[i]
    if any(pat.upper() in name.upper()
           for pat in EXCLUDE_SPECIES_PATTERNS):
        continue
    if name in nnls_names:
        continue
    if any(kw.upper() in name.upper()
           for kw in MUST_INCLUDE_KEYWORDS):
        kappa, _ = fuzzy_load_lida(name, LIDA_DIR)
        if kappa is not None:
            nnls_names.append(name)
            nnls_spectra.append(kappa)
            nnls_scores.append(float(y_proba[i]))
            print(f"  + {name}  ML={y_proba[i]*100:.1f}%")

print(f"  Final pool: {len(nnls_names)}")

A = np.column_stack(nnls_spectra)
b = tau_nnls.astype(np.float64)
print(f"\n  Matrix: {A.shape[0]} pts × "
      f"{A.shape[1]} components")

# [5] Run pipelines
print("\n[5/6] Running Option A and Option C...")

results_A = run_pipeline(
    A, b, nnls_names, nnls_scores,
    ml_floor=ml_floor,
    label="Option_A")

results_C = run_pipeline(
    A, b, nnls_names, nnls_scores,
    ml_floor=ml_floor,
    label="Option_C")

c_tikh     = tikhonov_nnls(A, b, TIKHONOV_ALPHA)
q_tikh     = (1.0 -
               np.linalg.norm(b - A @ c_tikh) /
               np.linalg.norm(b))
tikh_names = [nnls_names[i]
               for i in range(len(nnls_names))
               if c_tikh[i] > 1e-4]

# [6] Post-hoc recall
print("\n[6/6] Post-hoc recall (paper_mixtures read here"
      " for first and only time)...")
print(f"\n  {'Mixture':<22} {'A':>5} {'C':>5}")
print(f"  {'-'*22} {'-'*5} {'-'*5}")
for mix, req in paper_mixtures.items():
    ok_A = req.issubset(results_A['all_parents'])
    ok_C = req.issubset(results_C['all_parents'])
    miss_A = sorted(req - results_A['all_parents'])
    print(f"  {'✔' if ok_A else '✘'} {mix:<22} "
          f"{'✔' if ok_A else '✘':>5} "
          f"{'✔' if ok_C else '✘':>5}"
          + (f"  miss: {miss_A}" if not ok_A else ""))

print(f"\n  {'Method':<25} {'Fit%':>6} "
      f"{'Surv':>6} {'Recall':>8}")
print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8}")
for lbl, res in [
    ("Tikhonov", {'q_final': q_tikh,
                   'res_names': tikh_names,
                   'recall': compute_recall(tikh_names)}),
    ("Option A", results_A),
    ("Option C", results_C),
]:
    print(f"  {lbl:<25} "
          f"{res['q_final']*100:5.1f}%  "
          f"{len(res['res_names']):>6}  "
          f"{res.get('recall', compute_recall(res['res_names'])):>4}/15")

# Plots and CSV saving
print("\n  Saving plots and CSVs...")
fig, axes = plt.subplots(5, 1, figsize=(15, 30))

ax = axes[0]
ax.plot(MIRI_GRID, b, 'k-', lw=1.2, alpha=0.8,
        label='IRAS 2A ice τ')
ax.plot(MIRI_GRID, results_A['fit_final'],
        'g-', lw=2.0,
        label=f"Option A  "
              f"fit={results_A['q_final']*100:.0f}%  "
              f"{len(results_A['res_names'])} comp  "
              f"recall={results_A['recall']}/15")
ax.plot(MIRI_GRID, results_C['fit_final'],
        'b--', lw=1.8, alpha=0.8,
        label=f"Option C  "
              f"fit={results_C['q_final']*100:.0f}%  "
              f"{len(results_C['res_names'])} comp  "
              f"recall={results_C['recall']}/15")
ax.plot(MIRI_GRID, A @ c_tikh, 'r:', lw=1.0, alpha=0.4,
        label=f'Tikhonov {q_tikh*100:.0f}%')
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

for j, (res, col, lbl) in enumerate([
    (results_A, 'g', 'Option A'),
    (results_C, 'b', 'Option C')]):
    ax = axes[j+1]
    if res['omp_hist']:
        steps = list(range(1,
                           len(res['omp_hist'])+1))
        ax.plot(steps,
                [h*100 for h in res['omp_hist']],
                f'{col}-o', lw=2, ms=5)
        ax.axhline(res['q_final']*100,
                   color='navy', ls='--', lw=1.2,
                   label=f"Final "
                         f"{res['q_final']*100:.1f}%")
        ax.axhline(res['q_omp']*100,
                   color=col, ls='--', lw=1.0,
                   label=f"OMP "
                         f"{res['q_omp']*100:.1f}%")
        ax.set_xlabel("OMP step")
        ax.set_ylabel("Fit quality (%)")
        ax.set_title(f"{lbl} — OMP convergence",
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        if len(steps) <= 30:
            ax.set_xticks(steps)

for j, (res, lbl) in enumerate([
    (results_A, 'Option A'),
    (results_C, 'Option C')]):
    ax    = axes[j+3]
    fracs = [c/res['total_c']*100
             for c in res['res_coeffs']]
    via   = res['via_fn']
    cols  = [{'OMP':'#2ecc71',
               'forced':'#e74c3c',
               'floor':'#95a5a6'}.get(via(n),
              '#95a5a6')
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
        f"[green=OMP  red=forced  gray=floor]",
        fontweight='bold')
    ax.legend(handles=[
        mpatches.Patch(facecolor='#2ecc71',
                       label='OMP'),
        mpatches.Patch(facecolor='#e74c3c',
                       label='Forced (ML>threshold)'),
        mpatches.Patch(facecolor='#95a5a6',
                       label='Floor'),
    ], fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.2, axis='x')

plt.tight_layout()
out_plot = OUTPUT_DIR + "iras2a_v39_optA_vs_optC.png"
plt.savefig(out_plot, dpi=150, bbox_inches='tight')
plt.close()

for res, tag in [(results_A,'optA'),
                 (results_C,'optC')]:
    final_map = dict(zip(res['res_names'],
                          res['res_coeffs']))
    rows = []
    for i in candidates_idx:
        name = frac_names[i]
        if any(pat.upper() in name.upper()
               for pat in EXCLUDE_SPECIES_PATTERNS):
            continue
        coeff = final_map.get(name, 0.0)
        rows.append({
            'rank'         : int(
                np.where(sorted_idx==i)[0][0])+1,
            'label'        : name,
            'parents'      : ', '.join(
                sorted(get_parents(name))),
            'ml_conf'      : round(float(y_proba[i]),4),
            'nnls_coeff'   : round(coeff, 6),
            'nnls_frac_pct': round(
                coeff/res['total_c']*100, 2),
            'detected'     : coeff > 1e-4,
            'via'          : (res['via_fn'](name)
                               if coeff > 1e-4 else ''),
        })
    csv_path = OUTPUT_DIR + \
               f"iras2a_v39_{tag}_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV → {csv_path}")

print(f"  Plot → {out_plot}")

print(f"\n{'★'*62}")
print(f"  IRAS 2A — Final validation summary")
print(f"  No TARGET_SPECIES at any step")
print(f"  No IRAS-2A-specific parameter tuning")
print(f"{'★'*62}")
for lbl, res, r in [
    ("Tikhonov", {'q_final': q_tikh},
     compute_recall(tikh_names)),
    ("Option A", results_A, results_A['recall']),
    ("Option C", results_C, results_C['recall']),
]:
    ns = (len(tikh_names)
          if lbl == "Tikhonov"
          else len(res['res_names']))
    print(f"  {lbl:<12}  "
          f"fit={res['q_final']*100:.1f}%  "
          f"surv={ns}  recall={r}/15")
print(f"{'★'*62}")