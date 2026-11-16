# ============================================================
# INDRA 2.0 v3.9 — Complete pipeline with all fixes
#
# Fixes applied:
#   F1: CH3NH2 removed from species map (hallucination)
#   F2: NH2CHO requires 50% ML confidence minimum
#   F3: Per-species ML confidence floor for ambiguous species
#   F4: Temperature sanity check per species
#   F5: Summary print loop NameError fixed
#   F6: Two-stage architecture (ML detect → multi-temp NNLS)
#   F7: All three continuum fixes from v3 pipeline
#       (L-BFGS-B, two-component silicate, 17.5+18µm anchors)
#
# Pipeline flow:
#   Flux → Continuum (L-BFGS-B) → tau total
#       → Silicate (ol + pyr Drude) → tau ice-only
#       → Stage 1: ML classifiers on tau WITH silicate
#       → Stage 2: Multi-temp NNLS on tau ice-only
#       → Temperature-sanity filtered results
#       → CSV + plots
# ============================================================

import numpy as np
import os
import re
import joblib
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.optimize import (nnls, lsq_linear,
                             minimize, minimize_scalar)
from sklearn.metrics.pairwise import cosine_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# PATHS — edit only here
# ══════════════════════════════════════════════════════════
BASE_DIR  = "C:/Users/sahil/Downloads/"
CLF_FILE  = BASE_DIR + "lgbm_classifiers_v38patch.pkl"
SPE_FILE  = BASE_DIR + "lgbm_species_names_v38patch.pkl"
THR_FILE  = BASE_DIR + "lgbm_thresholds_recall90_v38patch.pkl"
LIDA_DIR  = BASE_DIR + "lida_raw_data/"
MIRI_GRID = np.linspace(5.0, 28.0, 3000)

# ══════════════════════════════════════════════════════════
# SOURCE CONFIGURATIONS
# Anchor wavelengths are identical for all Class 0
# protostars — physics-based, not source-tuned
# ══════════════════════════════════════════════════════════
_LEFT_WAV   = np.array([5.00, 5.05, 5.10])
_EXTRA_WAV  = np.array([7.20, 7.30])
_FRAC_WAV   = np.array([8.10, 8.20, 12.8, 13.1, 13.4])
_RIGHT_WAV  = np.array([17.5, 18.0,          # FIX: added
                          19.6, 20.0, 20.4,
                          23.1, 23.5,
                          27.2, 27.6, 28.0])
_MIDDLE_WAV = np.array([5.5,  5.7,  6.0,  6.3,
                          6.6,  7.0,  7.5,  7.8,
                          8.5,  9.0,  9.5, 10.0,
                         10.5, 11.0, 11.5, 12.0,
                         12.5, 14.0, 14.5, 15.0,
                         16.0, 17.0, 18.5, 19.0,
                         21.0, 22.0, 24.0, 25.0,
                         26.0])

SOURCES = [
    {
        'name'       : 'IRAS_2A',
        'label'      : 'NGC1333 IRAS 2A (Class 0)',
        'mode'       : 'precomputed',  # use existing tau files
        'sil_tau'    : BASE_DIR +
            "iras2A_tau_with_silicate_reconstructed.dat",
        'ice_tau'    : BASE_DIR +
            "iras2A_global_opticaldepth.dat",
        'output_dir' : BASE_DIR + "IRAS2A_v39_results/",
    },
    {
        'name'       : 'L1448_IRS2',
        'label'      : 'L1448 IRS2 (Class 0, 5.2 Lsun)',
        'mode'       : 'from_flux',
        'flux_file'  : BASE_DIR + "IRS2B_C0_spectra.txt",
        'output_dir' : BASE_DIR + "L1448_IRS2_v39_results/",
        'left_wav'   : _LEFT_WAV,
        'extra_wav'  : _EXTRA_WAV,
        'frac_wav'   : _FRAC_WAV,
        'right_wav'  : _RIGHT_WAV,
        'middle_wav' : _MIDDLE_WAV,
        'w_anchor'   : 5.3,
        'w_curve'    : 13.5,
        'poly_deg'   : 4,
    },
    {
        'name'       : 'L1448_IRS3B',
        'label'      : 'L1448 IRS3B (Class 0 binary)',
        'mode'       : 'from_flux',
        'flux_file'  : BASE_DIR + "IRS3B_B_C0_spectra.txt",
        'output_dir' : BASE_DIR + "L1448_IRS3B_v39_results/",
        'left_wav'   : _LEFT_WAV,
        'extra_wav'  : _EXTRA_WAV,
        'frac_wav'   : _FRAC_WAV,
        'right_wav'  : _RIGHT_WAV,
        'middle_wav' : _MIDDLE_WAV,
        'w_anchor'   : 5.3,
        'w_curve'    : 13.5,
        'poly_deg'   : 4,
    },
]

# ══════════════════════════════════════════════════════════
# SPECIES MAPS AND CONSTRAINTS
#
# F1: CH3NH2 removed entirely — hallucination in v3.8patch.
#     Its N-H bending mode at 6.8 µm is fully degenerate
#     with NH4+ and H2O:NH3 mixtures. No confirmed
#     interstellar ice detection exists for methylamine.
#
# F2/F3: Per-species minimum ML confidence floor.
#     Species whose bands strongly overlap with dominant
#     ices require higher ML confidence before inclusion.
# ══════════════════════════════════════════════════════════
PARENT_TO_LIDA_KEYWORDS = {
    # Simple ices
    'H2O'      : ['H2O', 'H2Oice', 'PureH2O',
                  'H2O160K', 'H2Ogsh'],
    'CO2'      : ['CO2', 'C18O2', 'PureCO2',
                  'PureC18O2', 'H2O_CO2'],
    'CO'       : ['PureCO', 'COover', 'COunder',
                  'CO_CO2', 'CO_N2', 'CO_CH',
                  'COoverCO2', 'COunderCO2'],
    'CH4'      : ['CH4', 'PureCH4', 'H2O_CH4',
                  'CO2_CH4'],
    # Oxygen-bearing COMs
    'CH3OH'    : ['CH3OH', 'PureCH3OH',
                  'HCOOH_H2O_CH3OH',
                  'H2O_CH3OH'],
    'HCOOH'    : ['HCOOH', 'PureHCOOH',
                  'HCOOH_H2O', 'HCOOH6pct',
                  'HCOOH11pct', 'CO_HCOOH'],
    'HCOO-'    : ['H2O_NH3_HCOOH', 'Formate',
                  'HCOO', 'NH4HCOO'],
    'H2CO'     : ['H2CO', 'PureH2CO',
                  'H2CO_TvS', 'CH3OCHO_H2CO'],
    'CH3CHO'   : ['CH3CHO', 'PureCH3CHO',
                  'H2O_CH3CHO'],
    'CH3CH2OH' : ['CH3CH2OH', 'C2H5OH',
                  'CH3CH2OH_H2O',
                  'CO_CH3CH2OH'],
    'CH3COOH'  : ['CH3COOH', 'H2O_CH3COOH',
                  'AceticAcid'],
    'CH3OCHO'  : ['CH3OCHO', 'HCOOCH3',
                  'CH3OCHO_CO', 'CH3OCHO_H2CO'],
    # Nitrogen-bearing species
    'NH3'      : ['NH3', 'PureNH3', 'H2O_NH3',
                  'NH3_CH3OH'],
    'NH4+'     : ['NHplus4', 'NH4plus', 'NH4+',
                  'NH4COOH', 'Salt_residue',
                  'h2o-nh3-h2s'],
    'OCN-'     : ['OCN', 'HNCO_NH3',
                  'NH3_HNCO'],
    'HNCO'     : ['HNCO_NH3', 'HNCO'],
    # F2: NH2CHO requires 50% ML — ambiguous bands
    'NH2CHO'   : ['NH2CHO', 'Formamide',
                  'PureNH2CHO'],
    # Sulfur-bearing
    'H2S'      : ['H2S', 'h2o-nh3-h2s'],
    'SO2'      : ['SO2', 'PureSO2', 'SO2_CH3OH'],
    # Solids
    'organic'  : ['Spec_A', 'Spec_B', 'Spec_C',
                  'Spec_D', 'Organic', 'Residue',
                  'NH4COOH_H2O'],
    'Silicate' : ['SILICATE', 'Silicate',
                  'olivine', 'pyroxene', 'GCS3'],
    # Minor species
    'OCS'      : ['OCS', 'PureOCS'],
    # F1: CH3NH2 REMOVED — hallucination
    # Bands degenerate with NH4+, no Class 0 detection
}

# F3: Minimum ML confidence required per species
# Default 0.0 (no minimum) unless listed here
MIN_ML_CONFIDENCE = {
    'NH2CHO'   : 0.50,   # F2: very ambiguous
    'OCS'      : 0.50,   # rare, easily confused
    'H2CO'     : 0.15,   # weak but real
    'CH3CH2OH' : 0.10,   # low ML expected, keep
    'CH3OCHO'  : 0.10,
    'CH3CHO'   : 0.10,
    'CH3COOH'  : 0.10,
}

# F4: Physical temperature sanity bounds (K)
# Species found at temperatures outside these ranges
# are physically inconsistent and removed from results
TEMP_SANITY_BOUNDS = {
    'CO'       : (5,   55),   # volatile, cold only
    'CH4'      : (5,   35),   # volatile, cold only
    'CO2'      : (5,  120),   # survives to warm regions
    'H2CO'     : (5,  100),
    'CH3CH2OH' : (10,  80),   # COM, moderate temp
    'CH3CHO'   : (10, 100),
    'CH3OH'    : (10, 160),   # wide range observed
    'HCOOH'    : (10, 135),
    # NH4+, H2S at 135K are correct for warm envelope
    # No bound needed — let NNLS decide
}

# Preferred temperature ranges for file selection
PREFERRED_TEMP_RANGES = {
    'H2O'      : (10,  160),
    'CO2'      : (10,  120),
    'CO'       : (10,   55),
    'CH4'      : (10,   30),
    'CH3OH'    : (15,  160),
    'NH3'      : (10,  135),
    'NH4+'     : (12,  135),
    'H2S'      : (10,  135),
    'SO2'      : (10,   15),
    'HCOOH'    : (15,  135),
    'HCOO-'    : (14,   80),
    'OCN-'     : (80,   80),
    'HNCO'     : (80,   80),
    'H2CO'     : (15,   80),
    'CH3CHO'   : (15,  120),
    'CH3CH2OH' : (15,  160),
    'CH3COOH'  : (16,   16),
    'CH3OCHO'  : (12,  120),
    'NH2CHO'   : (15,  170),
    'organic'  : (12,  160),
    'Silicate' : (0,   999),
    'OCS'      : (10,   80),
}

# Literature reference — Boogert+2008 c2d, Perseus Class 0
C2D_LITERATURE = {
    'H2O'     : (100.0, 100.0),
    'CO2'     : (11.0,  38.0),
    'CH4'     : (2.0,   8.0),
    'CH3OH'   : (1.0,   25.0),
    'NH3'     : (2.0,   15.0),
    'NH4+'    : (4.0,   15.0),
    'CO'      : (3.0,   60.0),
    'OCN-'    : (0.1,   1.1),
    'HCOOH'   : (0.5,   4.0),
    'HCOO-'   : (0.3,   2.3),
    'SO2'     : (0.08,  0.76),
    'H2CO'    : (2.0,   7.0),
}

# Prathap's 15 components for IRAS 2A recall check
IRAS2A_PAPER_COMPONENTS = {
    'H2O', 'CO2', 'NH4+', 'organic',
    'CH3OH', 'SO2', 'CH3OCHO', 'CO',
    'HCOOH', 'CH4', 'NH3', 'CH3CHO',
    'CH3CH2OH', 'HNCO', 'OCN-',
}

# ══════════════════════════════════════════════════════════
# PART 1 — FLUX → OPTICAL DEPTH
# Uses Prathap's continuum method with three fixes
# ══════════════════════════════════════════════════════════

# ── Drude silicate profiles (two-component, FIX) ─────────
def drude(wl, tau_peak, wl0, gamma):
    return (tau_peak * (wl/wl0)**2 * gamma**2 /
            ((wl**2 - wl0**2)**2 +
             (wl * gamma)**2) * wl0**2)

def silicate_two_component(wl, tau_ol, tau_pyr):
    """
    Two-component interstellar silicate.
    Olivine (MgFeSiO4): 9.8 µm + 18 µm secondary
    Pyroxene (Mg0.5Fe0.5SiO3): 9.65 µm + 18 µm
    Parameters from Weingartner & Draine 2001 /
    Prathap's INDRA paper Table 1.
    Source-agnostic: same profile for all protostars.
    """
    ol_97  = drude(wl, tau_ol,        9.80, 1.00)
    ol_18  = drude(wl, tau_ol*0.40,  18.00, 0.45)
    pyr_97 = drude(wl, tau_pyr,       9.65, 0.85)
    pyr_18 = drude(wl, tau_pyr*0.35, 18.00, 0.40)
    return ol_97 + ol_18 + pyr_97 + pyr_18

def fit_silicate_two_component(wl_raw, tau_raw):
    sil_mask = (((wl_raw >= 8.0) & (wl_raw <= 8.5)) |
                ((wl_raw >= 11.8) & (wl_raw <= 12.5)))
    sil_wl   = wl_raw[sil_mask]
    sil_tau  = tau_raw[sil_mask]
    if len(sil_wl) < 10:
        # Fallback: single Drude
        res = minimize_scalar(
            lambda tp: np.sum(
                (sil_tau - drude(sil_wl, tp, 9.7, 0.97)
                 - drude(sil_wl, tp*0.4, 18.0, 0.45))**2),
            bounds=(0.1, 5.0), method='bounded')
        return res.x, 0.0
    def resid(params):
        to, tp = params
        if to < 0.05 or tp < 0.05: return 1e8
        if to > 5.0  or tp > 5.0:  return 1e8
        return np.sum(
            (sil_tau - silicate_two_component(
                sil_wl, to, tp))**2)
    best_res, best_val = None, np.inf
    for to0 in [0.5, 1.0, 2.0]:
        for tp0 in [0.5, 1.0, 2.0]:
            res = minimize(resid, [to0, tp0],
                           method='Nelder-Mead',
                           options={'maxiter': 5000,
                                    'xatol': 1e-6,
                                    'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
    tau_ol  = max(best_res.x[0], 0.05)
    tau_pyr = max(best_res.x[1], 0.05)
    return tau_ol, tau_pyr

# ── L-BFGS-B continuum fitter (FIX) ──────────────────────
def fit_continuum_lbfgsb(wavelength, flux, cfg):
    """
    Prathap's log-log polynomial continuum with
    L-BFGS-B optimizer and soft exponential barrier.
    Converges in ~1000 iterations vs Nelder-Mead's
    200k+ needed for 34 parameters.
    """
    w_anchor  = cfg['w_anchor']
    w_curve   = cfg['w_curve']
    poly_deg  = cfg['poly_deg']
    BARRIER_K = 20.0

    fl_interp = interp1d(wavelength, flux,
                         bounds_error=False,
                         fill_value='extrapolate')
    def get_flux(wav):
        return np.maximum(fl_interp(wav), 1e-15)

    wl_min, wl_max = wavelength.min(), wavelength.max()
    def in_range(arr):
        return arr[(arr >= wl_min) & (arr <= wl_max)]

    left_wav  = in_range(cfg['left_wav'])
    right_wav = in_range(cfg['right_wav'])
    extra_wav = in_range(cfg['extra_wav'])
    frac_wav  = in_range(cfg['frac_wav'])
    mid_wav   = in_range(cfg['middle_wav'])

    def filt(w, f):
        ok = f > 0
        return w[ok], f[ok]

    left_wav,  left_flux  = filt(left_wav,
                                  get_flux(left_wav))
    right_wav, right_flux = filt(right_wav,
                                  get_flux(right_wav))
    extra_wav, extra_flux = filt(extra_wav,
                                  get_flux(extra_wav))
    frac_wav,  frac_flux  = filt(frac_wav,
                                  get_flux(frac_wav))
    mid_wav,   mid_flux   = filt(mid_wav,
                                  get_flux(mid_wav))

    ok_obs        = flux > 0
    log_wav_full  = np.log(wavelength[ok_obs])
    log_flux_obs  = np.log(flux[ok_obs])

    fixed_wav     = np.concatenate([
        left_wav, frac_wav, right_wav, extra_wav])
    fixed_flux    = np.concatenate([
        left_flux, frac_flux, right_flux, extra_flux])
    log_fixed_wav  = np.log(fixed_wav)
    log_fixed_flux = np.log(fixed_flux)
    log_mid_wav    = np.log(mid_wav)
    log_mid_flux   = np.log(mid_flux)
    n_free         = len(mid_wav)
    n_coeff        = poly_deg + 1

    def DM(lw):
        return np.vstack(
            [lw**i for i in range(poly_deg+1)]).T

    def CM(lw):
        cols = [np.zeros_like(lw),
                np.zeros_like(lw),
                np.full_like(lw, 2.0),
                6.0 * lw,
                12.0 * lw**2]
        return np.vstack(cols[:n_coeff]).T

    A_fixed  = DM(log_fixed_wav)
    A_middle = DM(log_mid_wav)
    A_full   = DM(log_wav_full)
    A_curv   = CM(log_wav_full)

    A_init = np.vstack([A_fixed, A_middle])
    logf_i = np.concatenate([log_fixed_flux,
                               log_mid_flux])
    c0, _, _, _ = np.linalg.lstsq(A_init, logf_i,
                                    rcond=None)
    x0 = np.concatenate([c0, np.zeros(n_free)])

    def objective(x):
        coeffs     = x[:n_coeff]
        raw_deltas = x[n_coeff:]
        deltas     = np.maximum(raw_deltas, 0.0)
        log_cont   = A_full @ coeffs
        below_obs  = log_flux_obs - log_cont
        if np.any(below_obs > 0):
            barrier = np.sum(np.exp(np.minimum(
                BARRIER_K * below_obs[below_obs > 0],
                50.0)))
        else:
            barrier = 0.0
        fixed_res  = np.sum(
            (A_fixed @ coeffs - log_fixed_flux)**2)
        log_adj    = log_mid_flux + np.log1p(deltas)
        middle_res = np.sum(
            (A_middle @ coeffs - log_adj)**2)
        anchor_t   = w_anchor * (fixed_res + middle_res)
        curv_t     = w_curve * np.mean(
            (A_curv @ coeffs)**2)
        return anchor_t + curv_t + barrier

    bounds = ([(None, None)] * n_coeff +
               [(0.0,  None)] * n_free)
    result = minimize(objective, x0,
                      method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 10000,
                               'ftol'   : 1e-12,
                               'gtol'   : 1e-8})
    converged = result.success or result.fun < 1.0
    print(f"    Optimizer: "
          f"{'converged' if converged else 'stopped'}"
          f"  fun={result.fun:.4f}  nit={result.nit}")

    coeffs_final = result.x[:n_coeff]
    log_wav_all  = np.log(
        np.maximum(wavelength, 1e-10))
    A_all        = DM(log_wav_all)
    continuum    = np.exp(A_all @ coeffs_final)
    return np.maximum(continuum, flux)


def process_flux_to_tau(flux_file, cfg):
    """Convert flux spectrum to tau files."""
    source_name = cfg['name']
    output_dir  = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Part 1: {source_name} — Flux → τ")
    print(f"{'='*60}")

    raw    = np.loadtxt(flux_file, comments='#')
    wl_raw = raw[:, 0]
    fl_raw = raw[:, 1]
    if wl_raw[0] > wl_raw[-1]:
        wl_raw = wl_raw[::-1]
        fl_raw = fl_raw[::-1]
    mask   = ((wl_raw >= 4.85) & (wl_raw <= 28.6) &
               (fl_raw > 0) & np.isfinite(fl_raw))
    wl_raw = wl_raw[mask]
    fl_raw = fl_raw[mask]
    print(f"  Loaded: {len(wl_raw)} pts  "
          f"{wl_raw.min():.2f}–{wl_raw.max():.2f} µm")

    emission_lines = [
        5.051, 5.340, 5.511, 6.109, 6.634, 6.910,
        8.025, 9.665, 10.511, 12.279, 12.814,
        17.035, 17.936, 25.249, 25.988,
    ]
    mask_good = np.ones(len(wl_raw), dtype=bool)
    for line in emission_lines:
        mask_good &= ~((wl_raw > line - 0.06) &
                       (wl_raw < line + 0.06))
    fl_med     = medfilt(fl_raw, kernel_size=21)
    mask_good &= ~(fl_raw > 5.0 * fl_med)
    print(f"  Masked: {(~mask_good).sum()} pixels")

    fl_clean = fl_raw.copy()
    fl_clean[~mask_good] = fl_med[~mask_good]
    fl_clean = np.maximum(fl_clean, 1e-10)

    print(f"\n  Fitting continuum (L-BFGS-B)...")
    continuum = fit_continuum_lbfgsb(
        wl_raw, fl_clean, cfg)

    ratio   = continuum / np.maximum(fl_clean, 1e-15)
    ratio   = np.maximum(ratio, 1.0)
    tau_raw = np.log(ratio)
    tau_raw = np.maximum(tau_raw, 0.0)

    def tau_at(wl0):
        return float(np.interp(wl0, wl_raw, tau_raw))

    t6   = tau_at(6.0)
    t77  = tau_at(7.7)
    t97  = tau_at(9.7)
    t152 = tau_at(15.2)
    t20  = tau_at(20.0)
    print(f"\n  τ(6.0µm) H₂O  = {t6:.3f}")
    print(f"  τ(7.7µm) CH₄  = {t77:.3f}")
    print(f"  τ(9.7µm) Sil  = {t97:.3f}")
    print(f"  τ(15.2µm) CO₂ = {t152:.3f}")
    print(f"  τ(20µm)       = {t20:.3f}  "
          f"(target: <0.15)")
    if t6 < 0.3:
        print(f"  ⚠ τ(6µm) low — check left anchors")
    if t77 < 0.05:
        print(f"  ⚠ τ(7.7µm) low — CH4 may be missed")
    if t20 > 0.2:
        print(f"  ⚠ τ(20µm) elevated — "
              f"continuum needs right anchor tuning")

    print(f"\n  Two-component silicate subtraction...")
    tau_ol, tau_pyr = fit_silicate_two_component(
        wl_raw, tau_raw)
    tau_sil = silicate_two_component(
        wl_raw, tau_ol, tau_pyr)
    tau_ice = np.maximum(tau_raw - tau_sil, 0.0)
    print(f"  Olivine  τ = {tau_ol:.3f}")
    print(f"  Pyroxene τ = {tau_pyr:.3f}")
    print(f"  τ_ice max  = {tau_ice.max():.3f}")

    def to_grid(wl, tau):
        valid = ((wl >= 5.0) & (wl <= 28.0) &
                 np.isfinite(tau) & (tau >= 0))
        if valid.sum() < 10:
            return np.zeros(len(MIRI_GRID))
        fi  = interp1d(wl[valid], tau[valid],
                       bounds_error=False,
                       fill_value=0.0)
        return np.maximum(
            medfilt(np.maximum(fi(MIRI_GRID), 0.0), 5),
            0.0)

    tau_sil_grid = to_grid(wl_raw, tau_raw)
    tau_ice_grid = to_grid(wl_raw, tau_ice)

    sn = source_name
    np.savetxt(
        output_dir + f"{sn}_v39_tau_ice.dat",
        np.column_stack([MIRI_GRID, tau_ice_grid]),
        fmt='%.6f  %.6f',
        header='wavelength(um)  tau_ice')
    np.savetxt(
        output_dir + f"{sn}_v39_tau_withsil.dat",
        np.column_stack([MIRI_GRID, tau_sil_grid]),
        fmt='%.6f  %.6f',
        header='wavelength(um)  tau_with_sil')

    # Diagnostic plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    ax = axes[0]
    ax.semilogy(wl_raw, fl_clean, 'k-', lw=0.7,
                alpha=0.6, label='Observed flux')
    ax.semilogy(wl_raw, continuum, 'g--', lw=2.0,
                label='Continuum (L-BFGS-B)')
    fl_i = interp1d(wl_raw, fl_clean,
                    bounds_error=False,
                    fill_value='extrapolate')
    for wl_arr, col, lab in [
        (_LEFT_WAV,  'blue',   'Left fixed'),
        (_RIGHT_WAV, 'red',    'Right fixed'),
        (_FRAC_WAV,  'orange', 'Frac anchors'),
        (_EXTRA_WAV, 'purple', 'Extra fixed'),
    ]:
        wl_ok = wl_arr[(wl_arr >= wl_raw.min()) &
                        (wl_arr <= wl_raw.max())]
        if len(wl_ok):
            ax.semilogy(wl_ok, fl_i(wl_ok), 'o',
                        color=col, ms=5,
                        label=lab, zorder=5)
    ax.set_ylabel("Flux (Jy)")
    ax.set_title(f"{source_name} — Continuum fit",
                 fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(5, 28)
    ax.grid(True, alpha=0.2)
    ax = axes[1]
    ax.plot(wl_raw, tau_raw, 'k-', lw=0.8,
            alpha=0.7, label='Total τ')
    ax.plot(wl_raw, tau_sil, 'r--', lw=1.5,
            label=f'Silicate: ol({tau_ol:.2f})'
                  f'+pyr({tau_pyr:.2f})')
    ax.plot(wl_raw, tau_ice, 'g-', lw=1.5,
            label='Ice-only τ → NNLS input')
    for wl_b, lbl in [(6.0,'H₂O'), (6.85,'NH₄⁺'),
                       (7.7,'CH₄'), (9.7,'Sil'),
                       (15.2,'CO₂')]:
        ax.axvline(wl_b, color='steelblue',
                   lw=0.7, ls=':', alpha=0.5)
        if tau_raw.max() > 0:
            ax.text(wl_b+0.1, tau_raw.max()*0.82,
                    lbl, fontsize=7, color='steelblue')
    ax.set_ylabel("Optical depth τ")
    ax.set_title("Two-component silicate subtraction",
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(5, 28)
    ax.grid(True, alpha=0.2)
    ax = axes[2]
    ax.plot(MIRI_GRID, tau_ice_grid, 'g-', lw=1.5,
            label=f'{source_name} ice τ')
    try:
        d2a = np.loadtxt(
            BASE_DIR + "iras2A_global_opticaldepth.dat",
            comments='#')
        wl2a, t2a = d2a[:,0], d2a[:,1]
        if wl2a.mean() > 100:
            wl2a = 1e4/wl2a; wl2a=wl2a[::-1]
            t2a  = t2a[::-1]
        ax.plot(wl2a, t2a, 'k--', lw=0.8, alpha=0.3,
                label='IRAS 2A (reference)')
    except Exception:
        pass
    ax.set_ylabel("Optical depth τ")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_title("Ice-only τ vs IRAS 2A",
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(5, 28)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir + f"{sn}_v39_tau_diag.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Tau diag → "
          f"{output_dir}{sn}_v39_tau_diag.png")
    return tau_sil_grid, tau_ice_grid

# ══════════════════════════════════════════════════════════
# LOAD PRE-COMPUTED TAU (for IRAS 2A)
# ══════════════════════════════════════════════════════════
def load_tau_files(sil_path, ice_path):
    """Load pre-computed tau files onto MIRI grid."""
    def _load(path):
        try:
            d  = np.loadtxt(path, comments='#')
            wl = d[:, 0]
            t  = d[:, 1]
            if wl.mean() > 100:
                wl = 1e4/wl; wl=wl[::-1]; t=t[::-1]
            fi = interp1d(wl, t,
                          bounds_error=False,
                          fill_value=0.0)
            return np.maximum(fi(MIRI_GRID), 0.0)
        except Exception as e:
            print(f"  ERROR loading {path}: {e}")
            return np.zeros(len(MIRI_GRID))
    return _load(sil_path), _load(ice_path)


# ══════════════════════════════════════════════════════════
# PARENT RULES (for get_parents helper)
# ══════════════════════════════════════════════════════════
PARENT_RULES = [
    ('CH3CH2OH','CH3CH2OH'),('C2H5OH','CH3CH2OH'),
    ('CH3COOH','CH3COOH'),  ('CH3OCHO','CH3OCHO'),
    ('CH3CHO','CH3CHO'),    ('CH3OH','CH3OH'),
    ('CH3NH2','CH3NH2'),    ('CH4','CH4'),
    ('H2CO','H2CO'),        ('HCOOH','HCOOH'),
    ('HCOO','HCOO-'),       ('NH2CHO','NH2CHO'),
    ('HNCO','HNCO'),        ('NH4','NH4+'),
    ('NHPLUS','NH4+'),      ('NH3','NH3'),
    ('H2S','H2S'),          ('SO2','SO2'),
    ('CO2','CO2'),          ('C18O','CO2'),
    ('H2O','H2O'),          ('CO','CO'),
    ('SILICATE','Silicate'),('OLIVINE','Silicate'),
    ('SPEC_D','organic'),   ('SPEC_A','organic'),
    ('ORGANIC','organic'),  ('RESIDUE','organic'),
    ('C3H6O','CH3CHO'),     ('OCS','OCS'),
    ('OCN','OCN-'),         ('OCN','HNCO'),
    ('OCN','NH3'),          ('HNCO_NH3','HNCO'),
    ('HNCO_NH3','NH3'),
]

def get_parents(label):
    lu = label.upper()
    return {p for k, p in PARENT_RULES
            if k.upper() in lu}


# ══════════════════════════════════════════════════════════
# LIDA FILE LOADER
# ══════════════════════════════════════════════════════════
def _load_kappa(path):
    try:
        with open(path, 'rb') as f:
            sig = f.read(4)
        if sig[:2] in [b'\xff\xfe', b'\xfe\xff']:
            return None
        with open(path, 'r', encoding='latin-1',
                  errors='replace') as fh:
            lines = fh.readlines()
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
        d   = np.array(rows)
        wl  = d[:, 0]
        kap = d[:, 1]
        if wl.mean() > 100:
            wl  = 1e4/wl; wl=wl[::-1]; kap=kap[::-1]
        valid = ((wl >= 5.0) & (wl <= 28.0) &
                 np.isfinite(kap) & (kap >= 0))
        if valid.sum() < 30:
            return None
        fi  = interp1d(wl[valid], kap[valid],
                       bounds_error=False,
                       fill_value=0.0)
        kg  = fi(MIRI_GRID)
        km  = kg.max()
        return kg/km if km > 0 else None
    except Exception:
        return None


def extract_temperature(filename):
    """Extract temperature (K) from LIDA filename."""
    match = re.search(
        r'[_\-](\d+(?:\.\d+)?)[Kk]', filename)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


# ══════════════════════════════════════════════════════════
# ADAPTIVE ML FLOOR
# ══════════════════════════════════════════════════════════
def adaptive_params(tau_ice_grid, source_name):
    tau_max = float(tau_ice_grid.max())
    if tau_max > 4.0:
        floor, tier = 0.05, "rich"
    elif tau_max > 1.5:
        floor, tier = 0.08, "moderate"
    elif tau_max > 0.5:
        floor, tier = 0.10, "simple"
    else:
        floor, tier = 0.15, "faint"
    print(f"\n  Source tier: {tier} "
          f"(τ_max={tau_max:.2f})")
    print(f"  → ML floor: {floor*100:.0f}%")
    return floor, tau_max, tier


# ══════════════════════════════════════════════════════════
# STAGE 1 — ML DETECTION
# ══════════════════════════════════════════════════════════
def ml_detect_parents(tau_sil_grid, classifiers,
                       frac_names, ml_floor):
    """
    Run ML classifiers on tau WITH silicate.
    Returns detected parent species + confidence scores.
    All fixes applied:
      F1: CH3NH2 excluded via PARENT_TO_LIDA_KEYWORDS
      F2/F3: Per-species minimum ML confidence check
    """
    X       = tau_sil_grid.reshape(1,-1).astype(
        np.float32)
    y_proba = np.array([
        clf.predict_proba(X)[0, 1]
        for clf in classifiers])
    sorted_idx = np.argsort(y_proba)[::-1]

    print(f"\n  Stage 1 — ML detection "
          f"(floor={ml_floor*100:.0f}%):")
    print(f"  {'Rank':>4}  {'Score':>6}  Label")
    for i in sorted_idx[:15]:
        r = int(np.where(sorted_idx==i)[0][0]) + 1
        print(f"  {r:>4}  "
              f"{y_proba[i]*100:5.1f}%  "
              f"{frac_names[i]}")

    detected_parents = set()
    species_scores   = {}

    for i in sorted_idx:
        if y_proba[i] < ml_floor:
            break
        for p in get_parents(frac_names[i]):
            if p not in species_scores:
                species_scores[p] = y_proba[i]
            detected_parents.add(p)

    # Always include >70% confidence species
    for i in range(len(frac_names)):
        if y_proba[i] > 0.70:
            for p in get_parents(frac_names[i]):
                if p not in species_scores:
                    species_scores[p] = y_proba[i]
                detected_parents.add(p)

    # F3: Apply per-species minimum confidence
    excluded = []
    for sp in list(detected_parents):
        min_conf = MIN_ML_CONFIDENCE.get(sp, 0.0)
        actual   = species_scores.get(sp, 0)
        if actual < min_conf:
            detected_parents.discard(sp)
            excluded.append(
                f"{sp} (ML={actual*100:.1f}% "
                f"< required {min_conf*100:.0f}%)")

    # F1: Remove CH3NH2 unconditionally
    if 'CH3NH2' in detected_parents:
        detected_parents.discard('CH3NH2')
        excluded.append("CH3NH2 (excluded: "
                         "hallucination species)")

    if excluded:
        print(f"\n  Excluded species:")
        for ex in excluded:
            print(f"    ✗ {ex}")

    # Only keep species in our keyword map
    detected_parents = {
        p for p in detected_parents
        if p in PARENT_TO_LIDA_KEYWORDS}

    print(f"\n  Detected parent species "
          f"({len(detected_parents)}):")
    for sp in sorted(detected_parents):
        score = species_scores.get(sp, 0)
        print(f"    {sp:<15} ML = {score*100:.1f}%")

    return detected_parents, species_scores, y_proba


# ══════════════════════════════════════════════════════════
# STAGE 2a — BUILD MULTI-TEMPERATURE BASIS
# ══════════════════════════════════════════════════════════
def build_multitemp_basis(detected_parents,
                           species_scores,
                           lida_dir,
                           max_per_species=8,
                           dedup_threshold=0.02):
    """
    For each detected parent, load all temperature
    variants from LIDA within preferred temp range.
    Deduplicate spectrally near-identical files.
    """
    print(f"\n  Stage 2a — Multi-temp NNLS basis...")
    try:
        all_files = sorted(os.listdir(lida_dir))
    except Exception as e:
        print(f"  ERROR: {e}")
        return [], [], [], []

    basis_names   = []
    basis_spectra = []
    basis_parents = []
    basis_temps   = []

    for parent in sorted(detected_parents):
        keywords  = PARENT_TO_LIDA_KEYWORDS.get(
            parent, [parent])
        t_lo, t_hi = PREFERRED_TEMP_RANGES.get(
            parent, (0, 999))

        # Find matching files
        candidates = []
        for fname in all_files:
            fu = fname.upper()
            if not any(kw.upper() in fu
                       for kw in keywords):
                continue
            if not fname.endswith(
                    ('.dat', '.txt', '.dat1', '.csv')):
                continue
            temp = extract_temperature(fname)
            if temp is not None:
                if not (t_lo <= temp <= t_hi):
                    continue
            candidates.append((fname, temp))

        if not candidates:
            print(f"    {parent:<15} — no files found")
            continue

        candidates.sort(
            key=lambda x: x[1] if x[1] else 999)

        # Load and deduplicate
        kept    = []
        kept_sp = []
        for fname, temp in candidates:
            path  = os.path.join(lida_dir, fname)
            kappa = _load_kappa(path)
            if kappa is None:
                continue
            is_dup = False
            for ks in kept_sp:
                nk = np.linalg.norm(kappa)
                ns = np.linalg.norm(ks)
                if nk > 0 and ns > 0:
                    cos_sim = np.dot(kappa, ks) / (nk*ns)
                    if cos_sim > (1 - dedup_threshold):
                        is_dup = True
                        break
            if not is_dup:
                kept.append((fname, kappa, temp))
                kept_sp.append(kappa)

        # Sample evenly across temperature range
        if len(kept) > max_per_species:
            step = len(kept) / max_per_species
            idxs = [int(i * step)
                    for i in range(max_per_species)]
            kept = [kept[i] for i in idxs]

        for fname, kappa, temp in kept:
            label = fname.replace('.dat','')\
                         .replace('.txt','')
            basis_names.append(f"{parent}|{label}")
            basis_spectra.append(kappa)
            basis_parents.append(parent)
            basis_temps.append(temp)

        temps_str = [
            f"{t:.0f}K" if t else "?K"
            for _, _, t in kept]
        print(f"    {parent:<15} "
              f"{len(kept):>2} files: "
              f"{', '.join(temps_str)}")

    print(f"\n  Total basis vectors: {len(basis_names)}")
    return (basis_names, basis_spectra,
            basis_parents, basis_temps)


# ══════════════════════════════════════════════════════════
# STAGE 2b — MULTI-TEMPERATURE NNLS
# ══════════════════════════════════════════════════════════
def run_multitemp_nnls(tau_ice_grid,
                        basis_names,
                        basis_spectra,
                        basis_parents,
                        basis_temps,
                        alpha=0.15,
                        min_frac=0.005):
    """
    Tikhonov-regularised NNLS on multi-temp basis.
    Aggregates coefficients by parent species.
    Reports best-fit temperature per species.
    F4: Temperature sanity check applied after fitting.
    """
    b      = tau_ice_grid.astype(np.float64)
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-10:
        print("  ERROR: tau_ice is zero")
        return {}, 0.0

    A = np.column_stack(basis_spectra)
    n = A.shape[1]
    A_aug = np.vstack(
        [A, np.sqrt(alpha) * np.eye(n)])
    b_aug = np.concatenate([b, np.zeros(n)])
    c, _  = nnls(A_aug, b_aug)

    fit_q  = 1.0 - np.linalg.norm(
        b - A @ c) / b_norm
    total_c = max(c.sum(), 1e-10)

    # Aggregate by parent
    species_results = {}
    for i, (name, parent, temp) in enumerate(
            zip(basis_names, basis_parents,
                basis_temps)):
        coeff = float(c[i])
        if coeff < 1e-6:
            continue
        if parent not in species_results:
            species_results[parent] = {
                'total_coeff': 0.0,
                'components' : [],
                'best_temp'  : None,
                'best_coeff' : 0.0,
            }
        sr = species_results[parent]
        sr['total_coeff'] += coeff
        sr['components'].append({
            'name' : name,
            'coeff': coeff,
            'frac' : coeff / total_c,
            'temp' : temp,
        })
        if coeff > sr['best_coeff']:
            sr['best_coeff'] = coeff
            sr['best_temp']  = temp

    # Filter by minimum fraction
    species_results = {
        p: sr for p, sr in species_results.items()
        if sr['total_coeff'] / total_c >= min_frac}

    # F4: Temperature sanity check
    removed_temp = []
    for sp in list(species_results.keys()):
        bounds = TEMP_SANITY_BOUNDS.get(sp)
        if bounds is None:
            continue
        t = species_results[sp]['best_temp']
        if t is not None:
            t_lo, t_hi = bounds
            if not (t_lo <= t <= t_hi):
                removed_temp.append(
                    f"{sp} (T={t:.0f}K, "
                    f"range [{t_lo}-{t_hi}K])")
                del species_results[sp]

    if removed_temp:
        print(f"\n  Removed (temp sanity, F4):")
        for rm in removed_temp:
            print(f"    ✗ {rm}")

    # Sort by total coefficient
    species_results = dict(sorted(
        species_results.items(),
        key=lambda x: x[1]['total_coeff'],
        reverse=True))

    return species_results, fit_q


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════
def run_indra_v39(tau_sil_grid, tau_ice_grid,
                   source_name, output_dir,
                   classifiers, frac_names,
                   lida_dir,
                   ml_floor=0.08,
                   max_per_species=8,
                   alpha=0.15):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  INDRA 2.0 v3.9 — {source_name}")
    print(f"  Stage 1: ML → parents  "
          f"(tau WITH silicate)")
    print(f"  Stage 2: multi-temp NNLS  "
          f"(tau ice-only)")
    print(f"  All fixes: F1-F7 active")
    print(f"{'='*60}")

    # Stage 1
    detected_parents, species_scores, y_proba = \
        ml_detect_parents(
            tau_sil_grid, classifiers,
            frac_names, ml_floor)

    # Stage 2a
    (basis_names, basis_spectra,
     basis_parents, basis_temps) = \
        build_multitemp_basis(
            detected_parents, species_scores,
            lida_dir,
            max_per_species=max_per_species)

    if not basis_spectra:
        print("  ERROR: empty basis")
        return None, 0.0

    # Stage 2b
    print(f"\n  Stage 2b — NNLS "
          f"({len(basis_names)} basis vectors)...")
    species_results, fit_q = run_multitemp_nnls(
        tau_ice_grid,
        basis_names, basis_spectra,
        basis_parents, basis_temps,
        alpha=alpha)

    # ── Print results ─────────────────────────────────
    total_c = max(
        sum(sr['total_coeff']
            for sr in species_results.values()),
        1e-10)

    print(f"\n  {'─'*62}")
    print(f"  RESULT — {source_name} v3.9")
    print(f"  Fit quality: {fit_q*100:.1f}%")
    print(f"  Detected species: "
          f"{len(species_results)}")
    print(f"  {'─'*62}")
    print(f"\n  {'Species':<14} {'Frac%':>6}  "
          f"{'Best T':>7}  ML%    Components")
    print(f"  {'-'*14} {'-'*6}  {'-'*7}  "
          f"{'-'*5}  {'-'*32}")

    for parent, sr in species_results.items():
        frac   = sr['total_coeff'] / total_c
        ml_s   = species_scores.get(parent, 0)
        best_t = (f"{sr['best_temp']:.0f}K"
                  if sr['best_temp'] else "?K")
        comps  = sorted(
            sr['components'],
            key=lambda x: x['coeff'],
            reverse=True)[:2]
        cstr   = ', '.join(
            (f"{c['temp']:.0f}K"
             f"({c['frac']*100:.1f}%)"
             if c['temp'] else
             f"?K({c['frac']*100:.1f}%)")
            for c in comps)
        print(f"  {parent:<14} {frac*100:5.1f}%  "
              f"{best_t:>7}  {ml_s*100:4.1f}%  "
              f"{cstr}")

    # IRAS 2A recall check
    if source_name in ('IRAS_2A', 'IRAS2A',
                        'NGC1333_IRAS2A'):
        detected = set(species_results.keys())
        matched  = detected & IRAS2A_PAPER_COMPONENTS
        missed   = IRAS2A_PAPER_COMPONENTS - detected
        print(f"\n  IRAS 2A recall check:")
        print(f"  Matched: {len(matched)}/15  "
              f"({len(matched)/15*100:.0f}%)")
        print(f"  Matched: {sorted(matched)}")
        if missed:
            print(f"  Missed:  {sorted(missed)}")

    # Literature check
    print(f"\n  Boogert+2008 c2d check:")
    print(f"  {'Species':<12} {'Det':^5} "
          f"{'Range':^14}  Status")
    print(f"  {'-'*12} {'-'*5} {'-'*14}  {'-'*14}")
    for sp, (lo, hi) in C2D_LITERATURE.items():
        det = sp in species_results
        rng = f"{lo:.1f}–{hi:.1f}%"
        s   = ("✔" if det else
               "✘ expected" if lo > 0
               else "— upper limit")
        print(f"  {sp:<12} "
              f"{'YES' if det else 'no':^5} "
              f"{rng:^14}  {s}")

    # Save CSV
    rows = []
    for parent, sr in species_results.items():
        frac = sr['total_coeff'] / total_c
        for comp in sr['components']:
            rows.append({
                'parent'             : parent,
                'lida_file'          : comp['name'],
                'temperature_K'      : comp['temp'],
                'nnls_coeff'         : round(
                    comp['coeff'], 6),
                'component_frac_pct' : round(
                    comp['frac']*100, 3),
                'parent_frac_pct'    : round(
                    frac*100, 2),
                'best_temp_K'        : sr['best_temp'],
                'ml_score'           : round(
                    species_scores.get(parent, 0), 4),
            })
    csv_path = (output_dir +
                f"{source_name}_v39_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── Plots ─────────────────────────────────────────
    b         = tau_ice_grid.astype(np.float64)
    A         = np.column_stack(basis_spectra)
    n         = A.shape[1]
    c_fit, _  = nnls(
        np.vstack([A, np.sqrt(alpha)*np.eye(n)]),
        np.concatenate([b, np.zeros(n)]))
    fit_curve = A @ c_fit

    fig, axes = plt.subplots(2, 1, figsize=(14, 14))

    ax = axes[0]
    ax.plot(MIRI_GRID, b, 'k-', lw=1.2, alpha=0.8,
            label=f'{source_name} ice τ')
    ax.plot(MIRI_GRID, fit_curve, 'g-', lw=2.0,
            label=f'Multi-temp NNLS '
                  f'({fit_q*100:.0f}%  '
                  f'{len(species_results)} species)')
    residual = b - fit_curve
    ax.fill_between(MIRI_GRID, 0, residual,
                    where=residual > 0,
                    alpha=0.3, color='orange',
                    label='Under-fit')
    ax.fill_between(MIRI_GRID, 0, residual,
                    where=residual < 0,
                    alpha=0.3, color='red',
                    label='Over-fit')
    for wl_b, lbl in [(6.0,'H₂O'), (6.85,'NH₄⁺'),
                       (7.7,'CH₄'), (9.7,'Sil'),
                       (15.2,'CO₂')]:
        ax.axvline(wl_b, color='steelblue',
                   lw=0.7, ls=':', alpha=0.5)
        if b.max() > 0:
            ax.text(wl_b+0.1, b.max()*0.85, lbl,
                    fontsize=7, color='steelblue')
    ax.set_ylabel("Optical depth τ")
    ax.set_title(
        f"{source_name} v3.9 — Multi-temperature "
        f"NNLS fit",
        fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(5, 28)
    ax.grid(True, alpha=0.2)

    ax    = axes[1]
    sps   = list(species_results.keys())
    fracs = [species_results[s]['total_coeff'] /
             total_c * 100 for s in sps]
    temps = [species_results[s]['best_temp'] or 0.0
             for s in sps]
    t_arr   = np.array(temps, dtype=float)
    t_min   = t_arr.min()
    t_max_v = t_arr.max()
    t_norm  = ((t_arr - t_min) /
               (t_max_v - t_min + 1e-10))
    cmap    = plt.cm.RdYlBu_r
    colours = [cmap(t) for t in t_norm]
    ax.barh(range(len(sps)), fracs, color=colours)
    ax.set_yticks(range(len(sps)))
    ax.set_yticklabels(
        [f"{s}  "
         f"({temps[i]:.0f}K)"
         if temps[i] else s
         for i, s in enumerate(sps)],
        fontsize=9)
    ax.set_xlabel("Fractional abundance (%)")
    ax.set_title(
        f"{source_name} v3.9 — Species + "
        f"best-fit temperatures\n"
        f"[blue=cold  red=warm]",
        fontweight='bold')
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(
            vmin=t_min, vmax=t_max_v))
    sm.set_array([])
    plt.colorbar(sm, ax=ax,
                 label='Best-fit temperature (K)',
                 fraction=0.03, pad=0.02)
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plot_path = (output_dir +
                 f"{source_name}_v39_result.png")
    plt.savefig(plot_path, dpi=150,
                bbox_inches='tight')
    plt.close()

    print(f"\n  Plot → {plot_path}")
    print(f"  CSV  → {csv_path}")

    return species_results, fit_q


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("Loading INDRA 2.0 v3.9...")
    classifiers = joblib.load(CLF_FILE)
    frac_names  = joblib.load(SPE_FILE)
    thresholds  = joblib.load(THR_FILE)
    print(f"  {len(classifiers)} classifiers loaded\n")
    print(f"  Fixes active:")
    print(f"    F1: CH3NH2 excluded (hallucination)")
    print(f"    F2: NH2CHO requires 50% ML confidence")
    print(f"    F3: Per-species ML confidence floors")
    print(f"    F4: Temperature sanity bounds")
    print(f"    F5: Summary print loop fixed")
    print(f"    F6: Two-stage ML→multi-temp NNLS")
    print(f"    F7: L-BFGS-B + two-comp silicate\n")

    all_results = {}

    for cfg in SOURCES:
        os.makedirs(cfg['output_dir'], exist_ok=True)

        # Part 1: get tau files
        if cfg['mode'] == 'precomputed':
            print(f"\n  Loading pre-computed tau "
                  f"for {cfg['name']}...")
            tau_sil_grid, tau_ice_grid = \
                load_tau_files(
                    cfg['sil_tau'], cfg['ice_tau'])
        else:
            tau_sil_grid, tau_ice_grid = \
                process_flux_to_tau(
                    cfg['flux_file'], cfg)

        # Adaptive floor
        ml_floor, tau_max, tier = \
            adaptive_params(
                tau_ice_grid, cfg['name'])

        # Run v3.9 pipeline
        results, fit_q = run_indra_v39(
            tau_sil_grid, tau_ice_grid,
            cfg['name'], cfg['output_dir'],
            classifiers, frac_names,
            LIDA_DIR,
            ml_floor=ml_floor,
            max_per_species=8,
            alpha=0.15)

        if results is not None:
            all_results[cfg['name']] = {
                'results' : results,
                'fit_q'   : fit_q,
                'tau_max' : tau_max,
                'tier'    : tier,
            }

    # ── Final summary ──────────────────────────────────
    print(f"\n{'★'*65}")
    print(f"  INDRA 2.0 v3.9 — FINAL SUMMARY")
    print(f"  Two-stage: ML detect → multi-temp NNLS")
    print(f"  All fixes F1–F7 applied")
    print(f"{'★'*65}")

    for name, res in all_results.items():
        print(f"\n  {name}  [{res['tier']}]")
        print(f"    τ_max    = {res['tau_max']:.2f}")
        print(f"    Fit      = "
              f"{res['fit_q']*100:.1f}%")
        print(f"    Species  = "
              f"{len(res['results'])}")
        print(f"    Detected = "
              f"{sorted(res['results'].keys())}")
        total_c = max(
            sum(s['total_coeff']
                for s in res['results'].values()),
            1e-10)
        print(f"    Best-fit temperatures:")
        for sp, sr in res['results'].items():
            t    = sr['best_temp']
            frac = sr['total_coeff'] / total_c
            # F5: Fixed NameError — t_str defined first
            t_str = (f"{t:.0f}K"
                     if t is not None else "unknown")
            print(f"      {sp:<15} "
                  f"{frac*100:5.1f}%  "
                  f"T = {t_str}")

    # Cross-source species comparison
    if len(all_results) > 1:
        print(f"\n{'─'*65}")
        print(f"  Cross-source species matrix:")
        all_sp = sorted(set().union(
            *[set(r['results'].keys())
              for r in all_results.values()]))
        names  = list(all_results.keys())
        header = f"  {'Species':<15} " + \
                 "".join(f"{n[:12]:^14}"
                         for n in names)
        print(header)
        print(f"  {'-'*15} " +
              "".join(f"{'-'*12:^14}"
                      for _ in names))
        for sp in all_sp:
            row = f"  {sp:<15} "
            for nm in names:
                det = sp in all_results[nm]['results']
                t   = (all_results[nm]['results']
                       .get(sp, {})
                       .get('best_temp'))
                if det and t:
                    row += f"{'✔ '+str(int(t))+'K':^14}"
                elif det:
                    row += f"{'✔':^14}"
                else:
                    row += f"{'✗':^14}"
            print(row)

    print(f"\n{'★'*65}")