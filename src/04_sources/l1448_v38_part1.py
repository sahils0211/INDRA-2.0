# ============================================================
# INDRA 2.0 — Full pipeline with Prathap's continuum method
# L1448 IRS2 and IRS3B — v3.8patch baseline
# Continuum: degree-4 log-log polynomial + Nelder-Mead
# ============================================================

import numpy as np
import os
import re
import joblib
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.optimize import nnls, lsq_linear, minimize, minimize_scalar
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
BASE_DIR  = "C:/Users/sahil/Downloads/"
CLF_FILE  = BASE_DIR + "lgbm_classifiers_v38patch.pkl"
SPE_FILE  = BASE_DIR + "lgbm_species_names_v38patch.pkl"
THR_FILE  = BASE_DIR + "lgbm_thresholds_recall90_v38patch.pkl"
LIDA_DIR  = BASE_DIR + "lida_raw_data/"
MIRI_GRID = np.linspace(5.0, 28.0, 3000)

# ══════════════════════════════════════════════════════════
# SOURCE CONFIGURATIONS
# Anchor points defined from visual inspection of each
# source's flux spectrum.
#
# ANCHOR DESIGN RULES (from Prathap's INDRA paper):
#   left   = reliable continuum regions, blue of ice bands
#   frac   = fractional anchors in ambiguous regions
#            (code allows them to shift UP only)
#   right  = reliable long-wave continuum
#   extra  = additional hard constraints
#   middle = FREE points (optimizer can raise them)
#
# For Class 0 protostars in Perseus:
#   5.0–5.2 µm  → pre-H2O, reliable
#   7.2–7.4 µm  → between H2O bending and CH4, reliable
#   13.0–13.5 µm → between silicate and CO2 ice, reliable
#   19.5–20.5 µm → long-wave baseline, reliable
#   23.0–24.0 µm → long-wave baseline, reliable
#   27.0–28.0 µm → very long-wave, reliable
# ══════════════════════════════════════════════════════════

SOURCES = [
    {
        'name'       : 'L1448_IRS2',
        'label'      : 'L1448 IRS2 (Class 0, 5.2 L☉)',
        'flux_file'  : BASE_DIR + "IRS2B_C0_spectra.txt",
        'output_dir' : BASE_DIR + "L1448_IRS2_v2_results/",
        # Hard fixed anchors — definitely continuum
        'left_wav'   : np.array([5.00, 5.05, 5.10]),
        'right_wav'  : np.array([19.6, 20.0, 20.4,
                                  23.1, 23.5, 27.2,
                                  27.6, 28.0]),
        'extra_wav'  : np.array([7.20, 7.30]),
        # Fractional anchors — can shift up only
        # These bracket the silicate feature correctly
        'frac_wav'   : np.array([8.10, 8.20,
                                  12.8, 13.1, 13.4]),
        # Free middle points — between ice features
        # Optimizer can lift these up to trace continuum
        'middle_wav' : np.array([5.5, 5.7, 6.0, 6.3,
                                  6.6, 7.0, 7.5, 7.8,
                                  8.5, 9.0, 9.5, 10.0,
                                  10.5, 11.0, 11.5,
                                  12.0, 12.5, 14.0,
                                  14.5, 15.0, 16.0,
                                  17.0, 18.0, 19.0,
                                  21.0, 22.0, 24.0,
                                  25.0, 26.0]),
        # Optimizer hyperparameters (Prathap's best fit)
        'w_anchor'   : 5.3,
        'w_curve'    : 13.5,
        'poly_deg'   : 4,
    },
    {
        'name'       : 'L1448_IRS3B',
        'label'      : 'L1448 IRS3B (Class 0 binary)',
        'flux_file'  : BASE_DIR + "IRS3B_B_C0_spectra.txt",
        'output_dir' : BASE_DIR + "L1448_IRS3B_v2_results/",
        'left_wav'   : np.array([5.00, 5.05, 5.10]),
        'right_wav'  : np.array([19.6, 20.0, 20.4,
                                  23.1, 23.5, 27.2,
                                  27.6, 28.0]),
        'extra_wav'  : np.array([7.20, 7.30]),
        'frac_wav'   : np.array([8.10, 8.20,
                                  12.8, 13.1, 13.4]),
        'middle_wav' : np.array([5.5, 5.7, 6.0, 6.3,
                                  6.6, 7.0, 7.5, 7.8,
                                  8.5, 9.0, 9.5, 10.0,
                                  10.5, 11.0, 11.5,
                                  12.0, 12.5, 14.0,
                                  14.5, 15.0, 16.0,
                                  17.0, 18.0, 19.0,
                                  21.0, 22.0, 24.0,
                                  25.0, 26.0]),
        'w_anchor'   : 5.3,
        'w_curve'    : 13.5,
        'poly_deg'   : 4,
    },
]

# Literature reference
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


# ══════════════════════════════════════════════════════════
# PRATHAP'S CONTINUUM FITTER
# Direct port of the INDRA continuum optimizer
# ══════════════════════════════════════════════════════════
def fit_continuum_prathap(wavelength, flux, cfg):
    """
    Fit protostellar continuum using Prathap's log-log
    polynomial + Nelder-Mead approach from INDRA.

    Key idea: degree-4 polynomial in log(λ) vs log(F)
    - Hard penalty if continuum dips below observed flux
    - Curvature regularization for smoothness
    - Middle points can shift UP only (physical: continuum
      must be ≥ observed flux for absorption to be positive)

    Returns: continuum array (same length as wavelength)
    """
    w_anchor  = cfg['w_anchor']
    w_curve   = cfg['w_curve']
    poly_deg  = cfg['poly_deg']
    HARD_PEN  = 1e15

    # Interpolate anchor flux values from observed spectrum
    fl_interp = interp1d(wavelength, flux,
                         bounds_error=False,
                         fill_value='extrapolate')

    def get_anchor_flux(wav_arr):
        return fl_interp(wav_arr)

    left_wav  = cfg['left_wav']
    right_wav = cfg['right_wav']
    extra_wav = cfg['extra_wav']
    frac_wav  = cfg['frac_wav']
    mid_wav   = cfg['middle_wav']

    # Only keep anchor wavelengths within data range
    wl_min, wl_max = wavelength.min(), wavelength.max()
    left_wav  = left_wav[(left_wav  >= wl_min) &
                          (left_wav  <= wl_max)]
    right_wav = right_wav[(right_wav >= wl_min) &
                           (right_wav <= wl_max)]
    extra_wav = extra_wav[(extra_wav >= wl_min) &
                           (extra_wav <= wl_max)]
    frac_wav  = frac_wav[(frac_wav  >= wl_min) &
                          (frac_wav  <= wl_max)]
    mid_wav   = mid_wav[(mid_wav   >= wl_min) &
                         (mid_wav   <= wl_max)]

    left_flux  = get_anchor_flux(left_wav)
    right_flux = get_anchor_flux(right_wav)
    extra_flux = get_anchor_flux(extra_wav)
    frac_flux  = get_anchor_flux(frac_wav)
    mid_flux   = get_anchor_flux(mid_wav)

    # Remove any anchors where flux <= 0
    def filter_pos(w, f):
        ok = f > 0
        return w[ok], f[ok]

    left_wav,  left_flux  = filter_pos(left_wav,  left_flux)
    right_wav, right_flux = filter_pos(right_wav, right_flux)
    extra_wav, extra_flux = filter_pos(extra_wav, extra_flux)
    frac_wav,  frac_flux  = filter_pos(frac_wav,  frac_flux)
    mid_wav,   mid_flux   = filter_pos(mid_wav,   mid_flux)

    # Work in log space
    ok_obs        = flux > 0
    log_wav_full  = np.log(wavelength[ok_obs])
    log_flux_obs  = np.log(flux[ok_obs])

    fixed_wav     = np.concatenate([left_wav, frac_wav,
                                     right_wav, extra_wav])
    fixed_flux    = np.concatenate([left_flux, frac_flux,
                                     right_flux, extra_flux])
    log_fixed_wav  = np.log(fixed_wav)
    log_fixed_flux = np.log(fixed_flux)
    log_mid_wav    = np.log(mid_wav)
    log_mid_flux   = np.log(mid_flux)
    n_free         = len(mid_wav)

    # Design matrix: [1, log(λ), log(λ)², ..., log(λ)^d]
    def design_matrix(lw):
        return np.vstack([lw**i
                          for i in range(poly_deg+1)]).T

    A_fixed  = design_matrix(log_fixed_wav)
    A_middle = design_matrix(log_mid_wav)
    A_full   = design_matrix(log_wav_full)

    # Curvature matrix (second derivative of polynomial)
    def curvature_matrix(lw):
        cols = [np.zeros_like(lw),
                np.zeros_like(lw),
                np.full_like(lw, 2.0),
                6.0 * lw,
                12.0 * lw**2]
        return np.vstack(cols[:poly_deg+1]).T

    A_curv = curvature_matrix(log_wav_full)

    # Initial guess: least-squares fit to all anchors
    A_init      = np.vstack([A_fixed, A_middle])
    logf_init   = np.concatenate([log_fixed_flux,
                                   log_mid_flux])
    coeffs_init, _, _, _ = np.linalg.lstsq(
        A_init, logf_init, rcond=None)
    x0 = np.concatenate([coeffs_init,
                          np.zeros(n_free)])

    # Objective function (direct port from Prathap)
    def objective(x):
        coeffs     = x[:poly_deg + 1]
        raw_deltas = x[poly_deg + 1:]
        # Deltas ≥ 0: middle points can only move UP
        deltas     = np.clip(raw_deltas, 0, None)

        log_cont   = A_full @ coeffs

        # Hard penalty: continuum must be ≥ observed flux
        below_obs  = log_flux_obs - log_cont
        if np.any(below_obs > 0):
            return (HARD_PEN +
                    np.sum(below_obs[below_obs > 0]**2)
                    * 1e10)

        # Fixed anchor residuals
        fixed_res  = np.sum(
            (A_fixed @ coeffs - log_fixed_flux)**2)

        # Middle anchor residuals (with upward shifts)
        log_adj    = log_mid_flux + np.log1p(deltas)
        middle_res = np.sum(
            (A_middle @ coeffs - log_adj)**2)

        anchor_term = w_anchor * (fixed_res + middle_res)

        # Curvature penalty
        curv_term   = w_curve * np.mean(
            (A_curv @ coeffs)**2)

        return anchor_term + curv_term

    # Nelder-Mead optimization
    print(f"    Running Nelder-Mead optimizer "
          f"(n_params={len(x0)})...")
    result = minimize(
        objective, x0,
        method='Nelder-Mead',
        options={
            'maxiter' : 200000,
            'xatol'   : 1e-7,
            'fatol'   : 1e-7,
            'adaptive': True,  # better for high-dim
        })
    print(f"    Optimizer: "
          f"{'converged' if result.success else 'max iter'}"
          f"  fun={result.fun:.4f}")

    coeffs_final = result.x[:poly_deg + 1]

    # Evaluate continuum on full wavelength grid
    # (including points with flux <= 0)
    log_wav_all  = np.log(np.maximum(wavelength, 1e-10))
    A_all        = design_matrix(log_wav_all)
    continuum    = np.exp(A_all @ coeffs_final)

    # Safety: continuum must be ≥ observed flux everywhere
    continuum    = np.maximum(continuum, flux)

    return continuum


# ══════════════════════════════════════════════════════════
# PART 1 — FLUX → OPTICAL DEPTH (Prathap's method)
# ══════════════════════════════════════════════════════════
def process_flux_to_tau(flux_file, cfg):
    source_name = cfg['name']
    output_dir  = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Part 1: {source_name} — Flux → τ")
    print(f"  Continuum: Prathap's log-log polynomial")
    print(f"{'='*60}")

    # Load flux
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

    # Mask emission lines and spikes
    emission_lines = [
        5.051, 5.340, 5.511, 6.109, 6.634,
        6.910, 8.025, 9.665, 10.511,
        12.279, 12.814, 17.035, 17.936,
        25.249, 25.988,
    ]
    mask_good = np.ones(len(wl_raw), dtype=bool)
    for line in emission_lines:
        mask_good &= ~((wl_raw > line - 0.06) &
                       (wl_raw < line + 0.06))
    fl_med     = medfilt(fl_raw, kernel_size=21)
    spike_mask = fl_raw > 5.0 * fl_med
    mask_good &= ~spike_mask
    n_masked   = (~mask_good).sum()
    print(f"  Masked {n_masked} emission/spike pixels")

    # Replace masked pixels with local median
    # (optimizer needs continuous spectrum)
    fl_clean = fl_raw.copy()
    fl_clean[~mask_good] = fl_med[~mask_good]
    fl_clean = np.maximum(fl_clean, 1e-10)

    # Fit continuum using Prathap's method
    print(f"\n  Fitting continuum...")
    continuum = fit_continuum_prathap(
        wl_raw, fl_clean, cfg)

    # τ = ln(F_cont / F_obs) — Prathap's convention
    # = positive where absorption exists
    ratio     = continuum / np.maximum(fl_clean, 1e-15)
    ratio     = np.maximum(ratio, 1.0)  # cont ≥ obs
    tau_raw   = np.log(ratio)
    tau_raw   = np.maximum(tau_raw, 0.0)

    # Diagnostic optical depths
    t6  = float(np.interp(6.0,  wl_raw, tau_raw))
    t97 = float(np.interp(9.7,  wl_raw, tau_raw))
    t15 = float(np.interp(15.2, wl_raw, tau_raw))
    t20 = float(np.interp(20.0, wl_raw, tau_raw))
    t25 = float(np.interp(25.0, wl_raw, tau_raw))
    print(f"\n  τ(6.0  µm) H₂O   = {t6:.3f}")
    print(f"  τ(7.7  µm) CH₄   = "
          f"{float(np.interp(7.7, wl_raw, tau_raw)):.3f}")
    print(f"  τ(9.7  µm) Sil   = {t97:.3f}")
    print(f"  τ(15.2 µm) CO₂   = {t15:.3f}")
    print(f"  τ(20.0 µm)       = {t20:.3f}  "
          f"(should be ~0)")
    print(f"  τ(25.0 µm)       = {t25:.3f}  "
          f"(should be ~0)")

    # Physical checks
    if t6 < 0.3:
        print(f"  ⚠ τ(6µm)={t6:.3f} low — "
              f"check left anchors at 5.0 µm")
    if t97 < 0.5:
        print(f"  ⚠ τ(9.7µm)={t97:.3f} low — "
              f"frac anchors at 8.1 µm may be too high")
    if t20 > 0.2:
        print(f"  ⚠ τ(20µm)={t20:.3f} still elevated — "
              f"right anchors need adjustment")

    # Silicate subtraction
    print(f"\n  Subtracting silicate...")

    def drude(wl, tp, wl0=9.7, g=0.97):
        return (tp * (wl/wl0)**2 * g**2 /
                ((wl**2 - wl0**2)**2 +
                 (wl*g)**2) * wl0**2)

    def drude18(wl, tp):
        return drude(wl, tp*0.4, wl0=18.0, g=0.45)

    # Fit silicate from wings — exclude core to avoid
    # confusing ice features with silicate peak
    sil_win = (((wl_raw >= 8.0) & (wl_raw <= 8.5)) |
               ((wl_raw >= 11.8) & (wl_raw <= 12.5)))
    sil_wl  = wl_raw[sil_win]
    sil_tau = tau_raw[sil_win]

    res = minimize_scalar(
        lambda tp: np.sum(
            (sil_tau - drude(sil_wl, tp) -
             drude18(sil_wl, tp))**2),
        bounds=(0.1, 8.0), method='bounded')
    tau_peak = res.x
    tau_sil  = (drude(wl_raw, tau_peak) +
                drude18(wl_raw, tau_peak))
    tau_ice  = np.maximum(tau_raw - tau_sil, 0.0)

    print(f"  Silicate τ_peak = {tau_peak:.3f}")
    print(f"  τ_ice max       = {tau_ice.max():.3f}")

    # Grid interpolation to MIRI grid
    def to_grid(wl, tau):
        valid = ((wl >= 5.0) & (wl <= 28.0) &
                 np.isfinite(tau) & (tau >= 0))
        fi  = interp1d(wl[valid], tau[valid],
                       bounds_error=False,
                       fill_value=0.0)
        return np.maximum(
            medfilt(np.maximum(fi(MIRI_GRID), 0.0), 5),
            0.0)

    tau_sil_grid = to_grid(wl_raw, tau_raw)
    tau_ice_grid = to_grid(wl_raw, tau_ice)

    # Save
    np.savetxt(
        output_dir + f"{source_name}_opticaldepth_ice.dat",
        np.column_stack([MIRI_GRID, tau_ice_grid]),
        fmt='%.6f  %.6f',
        header='wavelength(um)  tau_ice')
    np.savetxt(
        output_dir + f"{source_name}_opticaldepth_withsil.dat",
        np.column_stack([MIRI_GRID, tau_sil_grid]),
        fmt='%.6f  %.6f',
        header='wavelength(um)  tau_with_sil')

    # Diagnostic plot (3 panels)
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    ax = axes[0]
    ax.semilogy(wl_raw, fl_clean, 'k-', lw=0.7,
                alpha=0.6, label='Observed flux')
    ax.semilogy(wl_raw, continuum, 'g--', lw=2.0,
                label='Continuum (Prathap method)')
    # Mark anchors
    fl_i = interp1d(wl_raw, fl_clean,
                    bounds_error=False,
                    fill_value='extrapolate')
    for wl_a, col, lab in [
            (cfg['left_wav'],  'blue',   'Left fixed'),
            (cfg['right_wav'], 'red',    'Right fixed'),
            (cfg['frac_wav'],  'orange', 'Frac anchors'),
            (cfg['extra_wav'], 'purple', 'Extra fixed'),
    ]:
        wl_a_ok = wl_a[(wl_a >= wl_raw.min()) &
                        (wl_a <= wl_raw.max())]
        if len(wl_a_ok) > 0:
            ax.semilogy(wl_a_ok,
                        fl_i(wl_a_ok), 'o',
                        color=col, ms=5,
                        label=lab, zorder=5)
    ax.set_ylabel("Flux (Jy)")
    ax.set_title(f"{source_name} — Prathap continuum fit "
                 f"(degree={cfg['poly_deg']}, "
                 f"w_a={cfg['w_anchor']}, "
                 f"w_c={cfg['w_curve']})",
                 fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(5, 28)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(wl_raw, tau_raw, 'k-', lw=0.8,
            alpha=0.7, label='Total τ')
    ax.plot(wl_raw, tau_sil, 'r--', lw=1.5,
            label=f'Silicate (τ_peak={tau_peak:.2f})')
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
    ax.set_title("Total τ and silicate subtraction\n"
                 "τ (with sil) → ML   |   "
                 "τ (ice only) → NNLS",
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(5, 28)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.plot(MIRI_GRID, tau_ice_grid, 'g-', lw=1.5,
            label=f'{source_name} ice τ')
    ax.plot(MIRI_GRID, tau_sil_grid, 'b-', lw=0.8,
            alpha=0.3, label='τ + silicate (ML input)')
    try:
        d2a = np.loadtxt(
            BASE_DIR + "iras2A_global_opticaldepth.dat",
            comments='#')
        wl2a, t2a = d2a[:, 0], d2a[:, 1]
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
    plt.savefig(
        output_dir + f"{source_name}_tau_diag.png",
        dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {output_dir}{source_name}_tau_diag.png")

    tau_vals = {
        't6': t6, 't97': t97, 't15': t15,
        't20': t20, 't25': t25,
        'tau_peak': tau_peak,
        'tau_max': tau_ice_grid.max()
    }
    return tau_sil_grid, tau_ice_grid, tau_vals


# ══════════════════════════════════════════════════════════
# SHARED INDRA 2.0 FUNCTIONS (unchanged from v3.8patch)
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
        d = np.array(rows)
        wl, kap = d[:, 0], d[:, 1]
        if wl.mean() > 100:
            wl  = 1e4/wl; wl=wl[::-1]; kap=kap[::-1]
        valid = ((wl >= 5.0) & (wl <= 28.0) &
                 np.isfinite(kap) & (kap >= 0))
        if valid.sum() < 30:
            return None
        fi = interp1d(wl[valid], kap[valid],
                      bounds_error=False, fill_value=0.0)
        kg = fi(MIRI_GRID)
        km = kg.max()
        return kg/km if km > 0 else None
    except Exception:
        return None


def fuzzy_load_lida(label, lida_dir):
    base     = re.sub(r'_class[A-E]$', '', label)
    variants = [
        base,
        re.sub(r'_\d+[Kk]$', '', base),
        base.replace('-', '_'),
        base.replace('_', '-'),
        base.lower(),
    ]
    for sfx in ['_NASA', '_shapesmall', '_thin',
                '_large_mie', '_large']:
        variants.append(
            re.sub(sfx, '', base, flags=re.IGNORECASE))
    variants.append(re.sub(r'__+', '_', base))
    for v in variants:
        for ext in ['.dat', '.txt', '.dat1', '.csv']:
            p = os.path.join(lida_dir, v + ext)
            if os.path.exists(p):
                k = _load_kappa(p)
                if k is not None:
                    return k, v+ext
    try:
        all_files = os.listdir(lida_dir)
    except Exception:
        return None, None
    bu = base.upper()
    for fname in all_files:
        fu   = fname.upper()
        core = re.sub(
            r'(_\d+[Kk])?(_class[A-E])?'
            r'(\.(DAT|TXT|DAT1|CSV))$', '', fu)
        bc   = re.sub(
            r'(_\d+[Kk])?(_class[A-E])?$', '', bu)
        if core == bc:
            k = _load_kappa(os.path.join(lida_dir, fname))
            if k is not None:
                return k, fname
    return None, None


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


def tikhonov_nnls(A, b, alpha=0.15):
    n = A.shape[1]
    c, _ = nnls(
        np.vstack([A, np.sqrt(alpha)*np.eye(n)]),
        np.concatenate([b, np.zeros(n)]))
    return c


def refit_floors(A, b, keep_idx, names,
                  forced, alpha=0.15):
    Ak = A[:, keep_idx]
    lb = np.array([0.002 if names[i] in forced
                   else 0.0 for i in keep_idx])
    ub = np.full(len(keep_idx), np.inf)
    if (lb > 0).any():
        Aa  = np.vstack(
            [Ak, np.sqrt(alpha)*np.eye(len(keep_idx))])
        ba  = np.concatenate(
            [b, np.zeros(len(keep_idx))])
        res = lsq_linear(Aa, ba, bounds=(lb, ub),
                          method='bvls', max_iter=3000)
        ck  = np.maximum(res.x, 0.0)
    else:
        ck, _ = nnls(Ak, b)
    c_full = np.zeros(len(names))
    for k, idx in enumerate(keep_idx):
        c_full[idx] = ck[k]
    return c_full


def omp_nnls(A, b, ml_scores,
              max_components=15, min_corr=0.05):
    n     = A.shape[1]
    bn    = np.linalg.norm(b)
    if bn < 1e-10:
        return np.zeros(n), [], 0.0, []
    res   = b.copy()
    supp  = []
    hist  = []
    cn    = np.linalg.norm(A, axis=0)
    cn    = np.where(cn < 1e-10, 1.0, cn)
    An    = A / cn[np.newaxis, :]
    ml    = np.array(ml_scores, dtype=np.float64)
    for step in range(max_components):
        corr = An.T @ res
        corr = np.where(corr > 0, corr, 0.0)
        for idx in supp:
            corr[idx] = 0.0
        best = corr.max()
        if best < min_corr:
            print(f"  OMP stop step {step+1}: "
                  f"corr {best:.4f} < {min_corr}")
            break
        near = np.where(corr >= best - 0.05*best)[0]
        bidx = (near[np.argmax(ml[near])]
                if len(near) > 1 else near[0])
        supp.append(int(bidx))
        cs, _ = nnls(A[:, supp], b)
        res   = b - A[:, supp] @ cs
        hist.append(1.0 - np.linalg.norm(res)/bn)
    coeffs = np.zeros(n)
    if supp:
        cs, _ = nnls(A[:, supp], b)
        for k, idx in enumerate(supp):
            coeffs[idx] = cs[k]
    q = 1.0 - np.linalg.norm(b - A @ coeffs) / bn
    return coeffs, supp, q, hist


def backward_elim(A, b, c_init, names,
                   forced, alpha=0.15):
    NNLS_MIN = 1e-4
    cur_c    = c_init.copy()
    cur_set  = [names[i] for i in range(len(names))
                if cur_c[i] > NNLS_MIN]
    bn       = np.linalg.norm(b)
    if bn < 1e-10:
        return cur_c, cur_set, 0.0
    print(f"  Backward elim: {len(cur_set)} start")
    while True:
        tot  = max(sum(cur_c[names.index(n)]
                       for n in cur_set
                       if n in names), 1e-10)
        rem  = sorted(
            [(n, cur_c[names.index(n)]/tot)
             for n in cur_set if n not in forced],
            key=lambda x: x[1])
        done = True
        for nm, frac in rem:
            tset = [n for n in cur_set if n != nm]
            kidx = [i for i, n in enumerate(names)
                    if n in set(tset)]
            nc   = refit_floors(A, b, kidx, names,
                                 forced, alpha)
            cf   = 1.0 - np.linalg.norm(
                b - A @ cur_c) / bn
            nf   = 1.0 - np.linalg.norm(
                b - A @ nc) / bn
            if nf >= cf - 0.005:
                cur_c   = nc
                cur_set = tset
                print(f"    REMOVE {nm[:45]}  "
                      f"{frac*100:.1f}%  "
                      f"→ {nf*100:.1f}%")
                done = False
                break
        if done:
            break
    qf = 1.0 - np.linalg.norm(b - A @ cur_c) / bn
    print(f"  Final: {len(cur_set)} survivors  "
          f"fit={qf*100:.1f}%")
    return cur_c, cur_set, qf


# ══════════════════════════════════════════════════════════
# ADAPTIVE FLOOR
# ══════════════════════════════════════════════════════════
def adaptive_params(tau_ice_grid, source_name):
    tau_max = float(tau_ice_grid.max())
    if tau_max > 4.0:
        floor, omp_max, tier = 0.05, 25, "rich"
    elif tau_max > 1.5:
        floor, omp_max, tier = 0.08, 15, "moderate"
    elif tau_max > 0.5:
        floor, omp_max, tier = 0.10, 12, "simple"
    else:
        floor, omp_max, tier = 0.15, 8,  "faint"
    print(f"\n  Source tier: {tier} "
          f"(τ_max = {tau_max:.2f})")
    print(f"  → ML floor: {floor*100:.0f}%  "
          f"OMP max: {omp_max}")
    return floor, omp_max, tau_max, tier


# ══════════════════════════════════════════════════════════
# PART 2 — INDRA 2.0 PIPELINE (unchanged v3.8patch)
# ══════════════════════════════════════════════════════════
def run_indra(tau_sil_grid, tau_ice_grid,
              source_name, output_dir,
              classifiers, frac_names,
              ml_floor, omp_max, tau_max):
    print(f"\n{'='*60}")
    print(f"  Part 2: INDRA 2.0 v3.8patch — {source_name}")
    print(f"  Continuum: Prathap's method")
    print(f"  ML floor: {ml_floor*100:.0f}%  "
          f"OMP max: {omp_max}")
    print(f"{'='*60}")

    ALPHA     = 0.15
    DEDUP_THR = 0.015
    PROT_THR  = 0.08
    NNLS_MIN  = 1e-4
    NH4_KEYS  = ['NHplus4', 'NH4plus', 'NH+4']

    # ML scoring — tau WITH silicate
    print(f"\n  [1/5] ML scoring...")
    X       = tau_sil_grid.reshape(1,-1).astype(np.float32)
    y_proba = np.array([
        clf.predict_proba(X)[0, 1]
        for clf in classifiers])
    sorted_idx = np.argsort(y_proba)[::-1]

    print(f"  Top 15:")
    for i in sorted_idx[:15]:
        rank = int(np.where(sorted_idx==i)[0][0]) + 1
        print(f"  {rank:>4}  "
              f"{y_proba[i]*100:5.1f}%  "
              f"{frac_names[i]}")

    cands = [i for i in sorted_idx
             if y_proba[i] >= ml_floor]
    print(f"\n  Candidates ≥{ml_floor*100:.0f}%: "
          f"{len(cands)}")

    # Load LIDA
    print(f"\n  [2/5] Loading LIDA spectra...")
    nml, spl, scl = [], [], []
    for i in cands:
        k, _ = fuzzy_load_lida(frac_names[i], LIDA_DIR)
        if k is not None:
            nml.append(frac_names[i])
            spl.append(k)
            scl.append(float(y_proba[i]))

    # Dedup
    if len(spl) >= 2:
        sm  = np.array(spl)
        cd  = cosine_distances(sm)
        ord_ = np.argsort(scl)[::-1]
        kept = []
        for idx in ord_:
            if not any(cd[idx,k] < DEDUP_THR
                       for k in kept):
                kept.append(idx)
        nn = [nml[i] for i in kept]
        ss = [spl[i] for i in kept]
        sc = [scl[i] for i in kept]
    else:
        nn, ss, sc = nml, spl, scl

    # Re-add NH4+ if needed
    for i in cands:
        nm = frac_names[i]
        if (any(kw.upper() in nm.upper()
                for kw in NH4_KEYS)
                and nm not in nn):
            k, _ = fuzzy_load_lida(nm, LIDA_DIR)
            if k is not None:
                nn.append(nm); ss.append(k)
                sc.append(float(y_proba[i]))
                print(f"  Re-added NH4+: {nm}")

    print(f"  Pool after dedup: {len(nn)}")

    # OMP on ice-only tau
    print(f"\n  [3/5] OMP...")
    if len(ss) == 0:
        print("  ERROR: no LIDA spectra loaded")
        return None

    A = np.column_stack(ss)
    b = tau_ice_grid.astype(np.float64)

    if b.max() < 0.01:
        print("  ⚠ τ_ice_max too low — check tau conversion")
        return None

    c_tikh = tikhonov_nnls(A, b, ALPHA)
    q_tikh = 1.0 - np.linalg.norm(
        b - A @ c_tikh) / np.linalg.norm(b)

    c_omp, supp, q_omp, hist = omp_nnls(
        A, b, sc,
        max_components=omp_max, min_corr=0.05)

    omp_names = {nn[i] for i in supp}
    print(f"\n  OMP: {len(supp)} components  "
          f"fit={q_omp*100:.1f}%")
    for k, idx in enumerate(supp):
        print(f"    {k+1:>2}  {nn[idx][:52]:<52}  "
              f"ML={sc[idx]*100:.1f}%  "
              f"fit={hist[k]*100:.1f}%")

    # Force + protect
    print(f"\n  [4/5] Force + protect...")
    ml_arr    = np.array(sc)
    force_idx = set()
    for i in range(len(nn)):
        if c_omp[i] <= NNLS_MIN and ml_arr[i] > 0.70:
            force_idx.add(i)
            print(f"  Force: {nn[i][:50]}  "
                  f"{ml_arr[i]*100:.1f}%")

    if force_idx:
        lb = np.zeros(len(nn))
        for i in force_idx:
            lb[i] = 0.002
        Aa  = np.vstack(
            [A, np.sqrt(ALPHA)*np.eye(A.shape[1])])
        ba  = np.concatenate(
            [b, np.zeros(A.shape[1])])
        res = lsq_linear(Aa, ba,
                          bounds=(lb, np.inf),
                          method='bvls', max_iter=3000)
        c_ff = np.maximum(res.x, 0.0)
    else:
        c_ff = c_omp.copy()

    forced_names = {nn[i] for i in force_idx}

    c_tot = max(c_ff.sum(), 1e-10)
    prot  = set()
    print(f"  Protecting >{PROT_THR*100:.0f}%:")
    for i in range(len(nn)):
        if c_ff[i] > NNLS_MIN:
            frac = c_ff[i] / c_tot
            if frac > PROT_THR:
                prot.add(nn[i])
                print(f"    {nn[i][:52]}  {frac*100:.1f}%")
    forced_names |= prot

    c_final, final_names, q_final = backward_elim(
        A, b, c_ff, nn, forced_names, ALPHA)
    fit_final = A @ c_final

    # Results
    res_names  = [nn[i] for i in range(len(nn))
                  if c_final[i] > NNLS_MIN]
    res_coeffs = [float(c_final[i])
                  for i in range(len(nn))
                  if c_final[i] > NNLS_MIN]
    ordr       = np.argsort(res_coeffs)[::-1]
    res_names  = [res_names[i]  for i in ordr]
    res_coeffs = [res_coeffs[i] for i in ordr]
    total_c    = sum(res_coeffs) or 1.0

    def via(n):
        if n in omp_names:      return "OMP"
        elif n in forced_names: return "forced"
        return "floor"

    all_parents = set().union(
        *[get_parents(n) for n in res_names])

    print(f"\n  {'─'*62}")
    print(f"  RESULT — {source_name}  "
          f"[Prathap continuum]")
    print(f"  {'─'*62}")
    print(f"  Fit: OMP={q_omp*100:.1f}%  "
          f"Tikhonov={q_tikh*100:.1f}%  "
          f"Final={q_final*100:.1f}%")
    print(f"  Survivors: {len(res_names)}")
    print(f"\n  {'Component':<50} {'Frac%':>6}  Via")
    for nm, co in zip(res_names, res_coeffs):
        frac = co / total_c
        print(f"  {nm[:50]:<50} "
              f"{frac*100:5.1f}%  {via(nm)}")
    print(f"\n  Detected: {sorted(all_parents)}")

    # Literature check
    print(f"\n  Literature (Boogert+2008 c2d):")
    print(f"  {'Species':<12} {'Det':^6} "
          f"{'c2d range':^14} {'Status'}")
    for sp, (lo, hi) in C2D_LITERATURE.items():
        det    = sp in all_parents
        rng    = f"{lo:.1f}–{hi:.1f}%"
        status = ("✔" if det else
                  "✘ expected" if lo > 0 else "—")
        print(f"  {sp:<12} {'YES' if det else 'no':^6} "
              f"{rng:^14} {status}")

    # Save CSV
    final_map = dict(zip(res_names, res_coeffs))
    rows      = []
    for i in cands:
        nm    = frac_names[i]
        coeff = final_map.get(nm, 0.0)
        rows.append({
            'label'        : nm,
            'parents'      : ', '.join(
                sorted(get_parents(nm))),
            'ml_conf'      : round(float(y_proba[i]),4),
            'nnls_coeff'   : round(coeff, 6),
            'nnls_frac_pct': round(coeff/total_c*100, 3),
            'detected'     : coeff > NNLS_MIN,
            'via'          : (via(nm)
                              if coeff > NNLS_MIN else ''),
        })
    csv_path = (output_dir +
                f"{source_name}_results_prathap_cont.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 20))

    ax = axes[0]
    ax.plot(MIRI_GRID, b, 'k-', lw=1.2, alpha=0.8,
            label=f'{source_name} ice τ')
    ax.plot(MIRI_GRID, A @ c_tikh, 'b--', lw=1.0,
            alpha=0.5,
            label=f'Tikhonov ({q_tikh*100:.0f}%)')
    ax.plot(MIRI_GRID, fit_final, 'g-', lw=2.0,
            label=f'Final ({q_final*100:.0f}%  '
                  f'{len(res_names)} comp.)')
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
        f"{source_name} — NNLS fit "
        f"[Prathap continuum method]",
        fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(5, 28)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    residual = b - fit_final
    ax.plot(MIRI_GRID, b, 'k-', lw=0.8,
            alpha=0.5, label='Ice τ')
    ax.plot(MIRI_GRID, fit_final, 'g-', lw=1.5,
            label='Final fit')
    ax.fill_between(MIRI_GRID, 0, residual,
                    where=residual > 0,
                    alpha=0.4, color='orange',
                    label='Under-fit')
    ax.fill_between(MIRI_GRID, 0, residual,
                    where=residual < 0,
                    alpha=0.4, color='red',
                    label='Over-fit')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_ylabel("Residual τ")
    ax.set_title("Residuals", fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(5, 28)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    if res_names:
        fracs = [c/total_c*100 for c in res_coeffs]
        cmap  = {'OMP':    '#2ecc71',
                 'forced': '#e74c3c',
                 'floor':  '#95a5a6'}
        cols  = [cmap.get(via(n), '#95a5a6')
                 for n in res_names]
        ax.barh(range(len(res_names)), fracs, color=cols)
        ax.set_yticks(range(len(res_names)))
        ax.set_yticklabels(
            [n[:55] for n in res_names], fontsize=7)
        ax.set_xlabel("Fractional abundance (%)")
        ax.set_title(
            f"{source_name} — Composition  "
            f"fit={q_final*100:.0f}%  "
            f"τ_max={tau_max:.2f}  "
            f"ML_floor={ml_floor*100:.0f}%\n"
            f"[green=OMP  red=forced  gray=floor]",
            fontweight='bold')
        ax.legend(handles=[
            mpatches.Patch(
                facecolor='#2ecc71', label='OMP'),
            mpatches.Patch(
                facecolor='#e74c3c', label='Forced'),
            mpatches.Patch(
                facecolor='#95a5a6', label='Floor'),
        ], fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(
        output_dir +
        f"{source_name}_indra_result_prathap_cont.png",
        dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {output_dir}"
          f"{source_name}_indra_result_prathap_cont.png")
    print(f"  CSV  → {csv_path}")

    return {
        'name'      : source_name,
        'parents'   : all_parents,
        'components': res_names,
        'q_final'   : q_final,
        'q_omp'     : q_omp,
        'tau_max'   : tau_max,
    }


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
print("Loading INDRA 2.0 v3.8patch...")
classifiers = joblib.load(CLF_FILE)
frac_names  = joblib.load(SPE_FILE)
thresholds  = joblib.load(THR_FILE)
print(f"  {len(classifiers)} classifiers")
print(f"\n  Continuum method: Prathap's log-log "
      f"polynomial (degree=4, Nelder-Mead)")
print(f"  Expected improvements:")
print(f"    - Silicate τ_peak should be > 0.5")
print(f"    - τ(7.7µm) CH4 should be > 0")
print(f"    - τ(20-28µm) should be ~0\n")

all_results = {}

for cfg in SOURCES:
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # Part 1: Prathap's continuum method
    tau_sil_grid, tau_ice_grid, tau_vals = \
        process_flux_to_tau(cfg['flux_file'], cfg)

    # Adaptive parameters
    ml_floor, omp_max, tau_max, tier = \
        adaptive_params(tau_ice_grid, cfg['name'])

    # Part 2: INDRA pipeline (unchanged)
    result = run_indra(
        tau_sil_grid, tau_ice_grid,
        cfg['name'], cfg['output_dir'],
        classifiers, frac_names,
        ml_floor, omp_max, tau_max)

    if result is not None:
        result.update(tau_vals)
        result['tier'] = tier
        all_results[cfg['name']] = result


# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print(f"\n{'★'*65}")
print(f"  SUMMARY — Prathap continuum vs simple spline")
print(f"{'★'*65}")
for sname, res in all_results.items():
    print(f"\n  {sname}")
    print(f"    τ_max={res['tau_max']:.2f}  "
          f"τ(6µm)={res['t6']:.3f}  "
          f"τ(9.7µm)={res['t97']:.3f}  "
          f"τ(15µm)={res['t15']:.3f}")
    print(f"    τ(20µm)={res['t20']:.3f}  "
          f"τ(25µm)={res['t25']:.3f}  "
          f"Sil_peak={res['tau_peak']:.3f}")
    print(f"    Fit={res['q_final']*100:.1f}%  "
          f"Survivors={len(res['components'])}")
    print(f"    Detected: {sorted(res['parents'])}")
print(f"\n  Key checks vs previous run:")
print(f"    Silicate τ_peak: was 0.05, "
      f"should now be > 0.5")
print(f"    τ(7.7µm) CH4:   was 0.000, "
      f"should now be > 0.1")
print(f"    τ(25µm) artifact: was 0.66/0.27, "
      f"should now be ~0")
print(f"    Fit quality: was 30/44%, "
      f"expect 60-85% improvement")
print(f"{'★'*65}")