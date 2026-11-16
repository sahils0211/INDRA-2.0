# ============================================================
# INDRA 2.0 — Gaussian Peak Visualizer
# Interactive viewer for inspecting Gaussian decompositions
# of any LIDA lab file.
#
# Usage:
#   python gaussian_visualizer.py
#   python gaussian_visualizer.py H2O160K
#   python gaussian_visualizer.py --list-species CH3OH
#   python gaussian_visualizer.py --from-db H2O160Kgshapesmall_classA.dat
#
# What it shows:
#   Panel 1: Raw file data (wavenumber, absorbance)
#   Panel 2: Baseline-subtracted + each Gaussian component
#   Panel 3: Fit residuals
#   Panel 4: Reconstructed MIRI-grid spectrum
#   Panel 5: MIRI-range comparison (Gaussian vs raw)
#
# You can also load directly from the JSON database
# produced by gaussian_decompose_batch.py if it exists.
# ============================================================

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox
import json
import os
import re
import sys
import glob
import argparse
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import medfilt

# Try importing the decomposition functions
# from the batch script (same directory)
try:
    from gaussian_decompose_batch import (
        load_lida_file, subtract_baseline,
        find_peaks_auto, fit_gaussian_peaks,
        asymmetric_gaussian,
        MIRI_WN_MIN, MIRI_WN_MAX, MIRI_WN_GRID,
        MIRI_N_PTS, get_species, extract_temperature,
        PARENT_RULES
    )
    BATCH_IMPORTED = True
except ImportError:
    BATCH_IMPORTED = False
    print("  Note: gaussian_decompose_batch.py not found "
          "in same directory. Running standalone mode.")

# ══════════════════════════════════════════════════════════
# PATHS — edit here
# ══════════════════════════════════════════════════════════
LIDA_DIR = "C:/Users/sahil/Downloads/lida_raw_data/"
DB_PATH  = ("C:/Users/sahil/Downloads/gaussian_db/"
             "gaussian_peaks_database.json")

MIRI_WN_MIN  = 357.0
MIRI_WN_MAX  = 2000.0
MIRI_N_PTS   = 3000
MIRI_WN_GRID = np.linspace(
    MIRI_WN_MIN, MIRI_WN_MAX, MIRI_N_PTS)

# ══════════════════════════════════════════════════════════
# STANDALONE VERSIONS (if batch script not imported)
# ══════════════════════════════════════════════════════════
if not BATCH_IMPORTED:
    import scipy.signal as signal
    from pybaselines import Baseline
    import lmfit
    import warnings
    warnings.filterwarnings('ignore')

    PEAK_PROMINENCE_FRAC = 0.04
    PEAK_MIN_WIDTH_WN    = 5.0
    PEAK_MAX_WIDTH_WN    = 500.0
    MAX_PEAKS_PER_FILE   = 20
    SIGMA_MIN            = 3.0
    SIGMA_MAX            = 400.0
    AMPLITUDE_MIN        = 1e-5

    PARENT_RULES = [
        ('CH3CH2OH','CH3CH2OH'),('C2H5OH','CH3CH2OH'),
        ('CH3COOH','CH3COOH'), ('CH3OCHO','CH3OCHO'),
        ('CH3CHO','CH3CHO'),   ('CH3OH','CH3OH'),
        ('CH4','CH4'),         ('H2CO','H2CO'),
        ('HCOOH','HCOOH'),     ('HCOO','HCOO-'),
        ('NH2CHO','NH2CHO'),   ('HNCO','HNCO'),
        ('NH4','NH4+'),        ('NHPLUS','NH4+'),
        ('NH3','NH3'),         ('H2S','H2S'),
        ('SO2','SO2'),         ('CO2','CO2'),
        ('C18O','CO2'),        ('H2O','H2O'),
        ('CO','CO'),           ('SILICATE','Silicate'),
        ('OLIVINE','Silicate'),('SPEC_D','organic'),
        ('SPEC_A','organic'),  ('ORGANIC','organic'),
        ('RESIDUE','organic'), ('C3H6O','CH3CHO'),
        ('OCS','OCS'),         ('OCN','OCN-'),
        ('HNCO_NH3','HNCO'),   ('HNCO_NH3','NH3'),
    ]

    def get_species(label):
        lu = label.upper()
        return sorted({p for k, p in PARENT_RULES
                        if k.upper() in lu})

    def extract_temperature(fname):
        matches = list(re.finditer(
            r'(?<![0-9])(\d+(?:\.\d+)?)[Kk]'
            r'(?![a-zA-Z])', fname))
        if matches:
            try:
                return float(matches[-1].group(1))
            except ValueError:
                return None
        return None

    def load_lida_file(path):
        try:
            with open(path, 'rb') as f:
                sig = f.read(4)
            if sig[:2] in [b'\xff\xfe', b'\xfe\xff']:
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
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    if np.isfinite(x) and np.isfinite(y):
                        rows.append((x, y))
                except ValueError:
                    continue
            if len(rows) < 30:
                return None
            data = np.array(rows)
            x, y = data[:, 0], data[:, 1]
            order = np.argsort(x)
            x, y  = x[order], y[order]
            if x.mean() < 100:
                valid_x = x > 0
                wn = np.zeros_like(x)
                wn[valid_x] = 1e4 / x[valid_x]
                wn = wn[valid_x]
                y  = y[valid_x]
                wn = wn[::-1]
                y  = y[::-1]
                x  = wn
            y_max = np.percentile(y, 99)
            y_min = np.percentile(y, 1)
            if y_min >= -0.05 and y_max <= 1.05:
                y = -np.log10(np.clip(y, 1e-6, 1.0))
            elif y_min < -0.3:
                y = -y
            y = np.maximum(y, 0.0)
            valid = ((x >= 200) & (x <= 5000) &
                     np.isfinite(y))
            if valid.sum() < 30:
                return None
            return x[valid], y[valid]
        except Exception:
            return None

    def subtract_baseline(wn, absorbance,
                           lam=1e5, p=0.001):
        try:
            fitter   = Baseline(x_data=wn)
            bl, _    = fitter.asls(
                absorbance, lam=lam, p=p,
                max_iter=100)
            return np.maximum(absorbance - bl, 0.0), bl
        except Exception:
            bl = np.linspace(absorbance[0],
                              absorbance[-1],
                              len(absorbance))
            return np.maximum(absorbance - bl, 0.0), bl

    def find_peaks_auto(wn, absorbance):
        if absorbance.max() < 1e-6:
            return []
        if len(wn) > 1:
            wn_step = (wn.max() - wn.min()) / len(wn)
            min_distance = max(int(8.0/wn_step), 3)
            min_width_pts = max(
                int(PEAK_MIN_WIDTH_WN/wn_step), 2)
        else:
            return []
        prominence = (PEAK_PROMINENCE_FRAC *
                      absorbance.max())
        try:
            peak_idx, props = signal.find_peaks(
                absorbance,
                prominence=prominence,
                distance=min_distance,
                width=min_width_pts)
        except Exception:
            return []
        if len(peak_idx) == 0:
            return []
        if len(peak_idx) > MAX_PEAKS_PER_FILE:
            prom = signal.peak_prominences(
                absorbance, peak_idx)[0]
            top  = np.argsort(prom)[::-1]
            peak_idx = peak_idx[
                top[:MAX_PEAKS_PER_FILE]]
        peaks = [(float(wn[i]),
                  float(absorbance[i]))
                 for i in peak_idx]
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks

    def asymmetric_gaussian(x, center, amplitude,
                             sigma_left, sigma_right):
        result = np.zeros_like(x, dtype=float)
        left   = x < center
        right  = ~left
        if left.any():
            result[left] = amplitude * np.exp(
                -0.5 * ((x[left]-center)/sigma_left)**2)
        if right.any():
            result[right] = amplitude * np.exp(
                -0.5 * ((x[right]-center)/sigma_right)**2)
        return result

    def fit_gaussian_peaks(wn, absorbance,
                            initial_peaks,
                            verbose=False):
        if not initial_peaks:
            return None
        n = len(initial_peaks)
        params = lmfit.Parameters()
        for i, (c0, a0) in enumerate(initial_peaks):
            ci = np.argmin(np.abs(wn - c0))
            hm = a0 / 2.0
            li, ri = ci, ci
            while li > 0 and absorbance[li] > hm:
                li -= 1
            while ri < len(absorbance)-1 and \
                  absorbance[ri] > hm:
                ri += 1
            se = max((wn[ci]-wn[li])*0.8, 10.0)
            se = min(se, SIGMA_MAX/2)
            pfx = f'p{i}_'
            params.add(pfx+'center', value=c0,
                        min=max(wn.min(), c0-50),
                        max=min(wn.max(), c0+50))
            params.add(pfx+'amplitude', value=a0,
                        min=AMPLITUDE_MIN, max=a0*5)
            params.add(pfx+'sigma_left', value=se,
                        min=SIGMA_MIN, max=SIGMA_MAX)
            params.add(pfx+'sigma_right', value=se,
                        min=SIGMA_MIN, max=SIGMA_MAX)

        def composite(x, **kw):
            total = np.zeros_like(x, dtype=float)
            for i in range(n):
                pfx = f'p{i}_'
                total += asymmetric_gaussian(
                    x, kw[pfx+'center'],
                    kw[pfx+'amplitude'],
                    kw[pfx+'sigma_left'],
                    kw[pfx+'sigma_right'])
            return total

        model = lmfit.Model(composite,
                             independent_vars=['x'])
        model.nan_policy = 'omit'
        try:
            result = model.fit(
                absorbance, params, x=wn,
                method='leastsq', max_nfev=5000)
        except Exception as e:
            if verbose:
                print(f"Fit error: {e}")
            return None
        pv     = result.params.valuesdict()
        peaks_out = []
        for i in range(n):
            pfx = f'p{i}_'
            c   = pv.get(pfx+'center',
                         initial_peaks[i][0])
            a   = pv.get(pfx+'amplitude', 0.0)
            sl  = pv.get(pfx+'sigma_left',  10.0)
            sr  = pv.get(pfx+'sigma_right', 10.0)
            fwhm = (sl+sr) * np.sqrt(2*np.log(2))
            wl_um = 1e4/c if c > 0 else None
            in_miri = (c >= MIRI_WN_MIN and
                       c <= MIRI_WN_MAX)
            peaks_out.append({
                'center_wn'   : round(float(c), 2),
                'center_um'   : (round(float(wl_um),4)
                                  if wl_um else None),
                'amplitude'   : round(float(a), 6),
                'sigma_left'  : round(float(sl), 2),
                'sigma_right' : round(float(sr), 2),
                'fwhm_wn'     : round(float(fwhm), 2),
                'in_miri_range': in_miri,
            })
        peaks_out.sort(
            key=lambda x: x['amplitude'], reverse=True)
        rms = float(np.sqrt(np.mean(
            (absorbance - result.best_fit)**2)))
        return {
            'n_peaks'     : n,
            'n_miri_peaks': sum(
                1 for p in peaks_out
                if p['in_miri_range']),
            'peaks'       : peaks_out,
            'fit_rms'     : round(rms, 6),
            'fit_ok'      : result.success,
            'fit_message' : str(result.message),
            'redchi'      : (round(float(result.redchi),6)
                              if result.redchi else None),
        }

# ══════════════════════════════════════════════════════════
# COLOUR PALETTE for Gaussian components
# ══════════════════════════════════════════════════════════
PEAK_COLOURS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#34495e',
    '#e91e63', '#00bcd4', '#8bc34a', '#ff5722',
    '#673ab7', '#009688', '#ffc107', '#607d8b',
    '#795548', '#f44336', '#2196f3', '#4caf50',
]

# ══════════════════════════════════════════════════════════
# CORE VISUALIZER FUNCTION
# ══════════════════════════════════════════════════════════
def visualize_file(fpath, db_entry=None,
                   save_path=None):
    """
    Full 5-panel visualization of Gaussian decomposition
    for one LIDA file.

    fpath    : full path to .dat file
    db_entry : pre-computed entry from JSON database
               (if None, compute on the fly)
    save_path: if provided, save figure instead of show
    """
    fname = os.path.basename(fpath)
    print(f"\n  Visualizing: {fname}")

    # ── Load raw data ─────────────────────────────────
    raw = load_lida_file(fpath)
    if raw is None:
        print(f"  ERROR: Could not load {fpath}")
        return
    wn_raw, abs_raw = raw

    # ── Baseline subtraction ──────────────────────────
    abs_corr, baseline_full = subtract_baseline(
        wn_raw, abs_raw)

    # ── MIRI-range slice ──────────────────────────────
    miri_mask  = ((wn_raw >= MIRI_WN_MIN - 100) &
                   (wn_raw <= MIRI_WN_MAX + 100))
    wn_miri    = wn_raw[miri_mask]
    abs_miri   = abs_corr[miri_mask]
    abs_raw_m  = abs_raw[miri_mask]
    bl_miri    = baseline_full[miri_mask]

    if len(wn_miri) < 20:
        wn_miri   = wn_raw
        abs_miri  = abs_corr
        abs_raw_m = abs_raw
        bl_miri   = baseline_full

    # ── Get Gaussian fit ──────────────────────────────
    if db_entry is not None:
        # Use precomputed database entry
        peaks_data = db_entry['peaks']
        fit_rms    = db_entry['fit_rms']
        fit_ok     = db_entry['fit_ok']
        n_peaks    = db_entry['n_peaks']

        # Reconstruct fitted curve
        fitted = np.zeros_like(wn_miri)
        for p in peaks_data:
            fitted += asymmetric_gaussian(
                wn_miri,
                p['center_wn'],
                p['amplitude'],
                p['sigma_left'],
                p['sigma_right'])
        initial_peaks = [
            (p['center_wn'], p['amplitude'])
            for p in peaks_data]
    else:
        # Compute on the fly
        print(f"  Finding peaks...")
        initial_peaks = find_peaks_auto(
            wn_miri, abs_miri)
        print(f"  Found {len(initial_peaks)} peaks. "
              f"Fitting Gaussians...")
        fit_result = fit_gaussian_peaks(
            wn_miri, abs_miri, initial_peaks,
            verbose=True)
        if fit_result is None:
            print(f"  Fit failed — showing raw data only")
            peaks_data  = []
            fitted      = np.zeros_like(wn_miri)
            fit_rms     = None
            fit_ok      = False
            n_peaks     = 0
        else:
            peaks_data = fit_result['peaks']
            fit_rms    = fit_result['fit_rms']
            fit_ok     = fit_result['fit_ok']
            n_peaks    = fit_result['n_peaks']
            fitted     = np.zeros_like(wn_miri)
            for p in peaks_data:
                fitted += asymmetric_gaussian(
                    wn_miri,
                    p['center_wn'],
                    p['amplitude'],
                    p['sigma_left'],
                    p['sigma_right'])

    residual = abs_miri - fitted

    # ── Gaussian MIRI-grid reconstruction ────────────
    miri_gaussian = np.zeros(MIRI_N_PTS)
    for p in peaks_data:
        miri_gaussian += asymmetric_gaussian(
            MIRI_WN_GRID,
            p['center_wn'],
            p['amplitude'],
            p['sigma_left'],
            p['sigma_right'])

    # Raw data on MIRI grid (for comparison)
    if abs_miri.max() > 0:
        fi_raw   = interp1d(
            wn_miri, abs_miri,
            bounds_error=False, fill_value=0.0)
        miri_raw = np.maximum(
            fi_raw(MIRI_WN_GRID), 0.0)
    else:
        miri_raw = np.zeros(MIRI_N_PTS)

    # Convert wavenumber to wavelength (µm)
    # for x-axis labels
    def wn2um(wn):
        with np.errstate(divide='ignore'):
            return np.where(wn > 0, 1e4/wn, np.nan)

    # ── Metadata ──────────────────────────────────────
    species = get_species(fname)
    temp_K  = extract_temperature(fname)
    title   = (f"{fname}\n"
               f"Species: {', '.join(species) if species else 'unknown'}"
               f"  |  T = {temp_K:.0f} K" if temp_K
               else
               f"{fname}\n"
               f"Species: {', '.join(species) if species else 'unknown'}"
               f"  |  T = unknown")

    # ══════════════════════════════════════════════════
    # FIGURE LAYOUT
    # 5 panels:
    #   0: Full raw spectrum (all wavenumbers)
    #   1: MIRI-range: raw + baseline + corrected
    #   2: MIRI-range: baseline-subtracted + each Gaussian
    #   3: Residuals
    #   4: Reconstructed MIRI-grid Gaussian vs raw
    # ══════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 22),
                     facecolor='#0d1117')
    fig.suptitle(title,
                 color='white', fontsize=11,
                 fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(
        5, 1, figure=fig,
        hspace=0.45,
        top=0.96, bottom=0.04,
        left=0.08, right=0.97)

    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    style = {
        'facecolor': '#161b22',
        'text_color': '#c9d1d9',
        'grid_color': '#21262d',
        'raw_color': '#58a6ff',
        'baseline_color': '#f85149',
        'corrected_color': '#3fb950',
        'fitted_color': '#ffa657',
        'residual_color': '#bc8cff',
        'miri_shade': '#1f2937',
    }

    for ax in axes:
        ax.set_facecolor(style['facecolor'])
        ax.tick_params(colors=style['text_color'],
                       which='both')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['right'].set_color('#30363d')
        ax.xaxis.label.set_color(style['text_color'])
        ax.yaxis.label.set_color(style['text_color'])
        ax.grid(True, color=style['grid_color'],
                lw=0.5, alpha=0.8)

    # ── Panel 0: Full raw spectrum ────────────────────
    ax = axes[0]
    ax.plot(wn_raw, abs_raw,
            color=style['raw_color'],
            lw=0.8, alpha=0.9, label='Raw')
    ax.plot(wn_raw, baseline_full,
            color=style['baseline_color'],
            lw=1.2, ls='--', alpha=0.8,
            label='ALS baseline')
    # Shade MIRI range
    ax.axvspan(MIRI_WN_MIN, MIRI_WN_MAX,
               alpha=0.1, color='#ffa657',
               label=f'MIRI range '
                     f'({MIRI_WN_MIN:.0f}–'
                     f'{MIRI_WN_MAX:.0f} cm⁻¹)')
    ax.set_xlabel('Wavenumber (cm⁻¹)',
                  color=style['text_color'])
    ax.set_ylabel('Absorbance',
                  color=style['text_color'])
    ax.set_title('Full Spectrum — Raw + ALS Baseline',
                 color='#ffa657', pad=6,
                 fontsize=10)
    ax.legend(fontsize=8, facecolor='#21262d',
              labelcolor='white',
              edgecolor='#30363d')
    # Add wavelength axis on top
    ax2 = ax.twiny()
    ax2.set_facecolor(style['facecolor'])
    ax2.tick_params(colors=style['text_color'])
    wl_ticks = np.array([5, 6, 7, 8, 10, 12,
                          15, 20, 25, 28])
    wn_ticks = 1e4 / wl_ticks
    valid_t  = ((wn_ticks >= wn_raw.min()) &
                (wn_ticks <= wn_raw.max()))
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(wn_ticks[valid_t])
    ax2.set_xticklabels(
        [f'{w:.0f}' for w in wl_ticks[valid_t]],
        color=style['text_color'], fontsize=7)
    ax2.set_xlabel('Wavelength (µm)',
                   color=style['text_color'],
                   fontsize=8)

    # ── Panel 1: MIRI range raw vs corrected ─────────
    ax = axes[1]
    ax.plot(wn_miri, abs_raw_m,
            color=style['raw_color'],
            lw=0.8, alpha=0.7, label='Raw (MIRI range)')
    ax.plot(wn_miri, bl_miri,
            color=style['baseline_color'],
            lw=1.2, ls='--', alpha=0.8,
            label='ALS baseline')
    ax.plot(wn_miri, abs_miri,
            color=style['corrected_color'],
            lw=1.5, alpha=0.9,
            label='Baseline-subtracted')
    ax.set_xlabel('Wavenumber (cm⁻¹)',
                  color=style['text_color'])
    ax.set_ylabel('Absorbance',
                  color=style['text_color'])
    ax.set_title('MIRI Range — Baseline Subtraction',
                 color='#ffa657', pad=6,
                 fontsize=10)
    ax.legend(fontsize=8, facecolor='#21262d',
              labelcolor='white',
              edgecolor='#30363d')
    # Ice band markers
    ice_bands = {
        'H₂O str': 3280, 'H₂O bend': 1630,
        'CO₂ bend': 660,  'CO₂ str': 2342,
        'CH₄': 1305,      'NH₄⁺': 1460,
        'CO': 2090,        'SO₂': 1150,
        'CH₃OH': 1030,    'HCOOH': 1720,
        'NH₃': 1070,      'OCN⁻': 2165,
    }
    for lbl, wn_b in ice_bands.items():
        if wn_miri.min() < wn_b < wn_miri.max():
            ax.axvline(wn_b, color='#adbac7',
                       lw=0.5, ls=':', alpha=0.5)
            ax.text(wn_b + 5,
                    abs_raw_m.max() * 0.92,
                    lbl, color='#adbac7',
                    fontsize=6, rotation=90,
                    va='top')

    # ── Panel 2: Gaussian components ──────────────────
    ax = axes[2]
    ax.plot(wn_miri, abs_miri,
            color=style['corrected_color'],
            lw=1.5, alpha=0.8,
            label='Baseline-subtracted', zorder=10)
    ax.plot(wn_miri, fitted,
            color=style['fitted_color'],
            lw=2.0, ls='--', alpha=0.95,
            label=f'Total fit  '
                  f'(RMS={fit_rms:.4f})'
                  if fit_rms else 'Total fit',
            zorder=11)

    # Individual Gaussian components
    for i, p in enumerate(peaks_data):
        col = PEAK_COLOURS[i % len(PEAK_COLOURS)]
        g   = asymmetric_gaussian(
            wn_miri,
            p['center_wn'],
            p['amplitude'],
            p['sigma_left'],
            p['sigma_right'])
        miri_tag = '★' if p['in_miri_range'] else '○'
        um_str   = (f"{p['center_um']:.2f}µm"
                    if p['center_um'] else "")
        ax.fill_between(wn_miri, 0, g,
                        alpha=0.25, color=col)
        ax.plot(wn_miri, g, color=col,
                lw=1.0, alpha=0.9,
                label=f'{miri_tag} G{i+1}: '
                      f'{p["center_wn"]:.0f}cm⁻¹ '
                      f'{um_str} '
                      f'A={p["amplitude"]:.3f} '
                      f'FWHM={p["fwhm_wn"]:.0f}cm⁻¹')
        # Mark peak center
        ax.axvline(p['center_wn'],
                   color=col, lw=0.7,
                   ls=':', alpha=0.6)

    ax.set_xlabel('Wavenumber (cm⁻¹)',
                  color=style['text_color'])
    ax.set_ylabel('Absorbance',
                  color=style['text_color'])
    stars = ('★ = in MIRI range   '
             '○ = outside MIRI range')
    ax.set_title(
        f'Gaussian Decomposition  '
        f'({n_peaks} peaks)  |  {stars}',
        color='#ffa657', pad=6, fontsize=10)
    ax.legend(fontsize=7, facecolor='#21262d',
              labelcolor='white',
              edgecolor='#30363d',
              ncol=2,
              loc='upper right')

    # ── Panel 3: Residuals ────────────────────────────
    ax = axes[3]
    ax.plot(wn_miri, residual,
            color=style['residual_color'],
            lw=0.9, alpha=0.9,
            label='Residual (data − fit)')
    ax.fill_between(wn_miri, 0, residual,
                    where=residual > 0,
                    alpha=0.35, color='#f85149',
                    label='Over-estimated')
    ax.fill_between(wn_miri, 0, residual,
                    where=residual < 0,
                    alpha=0.35, color='#3fb950',
                    label='Under-estimated')
    ax.axhline(0, color='#58a6ff', lw=1.0, ls='--')
    rms_pct = (fit_rms / abs_miri.max() * 100
               if fit_rms and abs_miri.max() > 0
               else 0)
    ax.set_xlabel('Wavenumber (cm⁻¹)',
                  color=style['text_color'])
    ax.set_ylabel('Residual',
                  color=style['text_color'])
    ax.set_title(
        f'Fit Residuals  |  '
        f'RMS = {fit_rms:.5f}  '
        f'({rms_pct:.1f}% of peak)  |  '
        f'{"✔ Converged" if fit_ok else "✗ May not have converged"}',
        color='#ffa657', pad=6, fontsize=10)
    ax.legend(fontsize=8, facecolor='#21262d',
              labelcolor='white',
              edgecolor='#30363d')

    # ── Panel 4: MIRI-grid reconstruction ────────────
    ax = axes[4]
    # Wavelength axis for this panel
    wl_grid = 1e4 / MIRI_WN_GRID

    # Normalise both to max=1 for comparison
    raw_max  = miri_raw.max()
    gaus_max = miri_gaussian.max()
    raw_norm  = (miri_raw / raw_max
                  if raw_max > 0 else miri_raw)
    gaus_norm = (miri_gaussian / gaus_max
                  if gaus_max > 0 else miri_gaussian)

    ax.plot(wl_grid, raw_norm,
            color=style['raw_color'],
            lw=0.9, alpha=0.7,
            label='Raw data (normalised)')
    ax.plot(wl_grid, gaus_norm,
            color=style['fitted_color'],
            lw=2.0, alpha=0.95,
            label='Gaussian reconstruction (normalised)')
    ax.fill_between(wl_grid, 0, gaus_norm,
                    alpha=0.12, color=style['fitted_color'])
    ax.set_xlabel('Wavelength (µm)',
                  color=style['text_color'])
    ax.set_ylabel('Normalised absorbance',
                  color=style['text_color'])
    n_miri = sum(1 for p in peaks_data
                 if p['in_miri_range'])
    ax.set_title(
        f'MIRI-Grid Gaussian Reconstruction  |  '
        f'{n_miri} MIRI-range peaks  |  '
        f'5–28 µm',
        color='#ffa657', pad=6, fontsize=10)
    ax.legend(fontsize=8, facecolor='#21262d',
              labelcolor='white',
              edgecolor='#30363d')
    ax.set_xlim(5, 28)
    ax.invert_xaxis()
    # Ice band wavelength markers
    for lbl, wl_b in [
        ('H₂O', 6.0), ('NH₄⁺', 6.85),
        ('CH₄', 7.7),  ('Sil', 9.7),
        ('CO₂', 15.2), ('CO', 4.67),
        ('CH₃OH', 9.7), ('SO₂', 8.7),
    ]:
        ax.axvline(wl_b, color='#adbac7',
                   lw=0.6, ls=':', alpha=0.4)
        ax.text(wl_b + 0.15,
                gaus_norm.max() * 0.88,
                lbl, color='#adbac7',
                fontsize=6, rotation=90,
                va='top')

    # ── Print peak table ──────────────────────────────
    print(f"\n  {'Peak':>4}  {'Center(cm⁻¹)':>12}  "
          f"{'Center(µm)':>10}  {'Ampl':>8}  "
          f"{'σL':>6}  {'σR':>6}  "
          f"{'FWHM':>7}  {'MIRI':>5}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}  "
          f"{'-'*8}  {'-'*6}  {'-'*6}  "
          f"{'-'*7}  {'-'*5}")
    for i, p in enumerate(peaks_data):
        um_str = (f"{p['center_um']:.3f}"
                  if p['center_um'] else "—")
        print(f"  {i+1:>4}  "
              f"{p['center_wn']:>12.1f}  "
              f"{um_str:>10}  "
              f"{p['amplitude']:>8.5f}  "
              f"{p['sigma_left']:>6.1f}  "
              f"{p['sigma_right']:>6.1f}  "
              f"{p['fwhm_wn']:>7.1f}  "
              f"{'yes' if p['in_miri_range'] else 'no':>5}")
    if fit_rms:
        print(f"\n  Fit RMS: {fit_rms:.6f}  "
              f"({rms_pct:.2f}% of peak)  "
              f"{'OK' if fit_ok else 'FAILED'}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=150,
                    bbox_inches='tight',
                    facecolor='#0d1117')
        print(f"\n  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)

# ══════════════════════════════════════════════════════════
# FILE FINDER — fuzzy search in LIDA directory
# ══════════════════════════════════════════════════════════
def find_lida_file(query, lida_dir):
    """
    Find LIDA files matching a query string.
    Returns list of matching file paths.
    """
    query_up = query.upper()
    matches  = []
    try:
        all_files = os.listdir(lida_dir)
    except Exception:
        return []
    for fname in sorted(all_files):
        if not fname.lower().endswith(
                ('.dat', '.txt', '.dat1', '.csv')):
            continue
        if query_up in fname.upper():
            matches.append(
                os.path.join(lida_dir, fname))
    return matches

def list_by_species(species, lida_dir):
    """
    List all LIDA files containing a given species.
    """
    matches = []
    try:
        all_files = os.listdir(lida_dir)
    except Exception:
        return []
    sp_up = species.upper().replace('+','').replace('-','')
    for fname in sorted(all_files):
        if not fname.lower().endswith(
                ('.dat', '.txt', '.dat1', '.csv')):
            continue
        fsp = get_species(fname)
        fsp_str = ' '.join(
            s.upper().replace('+','').replace('-','')
            for s in fsp)
        if sp_up in fsp_str:
            temp = extract_temperature(fname)
            matches.append((fname, temp, fsp))
    return matches

# ══════════════════════════════════════════════════════════
# INTERACTIVE BROWSER
# For when you want to browse files interactively
# ══════════════════════════════════════════════════════════
def interactive_browser(lida_dir, db=None):
    """
    Simple command-line browser that lets you type
    a file name (or partial name) and immediately
    see the Gaussian decomposition plot.
    """
    print("\n" + "="*60)
    print("  INDRA 2.0 — Gaussian Peak Visualizer")
    print("  Interactive mode")
    print("="*60)
    print(f"\n  LIDA dir: {lida_dir}")
    if db:
        print(f"  Database loaded: {len(db)} entries")
    print("\n  Commands:")
    print("    <query>        — search and plot file")
    print("    list <species> — list files for species")
    print("    save <query>   — save plot instead of show")
    print("    q or quit      — exit\n")

    while True:
        try:
            cmd = input("  Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if not cmd:
            continue
        if cmd.lower() in ('q', 'quit', 'exit'):
            print("  Bye.")
            break

        # List mode
        if cmd.lower().startswith('list '):
            sp = cmd[5:].strip()
            matches = list_by_species(sp, lida_dir)
            if not matches:
                print(f"  No files found for "
                      f"species: {sp}")
            else:
                print(f"\n  {len(matches)} files "
                      f"containing '{sp}':")
                for fname, temp, fsp in matches:
                    t_str = (f"T={temp:.0f}K"
                              if temp else "T=?")
                    print(f"    {fname[:55]:<55}  "
                          f"{t_str}  "
                          f"{', '.join(fsp)}")
            print()
            continue

        # Save mode
        save_mode = False
        if cmd.lower().startswith('save '):
            cmd       = cmd[5:].strip()
            save_mode = True

        # Find matching files
        matches = find_lida_file(cmd, lida_dir)
        if not matches:
            print(f"  No files found matching '{cmd}'")
            print(f"  Try a partial name like: H2O, "
                  f"CO2, NH3, HCOOH, h2o-nh3")
            continue

        if len(matches) == 1:
            chosen = matches[0]
        else:
            print(f"\n  Found {len(matches)} matches:")
            for i, p in enumerate(matches[:20]):
                print(f"    [{i}] "
                      f"{os.path.basename(p)}")
            if len(matches) > 20:
                print(f"  ... and {len(matches)-20} more")
            try:
                idx_str = input(
                    f"  Choose [0–{min(len(matches),20)-1}]"
                    f" (Enter=0): ").strip()
                idx = int(idx_str) if idx_str else 0
                chosen = matches[idx]
            except (ValueError, IndexError):
                chosen = matches[0]

        fname    = os.path.basename(chosen)
        db_entry = db.get(fname) if db else None

        if db_entry:
            print(f"  Using precomputed database entry")
        else:
            print(f"  Computing on-the-fly "
                  f"(may take 10–30 seconds)...")

        save_path = None
        if save_mode:
            save_path = (
                fname.replace('.dat', '').replace(
                    '.txt', '') + '_gaussian.png')

        visualize_file(chosen,
                       db_entry=db_entry,
                       save_path=save_path)

# ══════════════════════════════════════════════════════════
# BATCH VISUALIZE (save all plots for a species)
# ══════════════════════════════════════════════════════════
def batch_visualize_species(species, lida_dir,
                              output_dir, db=None):
    """
    Save Gaussian decomposition plots for all files
    matching a given species.
    Useful for checking all temperature variants at once.
    """
    os.makedirs(output_dir, exist_ok=True)
    matches = list_by_species(species, lida_dir)
    if not matches:
        print(f"No files found for species: {species}")
        return
    print(f"\n  Saving plots for {len(matches)} "
          f"{species} files...")
    for fname, temp, fsp in matches:
        fpath    = os.path.join(lida_dir, fname)
        db_entry = db.get(fname) if db else None
        save_path = os.path.join(
            output_dir,
            fname.replace('.dat','').replace(
                '.txt','') + '_gauss.png')
        print(f"  → {fname}")
        try:
            visualize_file(fpath,
                           db_entry=db_entry,
                           save_path=save_path)
        except Exception as e:
            print(f"    ERROR: {e}")

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='INDRA 2.0 Gaussian Visualizer')
    parser.add_argument(
        'query', nargs='?', default=None,
        help='File name or partial name to visualize')
    parser.add_argument(
        '--list-species', metavar='SP',
        help='List all files for a species')
    parser.add_argument(
        '--from-db', metavar='FNAME',
        help='Load specific file from database')
    parser.add_argument(
        '--batch-species', metavar='SP',
        help='Save plots for all files of a species')
    parser.add_argument(
        '--save', metavar='PATH',
        help='Save plot to file instead of showing')
    parser.add_argument(
        '--no-db', action='store_true',
        help='Ignore database, always compute on-the-fly')
    args = parser.parse_args()

    # Load database if available
    db = None
    if not args.no_db and os.path.exists(DB_PATH):
        print(f"  Loading database from {DB_PATH}...")
        with open(DB_PATH, 'r') as f:
            db = json.load(f)
        print(f"  {len(db)} entries loaded")
    else:
        print(f"  No database found. "
              f"Computing fits on-the-fly.")

    # List species mode
    if args.list_species:
        matches = list_by_species(
            args.list_species, LIDA_DIR)
        print(f"\n  {len(matches)} files for "
              f"'{args.list_species}':")
        for fname, temp, fsp in matches:
            t_str = (f"T={temp:.0f}K"
                      if temp else "T=?K")
            print(f"  {fname[:55]:<55}  {t_str}")
        sys.exit(0)

    # Batch species mode
    if args.batch_species:
        out_dir = (os.path.dirname(DB_PATH) +
                   f"/plots_{args.batch_species}/")
        batch_visualize_species(
            args.batch_species, LIDA_DIR,
            out_dir, db=db)
        sys.exit(0)

    # Direct from-db mode
    if args.from_db:
        fname = args.from_db
        fpath = os.path.join(LIDA_DIR, fname)
        if not os.path.exists(fpath):
            # Try fuzzy search
            matches = find_lida_file(fname, LIDA_DIR)
            if matches:
                fpath = matches[0]
                fname = os.path.basename(fpath)
            else:
                print(f"File not found: {fname}")
                sys.exit(1)
        db_entry = db.get(fname) if db else None
        visualize_file(fpath,
                       db_entry=db_entry,
                       save_path=args.save)
        sys.exit(0)

    # Single query mode
    if args.query:
        matches = find_lida_file(args.query, LIDA_DIR)
        if not matches:
            print(f"No files found for '{args.query}'")
            sys.exit(1)
        fpath    = matches[0]
        fname    = os.path.basename(fpath)
        db_entry = db.get(fname) if db else None
        print(f"  Plotting: {fname}")
        visualize_file(fpath,
                       db_entry=db_entry,
                       save_path=args.save)
        sys.exit(0)

    # Default: interactive browser
    interactive_browser(LIDA_DIR, db=db)