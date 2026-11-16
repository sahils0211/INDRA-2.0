# ============================================================
# INDRA 2.0 — Gaussian Peak Decomposition (Parallel)
#
# Uses ProcessPoolExecutor to run across all CPU cores.
# On a Ryzen 7 (8 cores / 16 threads) this runs 8-12x
# faster than the sequential version.
#
# Key changes vs sequential version:
#   - Each file processed in a worker process (true parallelism,
#     bypasses Python GIL because each worker is a separate process)
#   - Results collected and saved in batches
#   - Progress bar shows live throughput and ETA
#   - Already-processed files skipped automatically (resumable)
#   - Worker processes are stateless — safe to kill and restart
#
# Usage:
#   python gaussian_decompose_parallel.py
#   python gaussian_decompose_parallel.py --workers 12
#   python gaussian_decompose_parallel.py --workers 6 --batch-size 100
# ============================================================

import numpy as np
import json
import os
import re
import sys
import time
import argparse
import warnings
from pathlib import Path
from concurrent.futures import (ProcessPoolExecutor,
                                 as_completed)
import multiprocessing

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# PATHS — edit here
# ══════════════════════════════════════════════════════════
LIDA_DIR   = "C:/Users/sahil/Downloads/lida_raw_data/"
OUTPUT_DIR = "C:/Users/sahil/Downloads/gaussian_db/"
DB_PATH    = OUTPUT_DIR + "gaussian_peaks_database.json"
FAIL_LOG   = OUTPUT_DIR + "gaussian_failed.txt"

# ══════════════════════════════════════════════════════════
# SPECTRAL CONSTANTS
# ══════════════════════════════════════════════════════════
MIRI_WN_MIN  = 357.0
MIRI_WN_MAX  = 2000.0
MIRI_N_PTS   = 3000
MIRI_WN_GRID = np.linspace(MIRI_WN_MIN,
                             MIRI_WN_MAX,
                             MIRI_N_PTS)

# Peak finding
PEAK_PROMINENCE_FRAC = 0.04
PEAK_MIN_WIDTH_WN    = 5.0
PEAK_MAX_WIDTH_WN    = 500.0
MAX_PEAKS_PER_FILE   = 20
SIGMA_MIN            = 3.0
SIGMA_MAX            = 400.0
AMPLITUDE_MIN        = 1e-5

# ══════════════════════════════════════════════════════════
# PARENT RULES
# ══════════════════════════════════════════════════════════
PARENT_RULES = [
    ('CH3CH2OH','CH3CH2OH'), ('C2H5OH','CH3CH2OH'),
    ('CH3COOH','CH3COOH'),   ('CH3OCHO','CH3OCHO'),
    ('CH3CHO','CH3CHO'),     ('CH3OH','CH3OH'),
    ('CH3NH2','CH3NH2'),     ('CH4','CH4'),
    ('H2CO','H2CO'),         ('HCOOH','HCOOH'),
    ('HCOO','HCOO-'),        ('NH2CHO','NH2CHO'),
    ('HNCO','HNCO'),         ('NH4','NH4+'),
    ('NHPLUS','NH4+'),       ('NH3','NH3'),
    ('H2S','H2S'),           ('SO2','SO2'),
    ('CO2','CO2'),           ('C18O','CO2'),
    ('H2O','H2O'),           ('CO','CO'),
    ('SILICATE','Silicate'),  ('OLIVINE','Silicate'),
    ('SPEC_D','organic'),    ('SPEC_A','organic'),
    ('ORGANIC','organic'),   ('RESIDUE','organic'),
    ('C3H6O','CH3CHO'),      ('OCS','OCS'),
    ('OCN','OCN-'),          ('HNCO_NH3','HNCO'),
    ('HNCO_NH3','NH3'),
]

def get_species(label):
    lu = label.upper()
    return sorted({p for k, p in PARENT_RULES
                    if k.upper() in lu})

def extract_temperature(fname):
    matches = list(re.finditer(
        r'(?<![0-9])(\d+(?:\.\d+)?)[Kk](?![a-zA-Z])',
        fname))
    if matches:
        try:
            return float(matches[-1].group(1))
        except ValueError:
            return None
    return None

# ══════════════════════════════════════════════════════════
# ALL PROCESSING LOGIC IN ONE SELF-CONTAINED FUNCTION
# This function runs inside a worker process.
# It must import everything it needs locally because
# worker processes do not share memory with the main
# process — all imports happen fresh in each worker.
# ══════════════════════════════════════════════════════════
def process_one_file(args):
    """
    Worker function — processes a single LIDA file.
    Designed to run in a subprocess via ProcessPoolExecutor.

    args: (fpath, miri_wn_grid, miri_n_pts,
            miri_wn_min, miri_wn_max)

    Returns: (filename, entry_dict, status_string)
      entry_dict is None if processing failed.
    """
    # Local imports — each worker process needs its own
    import numpy as np
    import re
    import os
    import warnings
    import scipy.signal as signal
    from scipy.interpolate import interp1d
    from pybaselines import Baseline
    import lmfit
    warnings.filterwarnings('ignore')

    (fpath,
     MIRI_WN_GRID,
     MIRI_N_PTS,
     MIRI_WN_MIN,
     MIRI_WN_MAX) = args

    fname = os.path.basename(fpath)

    # ── 1. Load file ──────────────────────────────────
    def load_file(path):
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
                mask = x > 0
                wn   = 1e4 / x[mask]
                y    = y[mask]
                wn   = wn[::-1]
                y    = y[::-1]
                x    = wn
            y_max = np.percentile(y, 99)
            y_min = np.percentile(y, 1)
            if y_min >= -0.05 and y_max <= 1.05:
                y = -np.log10(np.clip(y, 1e-6, 1.0))
            elif y_min < -0.3:
                y = -y
            y = np.maximum(y, 0.0)
            valid = ((x >= 200) & (x <= 5000) &
                     np.isfinite(y))
            return (x[valid], y[valid]) \
                   if valid.sum() >= 30 else None
        except Exception:
            return None

    raw = load_file(fpath)
    if raw is None:
        return fname, None, "load_failed"
    wn, absorbance = raw

    if absorbance.max() < 1e-5:
        return fname, None, "all_zero"

    # ── 2. Baseline subtraction ───────────────────────
    def subtract_baseline(wn, absorbance):
        try:
            bl, _ = Baseline(x_data=wn).asls(
                absorbance, lam=1e5, p=0.001,
                max_iter=100)
            return np.maximum(absorbance - bl, 0.0), bl
        except Exception:
            bl = np.linspace(
                absorbance[0], absorbance[-1],
                len(absorbance))
            return np.maximum(
                absorbance - bl, 0.0), bl

    abs_corr, _ = subtract_baseline(wn, absorbance)

    if abs_corr.max() < 1e-5:
        return fname, None, "zero_after_baseline"

    # ── 3. MIRI-range slice ───────────────────────────
    miri_mask = ((wn >= MIRI_WN_MIN - 100) &
                  (wn <= MIRI_WN_MAX + 100))
    if miri_mask.sum() >= 20:
        wn_fit  = wn[miri_mask]
        abs_fit = abs_corr[miri_mask]
    else:
        wn_fit  = wn
        abs_fit = abs_corr

    # ── 4. Peak finding ───────────────────────────────
    def find_peaks(wn_f, abs_f):
        if abs_f.max() < 1e-6 or len(wn_f) < 10:
            return []
        wn_step = (wn_f.max()-wn_f.min()) / len(wn_f)
        if wn_step <= 0:
            return []
        min_dist  = max(int(8.0/wn_step), 3)
        min_width = max(int(5.0/wn_step), 2)
        prom      = 0.04 * abs_f.max()
        try:
            pidx, _ = signal.find_peaks(
                abs_f,
                prominence=prom,
                distance=min_dist,
                width=min_width)
        except Exception:
            return []
        if len(pidx) == 0:
            return []
        if len(pidx) > 20:
            proms = signal.peak_prominences(
                abs_f, pidx)[0]
            top  = np.argsort(proms)[::-1]
            pidx = pidx[top[:20]]
        peaks = [(float(wn_f[i]), float(abs_f[i]))
                 for i in pidx]
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks

    initial_peaks = find_peaks(wn_fit, abs_fit)
    if not initial_peaks:
        return fname, None, "no_peaks_found"

    # ── 5. Asymmetric Gaussian fitting ───────────────
    def asym_gauss(x, center, amplitude,
                   sigma_left, sigma_right):
        r    = np.zeros_like(x, dtype=float)
        left = x < center
        rght = ~left
        if left.any():
            r[left] = amplitude * np.exp(
                -0.5*((x[left]-center)/sigma_left)**2)
        if rght.any():
            r[rght] = amplitude * np.exp(
                -0.5*((x[rght]-center)/sigma_right)**2)
        return r

    n      = len(initial_peaks)
    params = lmfit.Parameters()

    for i, (c0, a0) in enumerate(initial_peaks):
        ci = np.argmin(np.abs(wn_fit - c0))
        hm = a0 / 2.0
        li = ci
        ri = ci
        while li > 0 and abs_fit[li] > hm:
            li -= 1
        while ri < len(abs_fit)-1 and abs_fit[ri] > hm:
            ri += 1
        se = max((wn_fit[ci] - wn_fit[li]) * 0.8,
                 10.0)
        se = min(se, 200.0)
        pfx = f'p{i}_'
        params.add(pfx+'center', value=c0,
                    min=max(wn_fit.min(), c0-50),
                    max=min(wn_fit.max(), c0+50))
        params.add(pfx+'amplitude', value=a0,
                    min=1e-5, max=a0*5)
        params.add(pfx+'sigma_left',  value=se,
                    min=3.0, max=400.0)
        params.add(pfx+'sigma_right', value=se,
                    min=3.0, max=400.0)

    def composite(x, **kw):
        tot = np.zeros_like(x, dtype=float)
        for i in range(n):
            pfx = f'p{i}_'
            tot += asym_gauss(
                x, kw[pfx+'center'],
                kw[pfx+'amplitude'],
                kw[pfx+'sigma_left'],
                kw[pfx+'sigma_right'])
        return tot

    model = lmfit.Model(composite,
                         independent_vars=['x'])
    model.nan_policy = 'omit'

    try:
        result = model.fit(
            abs_fit, params, x=wn_fit,
            method='leastsq', max_nfev=5000)
    except Exception as e:
        return fname, None, f"fit_failed:{e}"

    # ── 6. Extract peaks ──────────────────────────────
    pv         = result.params.valuesdict()
    peaks_out  = []
    for i in range(n):
        pfx  = f'p{i}_'
        c    = pv.get(pfx+'center',
                      initial_peaks[i][0])
        a    = pv.get(pfx+'amplitude', 0.0)
        sl   = pv.get(pfx+'sigma_left',  10.0)
        sr   = pv.get(pfx+'sigma_right', 10.0)
        fwhm = (sl+sr) * np.sqrt(2*np.log(2))
        wl   = 1e4/c if c > 0 else None
        in_m = (MIRI_WN_MIN <= c <= MIRI_WN_MAX)
        peaks_out.append({
            'center_wn'    : round(float(c), 2),
            'center_um'    : (round(float(wl), 4)
                               if wl else None),
            'amplitude'    : round(float(a), 6),
            'sigma_left'   : round(float(sl), 2),
            'sigma_right'  : round(float(sr), 2),
            'fwhm_wn'      : round(float(fwhm), 2),
            'in_miri_range': in_m,
        })

    peaks_out.sort(key=lambda x: x['amplitude'],
                   reverse=True)

    rms = float(np.sqrt(np.mean(
        (abs_fit - result.best_fit)**2)))

    # ── 7. Reconstruct on MIRI grid ───────────────────
    miri_abs = np.zeros(MIRI_N_PTS)
    for p in peaks_out:
        if not p['in_miri_range']:
            continue
        miri_abs += asym_gauss(
            MIRI_WN_GRID,
            p['center_wn'],
            p['amplitude'],
            p['sigma_left'],
            p['sigma_right'])

    miri_max = miri_abs.max()
    miri_norm = (miri_abs / miri_max
                  if miri_max > 0
                  else miri_abs)

    # ── 8. Build database entry ───────────────────────
    base    = re.sub(
        r'\.(dat|txt|dat1|csv)$', '', fname,
        flags=re.IGNORECASE)

    # Get species (inline — workers don't share globals)
    PARENT_RULES_LOCAL = [
        ('CH3CH2OH','CH3CH2OH'),('C2H5OH','CH3CH2OH'),
        ('CH3COOH','CH3COOH'), ('CH3OCHO','CH3OCHO'),
        ('CH3CHO','CH3CHO'),   ('CH3OH','CH3OH'),
        ('CH3NH2','CH3NH2'),   ('CH4','CH4'),
        ('H2CO','H2CO'),       ('HCOOH','HCOOH'),
        ('HCOO','HCOO-'),      ('NH2CHO','NH2CHO'),
        ('HNCO','HNCO'),       ('NH4','NH4+'),
        ('NHPLUS','NH4+'),     ('NH3','NH3'),
        ('H2S','H2S'),         ('SO2','SO2'),
        ('CO2','CO2'),         ('C18O','CO2'),
        ('H2O','H2O'),         ('CO','CO'),
        ('SILICATE','Silicate'),('OLIVINE','Silicate'),
        ('SPEC_D','organic'),  ('SPEC_A','organic'),
        ('ORGANIC','organic'), ('RESIDUE','organic'),
        ('C3H6O','CH3CHO'),    ('OCS','OCS'),
        ('OCN','OCN-'),        ('HNCO_NH3','HNCO'),
        ('HNCO_NH3','NH3'),
    ]
    lu      = base.upper()
    species = sorted({p for k, p in PARENT_RULES_LOCAL
                       if k.upper() in lu})

    temp_matches = list(re.finditer(
        r'(?<![0-9])(\d+(?:\.\d+)?)[Kk](?![a-zA-Z])',
        fname))
    temp_K = None
    if temp_matches:
        try:
            temp_K = float(temp_matches[-1].group(1))
        except ValueError:
            pass

    entry = {
        'filename'              : fname,
        'species'               : species,
        'temperature_K'         : temp_K,
        'n_data_points'         : int(len(wn)),
        'wn_min'                : round(float(wn.min()),1),
        'wn_max'                : round(float(wn.max()),1),
        'absorbance_max'        : round(
            float(absorbance.max()), 4),
        'fit_rms'               : round(rms, 6),
        'fit_ok'                : bool(result.success),
        'fit_message'           : str(result.message),
        'redchi'                : (
            round(float(result.redchi), 6)
            if result.redchi else None),
        'n_peaks'               : n,
        'n_miri_peaks'          : sum(
            1 for p in peaks_out
            if p['in_miri_range']),
        'peaks'                 : peaks_out,
        'miri_gaussian_spectrum': [
            round(float(v), 8) for v in miri_norm],
    }

    return fname, entry, "ok"

# ══════════════════════════════════════════════════════════
# PROGRESS BAR
# ══════════════════════════════════════════════════════════
class ProgressBar:
    def __init__(self, total, width=50):
        self.total     = total
        self.width     = width
        self.done      = 0
        self.ok        = 0
        self.fail      = 0
        self.start_t   = time.time()

    def update(self, success=True):
        self.done += 1
        if success:
            self.ok += 1
        else:
            self.fail += 1

        elapsed = time.time() - self.start_t
        rate    = self.done / elapsed if elapsed > 0 \
                  else 0
        remain  = (self.total - self.done) / rate \
                  if rate > 0 else 0

        filled = int(self.width *
                     self.done / self.total)
        bar    = ('█' * filled +
                  '░' * (self.width - filled))

        mins, secs = divmod(int(remain), 60)
        sys.stdout.write(
            f"\r  [{bar}] "
            f"{self.done}/{self.total}  "
            f"✔{self.ok} ✗{self.fail}  "
            f"{rate:.1f} f/s  "
            f"ETA {mins}m{secs:02d}s  ")
        sys.stdout.flush()

    def finish(self):
        elapsed = time.time() - self.start_t
        mins, secs = divmod(int(elapsed), 60)
        sys.stdout.write(
            f"\r  Done! {self.done} files in "
            f"{mins}m{secs:02d}s  "
            f"✔{self.ok} ok  ✗{self.fail} failed"
            + " " * 20 + "\n")
        sys.stdout.flush()

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main(n_workers=None, batch_size=50):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Auto-detect worker count
    cpu_count = multiprocessing.cpu_count()
    if n_workers is None:
        # Use all physical cores minus 1
        # (keep one free for OS + main process)
        n_workers = max(1, cpu_count - 1)

    print("=" * 62)
    print("  INDRA 2.0 — Gaussian Decomposition (Parallel)")
    print("=" * 62)
    print(f"  CPU cores available : {cpu_count}")
    print(f"  Worker processes    : {n_workers}")
    print(f"  Batch size          : {batch_size}")
    print(f"  LIDA directory      : {LIDA_DIR}")
    print(f"  Database output     : {DB_PATH}")

    # Collect all files
    extensions = {'.dat', '.txt', '.dat1', '.csv'}
    all_files  = sorted([
        str(f) for f in Path(LIDA_DIR).iterdir()
        if f.is_file() and
           f.suffix.lower() in extensions])

    print(f"\n  Total files found   : {len(all_files)}")

    # Load existing database (resumable)
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            database = json.load(f)
        print(f"  Already processed   : {len(database)} "
              f"(will skip)")
    else:
        database = {}

    # Filter out already-done files
    todo = [f for f in all_files
            if os.path.basename(f) not in database]

    print(f"  Remaining to process: {len(todo)}")

    if not todo:
        print("\n  All files already processed!")
        print(f"  Database: {DB_PATH}")
        return

    print(f"\n  Starting parallel processing...\n")

    failed  = {}
    bar     = ProgressBar(len(todo))

    # Build args list — pass MIRI grid as argument
    # because worker processes cannot share global state
    args_list = [
        (fpath,
         MIRI_WN_GRID,
         MIRI_N_PTS,
         MIRI_WN_MIN,
         MIRI_WN_MAX)
        for fpath in todo]

    # ProcessPoolExecutor: each worker is a separate
    # Python process — true parallelism, no GIL.
    # max_workers controls how many processes run at once.
    with ProcessPoolExecutor(
            max_workers=n_workers) as executor:

        # Submit all jobs at once — executor manages
        # the queue internally
        futures = {
            executor.submit(process_one_file, args): args[0]
            for args in args_list}

        batch_results = {}

        for future in as_completed(futures):
            fpath = futures[future]
            try:
                fname, entry, status = future.result()
            except Exception as e:
                fname  = os.path.basename(fpath)
                entry  = None
                status = f"exception:{e}"

            if entry is not None:
                batch_results[fname] = entry
                bar.update(success=True)
            else:
                failed[fname] = status
                bar.update(success=False)

            # Save checkpoint every batch_size files
            if len(batch_results) >= batch_size:
                database.update(batch_results)
                with open(DB_PATH, 'w') as f:
                    json.dump(database, f, indent=2)
                batch_results = {}

        # Save remaining
        if batch_results:
            database.update(batch_results)
            with open(DB_PATH, 'w') as f:
                json.dump(database, f, indent=2)

    bar.finish()

    # Save failure log
    with open(FAIL_LOG, 'w') as f:
        f.write("filename,reason\n")
        for fname, reason in sorted(failed.items()):
            f.write(f"{fname},{reason}\n")

    # Final summary
    print(f"\n{'='*62}")
    print(f"  SUMMARY")
    print(f"{'='*62}")
    print(f"  Successfully processed : {len(database)}")
    print(f"  Failed                 : {len(failed)}")
    print(f"  Database               : {DB_PATH}")

    if failed:
        print(f"\n  Failure breakdown:")
        from collections import Counter
        reasons = Counter(
            r.split(':')[0] for r in failed.values())
        for r, c in reasons.most_common():
            print(f"    {r:<30} {c}")

    # Statistics
    ok_entries = [e for e in database.values()
                  if e.get('fit_ok')]
    if ok_entries:
        rms_vals  = [e['fit_rms'] for e in ok_entries]
        npk_vals  = [e['n_peaks'] for e in ok_entries]
        nmi_vals  = [e['n_miri_peaks']
                     for e in ok_entries]
        print(f"\n  Fit statistics (successful fits):")
        print(f"    Mean RMS        : "
              f"{np.mean(rms_vals):.5f}")
        print(f"    Median RMS      : "
              f"{np.median(rms_vals):.5f}")
        print(f"    Mean peaks/file : "
              f"{np.mean(npk_vals):.1f}")
        print(f"    Mean MIRI peaks : "
              f"{np.mean(nmi_vals):.1f}")

        from collections import Counter
        all_sp = []
        for e in database.values():
            all_sp.extend(e.get('species', []))
        sp_counts = Counter(all_sp)
        print(f"\n  Top species in database:")
        for sp, cnt in sp_counts.most_common(15):
            print(f"    {sp:<15} {cnt:>4} files")

    print(f"{'='*62}")


if __name__ == "__main__":
    # Required on Windows for ProcessPoolExecutor
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description='Parallel Gaussian decomposition '
                    'of LIDA ice spectra')
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of parallel worker processes '
             '(default: CPU count - 1)')
    parser.add_argument(
        '--batch-size', type=int, default=50,
        help='Save checkpoint every N files '
             '(default: 50)')
    args = parser.parse_args()

    main(n_workers=args.workers,
         batch_size=args.batch_size)