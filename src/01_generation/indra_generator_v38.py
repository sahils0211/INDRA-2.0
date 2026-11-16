# ============================================================
# INDRA Generator v3.8 — Google Colab
# Changes from v3.7:
#   - CRITICAL_NEW expanded: SO2, H2CO, CH3OCHO,
#     CH3CH2OH, OCN- (HNCO proxy) added
#   - HARD_SPECIES_KW: these species get alpha=5.0
#     so features always clear noise floor
#   - Co-occurrence probability 0.25 → 0.40
#   - N_SPECTRA stays 135K (volume is not the problem)
#   - .dat1 and .csv extensions retained
# ============================================================


import subprocess
subprocess.run(['pip', 'install', '-q', 'scikit-learn'],
               check=True)
print("Dependencies ready.")

# ── CELL 2: Imports ───────────────────────────────────────
import numpy as np
import pandas as pd
import os, re, gc, shutil
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_distances
from datetime import datetime
from joblib import Parallel, delayed
print("Imports done.")

# ── CELL 3: Paths and constants ───────────────────────────
LIDA_DIR   = "c:/Users/sahil/Downloads/lida_raw_data/"
OUTPUT_DIR = "c:/Users/sahil/Downloads/"
LOCAL_DIR  = "c:/Users/sahil/Downloads/indra_tmp/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOCAL_DIR,  exist_ok=True)

N_GENERAL  = 120000
N_TARGETED = 15000
N_SPECTRA  = N_GENERAL + N_TARGETED   # 135K — unchanged

MIRI_GRID  = np.linspace(5.0, 28.0, 3000)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TAU_MIN         = 0.1
TAU_MAX         = 5.0
TEMP_NORM       = 150.0
N_LAYERS_PEAK   = 15
N_LAYERS_MIN    = 10
N_LAYERS_MAX    = 25
SNR_MIN         = 200
SNR_MAX         = 1000
NOISE_SCALE_MIN = 1.0
NOISE_SCALE_MAX = 2.2
FRACTION_THRESH = 0.01
FLUX_CLIP_LOW   = 0.002

COSINE_MERGE_THRESHOLD = 0.05
MAX_TEMP_CLASSES       = 4

SILICATE_PROB = 0.80
SIL_TAU_MIN   = 0.3
SIL_TAU_MAX   = 6.0

CHUNK_SIZE = 5000

MANDATORY = ['H2O', 'CO2', 'CO']

# v3.8: expanded — hard-to-detect species added
CRITICAL_NEW = [
    # v3.7 originals
    'H2O_CH3COOH', 'Formate', 'h2o-nh3-h2s',
    'H2O_NH3_H2S', 'H2O160K', 'H2O_75K',
    'H2O_CH4', 'CO_CH4', 'CO2_CH4',
    'Spec_D', 'OrganicResidue',
    # v3.8 additions — previously missed species
    'SO2', 'CH3OH_SO2',
    'CH3CH2OH', 'C2H5OH',
    'H2CO', 'CH2O',
    'CH3OCHO',
    'OCN',          # proxy for HNCO:NH3 product
]

# v3.8: these species get alpha=5.0 so their bands
# are always visible above the noise floor
HARD_SPECIES_KW = [
    'SO2', 'H2CO', 'CH2O', 'CH3OCHO',
    'CH3CH2OH', 'C2H5OH', 'OCN',
    'CH3COOH', 'HCOOH', 'CH3CHO',
]

print(f"LIDA dir     : {LIDA_DIR}")
print(f"Output dir   : {OUTPUT_DIR}")
print(f"N general    : {N_GENERAL:,}")
print(f"N targeted   : {N_TARGETED:,}")
print(f"N total      : {N_SPECTRA:,}  (unchanged from v3.7)")
print(f"Layers peak  : {N_LAYERS_PEAK}")
print(f"Critical new : {len(CRITICAL_NEW)} keywords")
print(f"Hard species : {len(HARD_SPECIES_KW)} keywords")

# ── CELL 4: Load LIDA files ───────────────────────────────
def load_spectrum_file(path):
    try:
        with open(path, 'rb') as f:
            raw = f.read(10)
        if raw[:2] in [b'\xff\xfe', b'\xfe\xff']:
            return None, None
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
                rows.append((float(parts[0]),
                             float(parts[1])))
            except ValueError:
                continue
        if len(rows) < 50:
            return None, None
        data = np.array(rows)
        wl, kap = data[:, 0], data[:, 1]
        if wl.mean() > 100:
            wl  = 1e4 / wl
            wl  = wl[::-1]
            kap = kap[::-1]
        valid = ((wl >= 5.0) & (wl <= 28.0) &
                 np.isfinite(kap) & (kap >= 0))
        if valid.sum() < 50:
            return None, None
        f = interp1d(wl[valid], kap[valid],
                     bounds_error=False, fill_value=0.0)
        kap_grid = f(MIRI_GRID)
        kmax = kap_grid.max()
        if kmax <= 0:
            return None, None
        return MIRI_GRID, kap_grid / kmax
    except Exception:
        return None, None

print("\nLoading LIDA files...")
VALID_EXTENSIONS = {'.dat', '.txt', '.dat1'}
files = sorted([
    f for f in os.listdir(LIDA_DIR)
    if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
])

raw_data  = {}
raw_temps = {}
n_loaded = n_skipped = 0
for fname in files:
    name = os.path.splitext(fname)[0]
    path = os.path.join(LIDA_DIR, fname)
    _, kap = load_spectrum_file(path)
    if kap is not None:
        raw_data[name]  = kap
        m = re.search(r'_(\d+)[Kk]', fname)
        raw_temps[name] = float(m.group(1)) if m else 30.0
        n_loaded += 1
    else:
        n_skipped += 1

print(f"  Loaded   : {n_loaded}")
print(f"  Skipped  : {n_skipped}")

# Confirm critical new species are present
new_keywords = ['CH3COOH', 'Formate', 'nh3-h2s',
                'H2O160', 'H2O_75K', 'CH4', 'Spec_D',
                'h2o-nh3', 'SO2', 'H2CO', 'CH3OCHO',
                'CH3CH2OH', 'C2H5OH', 'OCN']
found_new = [n for n in raw_data
             if any(kw.lower() in n.lower()
                    for kw in new_keywords)]
print(f"\n  Critical files detected: {len(found_new)}")
for n in sorted(found_new)[:20]:
    print(f"    ✔ {n}")
if len(found_new) > 20:
    print(f"    ... and {len(found_new)-20} more")

if len(raw_data) == 0:
    raise RuntimeError("No files loaded. Check LIDA_DIR.")

# ── CELL 5: Spectral clustering ───────────────────────────
def extract_base_name(full_name):
    return re.sub(r'_\d+[Kk]$', '', full_name)

def cluster_temperature_variants(names_subset,
                                  spectra_subset,
                                  max_classes=MAX_TEMP_CLASSES):
    n = len(names_subset)
    if n == 1:
        lbl = names_subset[0] + '_classA'
        return {lbl: spectra_subset[0]}, \
               {lbl: names_subset}
    if n == 2:
        dist = cosine_distances(
            spectra_subset[0:1], spectra_subset[1:2])[0, 0]
        if dist < COSINE_MERGE_THRESHOLD:
            lbl = names_subset[0] + '_classA'
            return ({lbl: spectra_subset.mean(axis=0)},
                    {lbl: names_subset})
        else:
            r = {names_subset[0]+'_classA': spectra_subset[0],
                 names_subset[1]+'_classB': spectra_subset[1]}
            m = {names_subset[0]+'_classA': [names_subset[0]],
                 names_subset[1]+'_classB': [names_subset[1]]}
            return r, m
    dist_matrix    = cosine_distances(spectra_subset)
    dist_condensed = squareform(np.clip(dist_matrix, 0, None))
    link_mat       = linkage(dist_condensed, method='complete')
    best_n      = 1
    best_labels = np.zeros(n, dtype=int)
    for n_c in range(2, min(max_classes + 1, n + 1)):
        lbls    = fcluster(link_mat, t=n_c,
                           criterion='maxclust') - 1
        centers = np.array([
            spectra_subset[lbls == k].mean(axis=0)
            for k in range(n_c)])
        if len(centers) < 2:
            break
        inter = cosine_distances(centers).copy()
        np.fill_diagonal(inter, np.inf)
        if inter.min() > COSINE_MERGE_THRESHOLD:
            best_n      = n_c
            best_labels = lbls
        else:
            break
    letters  = 'ABCDE'
    reps, memb = {}, {}
    for k in range(best_n):
        mask    = best_labels == k
        center  = spectra_subset[mask].mean(axis=0)
        members = [names_subset[i]
                   for i in range(n) if mask[i]]
        cname   = members[0] + f'_class{letters[k]}'
        reps[cname]  = center
        memb[cname]  = members
    return reps, memb

print("\nRunning spectral clustering...")
base_groups = {}
for name in raw_data:
    base = extract_base_name(name)
    base_groups.setdefault(base, []).append(name)

species_data = {}
cluster_map  = {}
for base, members in base_groups.items():
    spectra_arr = np.array([raw_data[n] for n in members])
    reps, memb  = cluster_temperature_variants(
        members, spectra_arr)
    species_data.update(reps)
    cluster_map.update(memb)

# Cross-mixture merge
rep_names   = list(species_data.keys())
rep_spectra = np.array([species_data[n] for n in rep_names])
rep_dist    = cosine_distances(rep_spectra)
cross_link  = linkage(
    squareform(np.clip(rep_dist, 0, None)),
    method='complete')
cross_labels = fcluster(
    cross_link, t=COSINE_MERGE_THRESHOLD,
    criterion='distance') - 1

final_data    = {}
final_map     = {}
used_clusters = set()
for i, name in enumerate(rep_names):
    cid = cross_labels[i]
    if cid in used_clusters:
        for el in final_map:
            flat = [m for sub in final_map[el]
                    for m in (sub if isinstance(sub, list)
                              else [sub])]
            if cluster_map[name][0] in flat:
                final_map[el].extend(cluster_map[name])
                break
        continue
    used_clusters.add(cid)
    final_data[name] = species_data[name]
    final_map[name]  = cluster_map[name]

species_data  = final_data
cluster_map   = final_map
species_names = list(species_data.keys())
n_species     = len(species_names)

print(f"  Raw entries    : {len(raw_data)}")
print(f"  Cluster labels : {n_species}")
print(f"  Reduction      : "
      f"{100*(1-n_species/max(len(raw_data),1)):.0f}%")

cluster_map_rows = []
for lbl, members in cluster_map.items():
    for m in members:
        cluster_map_rows.append(
            {'cluster_label': lbl, 'raw_member': m})
cluster_map_df = pd.DataFrame(cluster_map_rows)

# Mandatory indices
mandatory_idx = {}
for mol in MANDATORY:
    idxs = [i for i, n in enumerate(species_names)
            if mol.upper() in n.upper()]
    mandatory_idx[mol] = idxs
    print(f"  Mandatory {mol:5s}: {len(idxs)} variants")

# Critical new indices
critical_idx = []
for i, name in enumerate(species_names):
    name_up = name.upper()
    if any(kw.upper() in name_up for kw in CRITICAL_NEW):
        critical_idx.append(i)
print(f"  Critical new   : {len(critical_idx)} labels")

# Hard species indices (subset of critical)
hard_idx_set = set()
for i, name in enumerate(species_names):
    name_up = name.upper()
    if any(kw.upper() in name_up for kw in HARD_SPECIES_KW):
        hard_idx_set.add(i)
print(f"  Hard species   : {len(hard_idx_set)} labels")
for i in sorted(hard_idx_set):
    print(f"    ★ {species_names[i]}")

# ── CELL 6: Silicate generator ────────────────────────────
def drude(wl, peak, width, amp):
    x = wl / peak
    p = amp * (x * width / peak)**2 / \
        ((x**2 - 1.0)**2 + (x * width / peak)**2)
    return np.maximum(p, 0.0)

print("Silicate generator ready.")

# ── CELL 7: Single spectrum generator ─────────────────────
def generate_one_spectrum(seed, mode='general',
                           forced_critical_idx=None):
    rng = np.random.default_rng(seed)

    n_layers = int(np.clip(
        int(np.round(rng.normal(N_LAYERS_PEAK, 2.5))),
        N_LAYERS_MIN, N_LAYERS_MAX))

    chosen_idx = []

    # Mandatory species
    for mol in MANDATORY:
        if mandatory_idx[mol]:
            chosen_idx.append(
                int(rng.choice(mandatory_idx[mol])))

    # Targeted: force one critical species
    if (mode == 'targeted' and
            forced_critical_idx is not None):
        if forced_critical_idx not in chosen_idx:
            chosen_idx.append(forced_critical_idx)

    # v3.8: 40% chance of guaranteed critical
    # co-occurrence (was 25%)
    elif mode == 'general' and rng.random() < 0.40:
        if critical_idx:
            crit = int(rng.choice(critical_idx))
            if crit not in chosen_idx:
                chosen_idx.append(crit)

    # Fill remaining layers
    remaining = n_layers - len(chosen_idx)
    if remaining > 0:
        pool  = [i for i in range(n_species)
                 if i not in chosen_idx]
        extra = rng.choice(
            pool,
            size=min(remaining, len(pool)),
            replace=False)
        chosen_idx.extend(extra.tolist())

    n_chosen = len(chosen_idx)

    # Dirichlet alphas
    alphas = np.ones(n_chosen) * 0.8
    for fi, idx in enumerate(chosen_idx):
        name_up = species_names[idx].upper()
        if 'H2O' in name_up and 'CH' not in name_up:
            alphas[fi] = 10.0
        elif any(m in name_up for m in
                 ['CO2', 'CO_CO2', 'PURECO2', 'PURECO_']):
            alphas[fi] = 5.0
        elif (mode == 'targeted' and
              idx == forced_critical_idx):
            alphas[fi] = 8.0
        elif mode == 'general' and idx in critical_idx:
            alphas[fi] = 4.0
        # v3.8: hard species always get alpha=5
        # so their bands clear the noise floor
        if idx in hard_idx_set:
            alphas[fi] = max(alphas[fi], 5.0)

    fractions = rng.dirichlet(alphas)

    # Ice tau spectrum
    tau_ice_total = float(rng.uniform(TAU_MIN, TAU_MAX))
    tau_spec      = np.zeros(len(MIRI_GRID),
                             dtype=np.float64)
    temps         = []
    for fi, idx in enumerate(chosen_idx):
        kappa = species_data[species_names[idx]]
        T = float(np.clip(
            rng.uniform(10, 50)
            if rng.random() < 0.7
            else rng.uniform(50, 150),
            10.0, 150.0))
        temps.append(T)
        tau_spec += (fractions[fi] *
                     tau_ice_total * kappa)

    # Silicate
    sil_present = rng.random() < SILICATE_PROB
    if sil_present:
        gsf   = float(rng.uniform(0.5, 3.0))
        cfrac = float(rng.uniform(0.0, 0.35))
        ofrac = float(rng.uniform(0.3, 0.85))
        pfrac = 1.0 - ofrac
        p97   = float(np.clip(9.7  + 0.35*(gsf-1),
                               9.4, 10.5))
        w97   = float(np.clip(1.0  + 0.40*(gsf-1),
                               0.8,  2.0))
        p18   = float(np.clip(18.0 + 0.50*(gsf-1),
                               17.5, 19.5))
        w18   = float(np.clip(4.0  + 0.80*(gsf-1),
                               3.0,  6.0))
        wl    = MIRI_GRID
        ksil  = (
            (1-cfrac)*(drude(wl, p97, w97, 1.0) +
                       drude(wl, p18, w18, 0.4)) +
            cfrac*ofrac*(drude(wl, 10.0, 0.20, 0.6) +
                         drude(wl, 11.2, 0.25, 0.8) +
                         drude(wl, 16.4, 0.30, 0.3)) +
            cfrac*pfrac*(drude(wl,  9.3, 0.20, 0.5) +
                         drude(wl, 10.5, 0.22, 0.7) +
                         drude(wl, 11.4, 0.25, 0.4))
        )
        km = ksil.max()
        if km > 0:
            ksil /= km
        tau_spec += float(rng.uniform(
            SIL_TAU_MIN, SIL_TAU_MAX)) * ksil

    # Flux + noise
    flux_clean = np.exp(-tau_spec)
    snr        = float(rng.uniform(SNR_MIN, SNR_MAX))
    nscale     = float(rng.uniform(
        NOISE_SCALE_MIN, NOISE_SCALE_MAX))
    noise      = ((nscale / snr) *
                  rng.standard_normal(len(MIRI_GRID)))
    flux_noisy = np.clip(
        flux_clean + noise, FLUX_CLIP_LOW, 1.05)
    flux_noisy = medfilt(
        flux_noisy.astype(np.float64), kernel_size=5)
    tau_out    = -np.log(
        np.clip(flux_noisy, FLUX_CLIP_LOW, 1.05))
    tau_out    = np.maximum(
        tau_out, 0.0).astype(np.float32)

    # Y vectors
    y_fractions = np.zeros(n_species, dtype=np.float32)
    y_temps     = np.zeros(n_species, dtype=np.float32)
    for fi, idx in enumerate(chosen_idx):
        y_fractions[idx] = fractions[fi]
        y_temps[idx]     = temps[fi] / TEMP_NORM

    return (tau_out, y_fractions,
            tau_ice_total, sil_present, mode)

print("Generator ready.")

# ── CELL 8: Generate dataset ──────────────────────────────
ts      = datetime.now().strftime("%Y%m%d_%H%M")
tag     = f"v38_{N_SPECTRA//1000}k_{ts}"
x_local = os.path.join(LOCAL_DIR, f"indra_X_{tag}.npy")
y_local = os.path.join(LOCAL_DIR, f"indra_Y_{tag}.npy")
x_path  = os.path.join(OUTPUT_DIR, f"indra_X_{tag}.npy")
y_path  = os.path.join(OUTPUT_DIR, f"indra_Y_{tag}.npy")
lb_path = os.path.join(OUTPUT_DIR,
                        f"indra_labels_{tag}.csv")
cm_path = os.path.join(OUTPUT_DIR,
                        f"indra_cluster_map_{tag}.csv")

# Y layout: col0=tau_total | cols 1..n_species=fractions
# | cols n_species+1..2*n_species=temperatures
X_mmap = np.lib.format.open_memmap(
    x_local, mode='w+', dtype=np.float32,
    shape=(N_SPECTRA, len(MIRI_GRID)))
Y_mmap = np.lib.format.open_memmap(
    y_local, mode='w+', dtype=np.float32,
    shape=(N_SPECTRA, 1 + 2 * n_species))

print(f"\nGenerating {N_SPECTRA:,} spectra (v3.8)...")
print(f"  {N_GENERAL:,} general + {N_TARGETED:,} targeted")
print(f"  X : {X_mmap.shape}   Y : {Y_mmap.shape}")
print(f"  Tag: {tag}\n")

mode_list = (['general']  * N_GENERAL +
             ['targeted'] * N_TARGETED)

# Distribute targeted spectra across critical species
if critical_idx:
    n_per = N_TARGETED // max(len(critical_idx), 1)
    forced_list = []
    for ci in critical_idx:
        forced_list.extend([ci] * n_per)
    rng_fill = np.random.default_rng(99)
    while len(forced_list) < N_TARGETED:
        forced_list.append(
            int(rng_fill.choice(critical_idx)))
else:
    forced_list = [None] * N_TARGETED

n_chunks  = (N_SPECTRA + CHUNK_SIZE - 1) // CHUNK_SIZE
sil_total = 0
gen_count = {'general': 0, 'targeted': 0}
t_start   = datetime.now()

for chunk_idx in range(n_chunks):
    i_start = chunk_idx * CHUNK_SIZE
    i_end   = min(i_start + CHUNK_SIZE, N_SPECTRA)

    chunk_args = []
    for i in range(i_start, i_end):
        mode   = mode_list[i]
        forced = (forced_list[i - N_GENERAL]
                  if mode == 'targeted' else None)
        chunk_args.append((i, mode, forced))

    results = Parallel(n_jobs=2, backend='loky')(
        delayed(generate_one_spectrum)(s, m, f)
        for s, m, f in chunk_args)

    for local_i, (tau_out, y_frac,
                  tau_tot, sil_flag,
                  mode) in enumerate(results):
        g = i_start + local_i
        X_mmap[g]                = tau_out
        Y_mmap[g, 0]             = tau_tot
        Y_mmap[g, 1:n_species+1] = y_frac
        # temperatures not used in v3.8 training
        # but kept for Y layout compatibility
        if sil_flag:
            sil_total += 1
        gen_count[mode] += 1

    X_mmap.flush()
    Y_mmap.flush()

    elapsed = (datetime.now()-t_start).total_seconds()/60
    pct     = 100 * i_end / N_SPECTRA
    active  = (Y_mmap[i_start:i_end, 1:n_species+1]
               > FRACTION_THRESH).sum(axis=1).mean()
    print(f"  Chunk {chunk_idx+1:>3}/{n_chunks}  "
          f"[{i_end:>6}/{N_SPECTRA}]  "
          f"{pct:5.1f}%  active:{active:.1f}  "
          f"τ_max:{X_mmap[:i_end].max():.2f}  "
          f"{elapsed:.1f}min")
    del results
    gc.collect()

del X_mmap, Y_mmap
gc.collect()

total_min = (datetime.now()-t_start).total_seconds()/60
print(f"\nGeneration complete in {total_min:.1f} min")

# ── CELL 9: Copy to Drive ─────────────────────────────────
print(f"\nCopying to Drive...")
import shutil
shutil.copy2(x_local, x_path)
print(f"  ✔ X → {x_path}")
shutil.copy2(y_local, y_path)
print(f"  ✔ Y → {y_path}")
pd.DataFrame({'label': species_names}).to_csv(
    lb_path, index=False)
cluster_map_df.to_csv(cm_path, index=False)
print(f"  ✔ Labels  → {lb_path}  ({n_species} rows)")
print(f"  ✔ Cluster → {cm_path}")
os.remove(x_local)
os.remove(y_local)

# ── CELL 10: Sanity checks ────────────────────────────────
print(f"\n{'='*55}")
print(f"  v3.8 SANITY CHECKS")
print(f"{'='*55}")
X_chk  = np.load(x_path,  mmap_mode='r')
Y_chk  = np.load(y_path,  mmap_mode='r')
Yb_chk = (Y_chk[:, 1:n_species+1]
           > FRACTION_THRESH).astype(np.int8)

print(f"  Total spectra   : {N_SPECTRA:,}")
print(f"  Label space     : {n_species}")
print(f"  τ range         : "
      f"{X_chk.min():.3f}–{X_chk.max():.3f}")
print(f"  Avg active/spec : "
      f"{Yb_chk.sum(axis=1).mean():.1f}")
print(f"  Silicate inj.   : "
      f"{sil_total/N_SPECTRA*100:.1f}%")

print(f"\n  Mandatory presence rates:")
for mol in MANDATORY:
    idxs = mandatory_idx[mol]
    if idxs:
        rate = Yb_chk[:, idxs].any(axis=1).mean()*100
        ok   = "✔" if rate > 95 else "⚠"
        print(f"    {ok} {mol:8s}: {rate:.1f}%")

print(f"\n  Hard species presence rates (target >5%):")
for idx in sorted(hard_idx_set):
    name = species_names[idx]
    rate = Yb_chk[:, idx].mean() * 100
    ok   = "✔" if rate > 5.0 else "⚠ low"
    print(f"    {ok} {name[:48]:<48}: {rate:.1f}%")

del X_chk, Y_chk, Yb_chk
gc.collect()

print(f"\n  Pass to LightGBM script:")
print(f"    X_FILE       = '{x_path}'")
print(f"    Y_FILE       = '{y_path}'")
print(f"    LABEL_FILE   = '{lb_path}'")
print(f"    CLUSTER_FILE = '{cm_path}'")
print(f"\nGenerator v3.8 complete.")