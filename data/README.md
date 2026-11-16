# Data

## In this directory

### `targets/` — observed spectra

| File | Source | Description |
|---|---|---|
| `iras2A_global_opticaldepth.dat` | NGC 1333 IRAS 2A | Silicate-removed optical depth — **Stage 2 (NNLS) input** |
| `iras2A_tau_with_silicate_reconstructed.dat` | NGC 1333 IRAS 2A | Silicate added back via Drude profile — **Stage 1 (ML) input** |
| `IRS2B_C0_spectra.txt` | L1448 IRS2 | Raw JWST/MIRI flux spectrum |
| `IRS3B_B_C0_spectra.txt` | L1448 IRS3B | Raw JWST/MIRI flux spectrum |

The two IRAS 2A files are the same observation in two forms. The classifiers
were trained on spectra that include silicate absorption, so they must see
silicate at inference; the laboratory reference spectra contain no silicate, so
the fitting target must have it removed. The reconstructed file exists because
the published IRAS 2A data was distributed with silicate already subtracted.

The L1448 files are raw flux and are converted to optical depth by
`src/04_sources/l1448_v38_part1.py`. Processed τ spectra for both are in
`results/l1448_irs2_v39/` and `results/l1448_irs3b_v39/`.

### `labels/` — cluster definitions

| File | Description |
|---|---|
| `indra_labels_v38_135k.csv` | The 284 spectral cluster labels used as classification targets |
| `indra_cluster_map_v38_135k.csv` | Maps each of the 1209 raw library files to its cluster |

## Not in this repository

| Asset | Size | Why excluded |
|---|---|---|
| LIDA laboratory library (`lida_raw_data/`) | ~700 MB, 1209 files | Third-party data, too large for git |
| `indra_X_v38_135k_*.npy` | 1.6 GB | Synthetic training spectra |
| `indra_Y_v38_135k_*.npy` | 294 MB | Synthetic training labels |
| `gaussian_peaks_database.json` | 65 MB | Derived artifact, regenerable |

### LIDA laboratory library

The Leiden Ice Database provides the reference spectra that form the dictionary
matrix **A**. Download from <https://icedb.strw.leidenuniv.nl>.

The working library contains 1209 files: the LIDA base set plus roughly 40 added
from other sources (NASA Ames, NASA OCdb, and files contributed by collaborators
covering formate, grain-shape-corrected H₂O, H₂O:CH₃COOH, ternary H₂O:NH₃:H₂S
mixtures, thermally processed organic residues and CO₂ bending-region spectra).

Loading requirements, all handled by the loaders in `src/`:

- Column 1 may be wavenumber (cm⁻¹) or wavelength (µm). Detect by checking
  whether the column mean exceeds 100; convert with λ = 10⁴/ν.
- Read with `encoding='latin-1'`. About 40 files carry a Windows-1252 en-dash
  in their header comment, which makes `np.loadtxt` fail under UTF-8.
- Interpolate onto `np.linspace(5.0, 28.0, 3000)` and normalise each spectrum
  to peak = 1. **The normalisation is not optional** — without it, species with
  intrinsically strong absorption dominate every synthetic mixture regardless
  of their assigned mixing fraction.
- Partial-range files are zero-padded outside their measured interval.

A curated, baseline-corrected subset of 535 files also exists and is planned as
the primary fitting basis in v3.11.

### Synthetic training arrays

Regenerate with:

```bash
python src/01_generation/indra_generator_v38.py
```

Takes roughly 6 minutes and produces 135,000 spectra (120,000 general plus
15,000 targeted at weak species).

- `X`: shape `(135000, 3000)`, optical depth, range 0–6.215
- `Y`: shape `(135000, 569)` — column 0 is total τ, columns 1–284 are mixing
  fractions, columns 285–568 are temperatures

> **Critical:** `X` is saved **already converted to optical depth**. Do not
> apply `-log(clip(X, ...))` when loading it. An earlier version of the training
> script did exactly that, inverting the signal — strong absorption became
> near-zero and continuum became 6.2 — while prediction fed correctly oriented
> data. Load with `X_tau = np.maximum(X_raw, 0.0)` and nothing else. Fixing this
> one line raised PR-AUC by nearly 10 percentage points.

### Gaussian peak database

Regenerate with:

```bash
python src/05_gaussian/gaussian_decomposer.py
```

Roughly 16 minutes using 7 worker processes. Decomposes each library spectrum
into asymmetric Gaussian components: 1198 of 1209 files succeed, averaging 7.5
peaks per file. The 12 failures are logged in `results/gaussian/gaussian_failed.txt`.

Mean fit RMS is currently 0.381, which is too high to use for retraining — the
ALS baseline under-performs on high-absorbance files (τ > 3) because its
smoothing parameter λ is not yet adaptive. Fixing this is roadmap item 2.
