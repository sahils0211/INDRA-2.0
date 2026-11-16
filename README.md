# INDRA 2.0

**Ice component identificatioN using Deep leaRning on Astrophysics spectra**

A machine learning pipeline that identifies and quantifies the ice mixtures frozen onto dust grains around newly forming stars, using infrared spectra from the James Webb Space Telescope.

School of Physical Sciences, NISER Bhubaneswar · In collaboration with Prathap Rayalacheruvu and Prof. Liton Majumdar (SEPS, NISER)

---

## What problem does this solve?

You do not need any background in astronomy or chemistry to understand the task.

When a star forms, molecules in the surrounding cloud freeze onto microscopic dust grains and build up layers of mixed ice — water, carbon dioxide, methanol, ammonia and dozens of others. These ices are the raw material that later becomes planets and comets, so knowing what they are made of matters.

JWST measures how much light a young star emits at each wavelength between 5 and 28 µm. Each molecule absorbs at its own characteristic wavelengths, so the recorded spectrum looks like a smooth hill with dips carved into it, each dip a molecular fingerprint. The difficulty is that **the dips overlap heavily**, and a broad absorption from ordinary rocky dust sits underneath everything and must be removed first.

Stripped of the astronomy, this is a standard applied-maths problem:

> Given an observed signal $\mathbf{b} \in \mathbb{R}^{3000}$ (absorption sampled on a fixed wavelength grid) and a dictionary $\mathbf{A} \in \mathbb{R}^{3000 \times 1209}$ whose columns are laboratory reference spectra, find coefficients $\mathbf{x}$ such that
>
> $$\mathbf{A}\mathbf{x} \approx \mathbf{b}, \qquad \mathbf{x} \ge 0, \qquad \|\mathbf{x}\|_0 \text{ small}$$

Non-negativity is physical — a mixture cannot contain a negative amount of something. Sparsity is physical too — a real ice mantle has 10–25 distinguishable components, not 1209.

Three things make it hard:

- **The dictionary is highly correlated.** The same ice measured at 20 K and 30 K gives two columns that differ by less than the noise, which destabilises ordinary least squares.
- **It is effectively underdetermined.** Many different coefficient vectors reproduce $\mathbf{b}$ about equally well.
- **Ground truth barely exists.** Exactly one source has a trusted published answer.

**What INDRA 2.0 adds:** the previous approach was manual — an expert decided which laboratory spectra to fit, taking weeks per source and producing results that another astronomer might not reproduce. INDRA replaces that judgement call with a trained classifier.

---

## Architecture

The pipeline has two deliberately separated stages:

```
                    ┌─────────────────────────────────────┐
   JWST spectrum →  │  PREPROCESSING                      │
                    │  flux → optical depth τ             │
                    │  continuum fit (L-BFGS-B)           │
                    │  silicate removal (2-component)     │
                    └──────────────┬──────────────────────┘
                                   │
               ┌───────────────────┴───────────────────┐
               │                                       │
     τ WITH silicate                          τ ICE-ONLY
               │                                       │
               ▼                                       ▼
   ┌───────────────────────┐             ┌───────────────────────────┐
   │  STAGE 1 — DETECTION  │             │  STAGE 2 — QUANTIFICATION │
   │  286 LightGBM         │  candidates │  OMP → NNLS               │
   │  classifiers          ├────────────►│  forced floor             │
   │  "WHICH species?"     │             │  backward elimination     │
   └───────────────────────┘             │  "HOW MUCH of each?"      │
                                         └─────────────┬─────────────┘
                                                       ▼
                                          ice composition + abundances
```

**Why two different spectra?** The classifiers were trained on synthetic spectra that *include* silicate absorption, so at inference they must see silicate present. The laboratory reference spectra contain *no* silicate, so the fitting target must have it removed. Using the wrong spectrum for either stage degrades results substantially.

---

## Repository structure

```
INDRA-2.0/
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/                           Written deliverables
│   ├── INDRA_Technical_Report.pdf      21-page full technical report
│   ├── INDRA_Technical_Report.tex      LaTeX source
│   └── INDRA_2.0_Progress_Review.pptx  16-slide progress deck
│
├── src/                            Pipeline code, in execution order
│   ├── 01_generation/
│   │   └── indra_generator_v38.py          Clustering + 135k synthetic spectra
│   ├── 02_training/
│   │   └── indra_classifier_v38.py         Two-tier LightGBM training
│   ├── 03_pipeline/
│   │   ├── indra_v39_clean_optA_optC.py    ★ Main v3.9 pipeline (leakage-free)
│   │   ├── indra_v39_oblivious.py          Latest v3.9 variant
│   │   └── indra_v39_reference_skeleton.py Clean-room rewrite, documented stubs
│   ├── 04_sources/
│   │   ├── iras2a_v39.py                   Benchmark validation run
│   │   ├── l1448_v38_part1.py              L1448 continuum + silicate fitting
│   │   └── l1448_v38_part2.py              L1448 ML + NNLS
│   └── 05_gaussian/
│       ├── gaussian_decomposer.py          Batch asymmetric-Gaussian fitting
│       └── gaussian_visualizer.py          Interactive inspection tool
│
├── data/
│   ├── targets/                    Observed spectra for the three sources
│   ├── labels/                     284 cluster labels + cluster membership map
│   └── README.md                   How to obtain the large data not in git
│
├── models/
│   ├── lgbm_species_names_v38patch.pkl       Label list
│   ├── lgbm_thresholds_recall90_v38patch.pkl Per-species decision thresholds
│   └── README.md                   How to obtain / regenerate the classifiers
│
└── results/
    ├── iras2a_v39_clean/           Option A vs Option C comparison
    ├── iras2a_v39/                 Benchmark run output
    ├── l1448_irs2_v39/             Blind run — L1448 IRS2
    ├── l1448_irs3b_v39/            Blind run — L1448 IRS3B
    ├── cross_source/               Cross-source comparison figure
    └── gaussian/                   Decomposition failure log
```

---

## Target sources

| Source | Role | Notes |
|---|---|---|
| **NGC 1333 IRAS 2A** | Validation benchmark | Published decomposition available (Rayalacheruvu et al. 2025): 76 laboratory spectra, 15 mixture components, 17 chemical species |
| **L1448 IRS2** | Blind generalisation test | Class 0 protostar, L ≈ 5.2 L☉. No published ground truth |
| **L1448 IRS3B** | Blind generalisation test | Class 0 binary system. No published ground truth |

All three lie in the Perseus molecular cloud at roughly 300 pc. The distinction matters throughout: IRAS 2A is the only source against which accuracy can be measured, and the two L1448 sources are the only real test of generalisation.

---

## ⚠️ Data leakage — the critical finding

**The v3.8 pipeline that reported 15/15 recall was consulting the published answer during prediction.** This is documented openly here because it shaped the entire v3.9 redesign.

Two mechanisms were found:

1. **`TARGET_SPECIES` forcing.** `force_missing_components()` compared the current solution against a hard-coded dictionary of species known to be in IRAS 2A, identified which were missing, then searched the library specifically to fill those gaps. Equivalent to checking the marking scheme mid-examination.

2. **Recall guard in backward elimination.** Before removing a component, the pruning step checked whether removal would reduce recall *measured against the published list*, and protected components specifically to preserve that number. The stopping criterion for an algorithm must never depend on the answer it is meant to produce.

Several parameters were also found to have been set by what worked on the benchmark: `N_CANDIDATES = 50`, `TARGET_SURVIVORS = 15`, and a forcing threshold of 0.15 chosen because it happened to rescue the species IRAS 2A was missing.

### The fix is structural, not cosmetic

In v3.9 the published ground truth (`paper_mixtures`) is confined to a **single reporting function that executes after every algorithmic decision has been made**. It can read the result; it cannot influence it. No function involved in selection, forcing, protection or elimination accepts it as an argument. Reintroducing the leak would require deliberately threading ground truth through signatures that do not currently accept it.

Backward elimination's stopping criterion was also changed from recall to **fit quality** (remove a component if doing so degrades the residual by less than 0.5%). Fit quality is computable on any source, so the algorithm now behaves identically on the benchmark and on sources with no published answer.

---

## Results

### Benchmark — IRAS 2A

| Version | Fit | ML recall | NNLS recall | Survivors | Status |
|---|---|---|---|---|---|
| v3.7 baseline | 92.1% | 27% | 53% | 23 | — |
| v3.8 OMP + floor | 92.3% | 87% | 87% | 33 | — |
| v3.8 backward elim. | 90.6% | 93% | 87% | 15 | — |
| v3.8patch + OCN⁻ | 90.6% | 100% | 100% | 16 | ⚠️ **leaked** |
| **v3.9 Option A** | **91.1%** | — | **15/15** | 21 | ✅ **clean** |
| v3.9 Option C | 91.1% | — | 12/15 | 17 | ✅ clean |

**The honest statement of 15/15:** roughly 13 species are quantitative detections (fitted fraction > 1%, confirmed independently by both OMP and the classifier), and roughly 6 sit at forced floor abundance (< 0.5%) and are detection claims rather than abundance measurements. The defensible claim is therefore **13 species detected quantitatively, plus 6 flagged as probable detections at upper-limit abundance** — reached without using any source-specific information.

### Blind generalisation — L1448

Neither source has a published decomposition, so there is no recall figure. This table *is* the result.

| Property | L1448 IRS2 | L1448 IRS3B |
|---|---|---|
| Type | Class 0 protostar | Class 0 binary |
| τ(6 µm) H₂O | 1.883 | 1.872 |
| τ(15.2 µm) CO₂ | 3.170 | 2.493 |
| τ(20 µm) artifact | 1.410 | 1.379 |
| NNLS fit quality | 73.9% | 73.4% |
| Surviving components | 13 | 13 |

**Detected in both:** H₂O (dominant), CO₂, CO, CH₄, NH₃, NH₄⁺, H₂S, HCOOH, HCOO⁻, OCN⁻, organic residue. The CH₄ detection is the first for these sources.

That both sources return the same species set and the same survivor count is meaningful evidence that the pipeline responds to physically consistent chemistry rather than source-specific noise.

**CH₃OH (methanol) is not detected in either source.** Its C–O stretch at 9.74 µm falls exactly where silicate over-subtraction drives the ice-only optical depth to zero. This is a continuum artifact, not a genuine non-detection, and it is the clearest evidence that the remaining continuum problem has real scientific cost.

---

## Methods used

| Technique | Why |
|---|---|
| Hierarchical clustering (complete linkage, cosine distance 0.05) | Merge spectrally indistinguishable temperature variants: 1209 files → 284 targets |
| Dirichlet mixture sampling | Produces non-negative fractions summing to one — the natural model of a physical mixture |
| Drude oscillator silicate model | Physically motivated profile for mineral dust absorption |
| Synthetic generation (135k spectra) | Supervised learning needs labels; real labelled spectra are almost nonexistent |
| LightGBM binary classification | Handles class imbalance well; binary framing avoids regression instability at 96% sparsity |
| PR-AUC as primary metric | R² is meaningless at high sparsity; F1 depends on an arbitrary threshold |
| Recall-first thresholds (90%) | A missed detection is unrecoverable; a false alarm is removed later by the fitting stage |
| Per-species `scale_pos_weight` | Species prevalence ranges 1%–8%; one global weight cannot serve both |
| Two-tier hyperparameters | Intrinsically weak species need more capacity and higher positive weight |
| Non-Negative Least Squares | Mixture coefficients cannot be negative |
| Tikhonov regularisation (α = 0.15) | Stops the solver concentrating all weight on H₂O and CO₂ |
| Orthogonal Matching Pursuit | Sparse, interpretable solutions; avoids dense simultaneous fitting |
| Cosine deduplication (0.015) | Removes near-collinear columns that destabilise the solver |
| Forced floor via bounded least squares | Rescues confidently detected species the solver zeroes out |
| Backward elimination on fit quality | Principled pruning with a criterion computable on any source |
| Structural isolation of ground truth | Makes the v3.8 leakage impossible to reintroduce accidentally |
| Quantitative / probabilistic split | Separates measured abundances from detection claims in the output |
| Asymmetric Gaussian decomposition | Clean, interpolatable basis vectors replacing noisy raw files |

Full derivations, parameter values and bug history are in [`docs/INDRA_Technical_Report.pdf`](docs/INDRA_Technical_Report.pdf).

---

## Running the pipeline

```bash
pip install -r requirements.txt
```

The large assets are not in this repository (see [`data/README.md`](data/README.md) and [`models/README.md`](models/README.md)). Once they are in place, edit the path constants at the top of each script and run in order:

```bash
# 1. Build the synthetic training set   (Google Colab, ~6 min)
python src/01_generation/indra_generator_v38.py

# 2. Train the classifiers              (local, ~8–10 h on a Ryzen 7)
python src/02_training/indra_classifier_v38.py

# 3. Validate on the benchmark source
python src/04_sources/iras2a_v39.py

# 4. Run blind on a new source
python src/04_sources/l1448_v38_part1.py    # continuum + silicate
python src/04_sources/l1448_v38_part2.py    # ML + NNLS
```

The classifiers do **not** need retraining to use v3.9 — the leakage was entirely downstream of Stage 1.

### Reading the code for the first time

If you are new to this project, read in this order: `01_generation` → `02_training` → `03_pipeline/indra_v39_clean_optA_optC.py`. The third file is where the Stage 1 / Stage 2 split is clearest. `indra_v39_reference_skeleton.py` is a documented clean-room rewrite of the same logic with the data-loading functions left as explicit stubs — useful as a map, not as a runnable script.

---

## Current state

### Working reliably

- LightGBM classifiers (286, including 2 OCN⁻ patch classifiers at PR-AUC 94.6% / 98.7%)
- L-BFGS-B continuum fitting — converges on all three sources
- Two-component silicate model (olivine + pyroxene)
- OMP + backward elimination — clean, source-agnostic, no leakage
- Option A parent forcing — generic 20% threshold, no per-source tuning
- Gaussian peak database — 1198 / 1209 files decomposed
- IRAS 2A validation and both L1448 blind runs complete

### Known problems

| Problem | Detail |
|---|---|
| **Long-wavelength artifact** | τ(20 µm) ≈ 1.4 on both L1448 sources where physics requires ≈ 0. A degree-4 polynomial cannot fit the ice region and the thermal tail simultaneously |
| **Silicate over-subtraction** | Silicate peak τ hits its upper bound of 8.0 on L1448, zeroing the 9.7 µm region and blocking methanol and ammonia detection |
| **Gaussian baseline quality** | ALS fails on high-absorbance files (τ > 3); mean fit RMS 0.381. Needs adaptive λ before the library can be used for retraining |
| **Floor-level detections** | 6 of 15 species on IRAS 2A are at forced floor abundance, not independently quantified. Must be reported as upper limits |
| **CO₂ isotopologue anomaly** | One component takes 14.6% of fitted abundance at 5.1% classifier confidence — likely a band-shape artifact. Flagged by `solver_override_suspect` |

---

## Roadmap

| Priority | Upgrade | Expected impact | Status |
|---|---|---|---|
| 1 — Critical | Silicate continuum with τ = 0 constraint beyond 19 µm | Fixes methanol detection; removes long-wave artifact | Planned |
| 2 — High | Adaptive ALS λ for Gaussian baselines | Enables reliable Gaussian library for retraining | In progress |
| 3 — High | Retrain classifiers on Gaussian-derived synthetic data | Improves SO₂, H₂CO, CH₃CH₂OH detection | Pending #2 |
| 4 — Medium | Noise-weighted NNLS (1/σᵢ weights) | +3–5% fit quality | Planned v3.10 |
| 5 — Medium | Curated baseline-corrected library as primary basis | +3–4% fit quality | Planned v3.11 |
| 6 — Medium | `optool` synthetic silicate profiles | Resolves silicate upper-bound problem | Planned v3.12 |
| 7 — Lower | Run on IRAS 4A, B335 and other Class 0 sources | Generalisation validation beyond Perseus | Pending data |

### Before a science paper

Methanol detection requires fixing the silicate subtraction; weak species need classifiers retrained on Gaussian-derived data; fit quality on new sources must reach ≥ 85% (currently 73–74%); and at least three to four more sources are needed before a generalisation claim is supportable.

---

## File name mapping

Files were renamed for clarity. This maps each repository path back to its original filename:

| Repository path | Original filename |
|---|---|
| `src/01_generation/indra_generator_v38.py` | `Indra/scripts/indra_38_generation.py` |
| `src/02_training/indra_classifier_v38.py` | `Indra/scripts/indra_classifier_v38.py` |
| `src/03_pipeline/indra_v39_clean_optA_optC.py` | `Indra/scripts/indra_v39_clean_optA&C_0805.py` |
| `src/03_pipeline/indra_v39_oblivious.py` | `iras2a_v3.9_obliv_2005.py` |
| `src/04_sources/iras2a_v39.py` | `iras2A_testing_v39.py` |
| `src/04_sources/l1448_v38_part1.py` | `IRS2B_3B_testing_v38.py` |
| `src/04_sources/l1448_v38_part2.py` | `IRS2B_3B_testing_v38_part2.py` |
| `src/05_gaussian/gaussian_decomposer.py` | `gaussian_decomposer_1_1605.py` |
| `data/labels/indra_labels_v38_135k.csv` | `indra_labels_v38_135k_20260404_0525.csv` |
| `data/labels/indra_cluster_map_v38_135k.csv` | `indra_cluster_map_v38_135k_20260404_0525.csv` |

---

## Key references

- Rayalacheruvu P., Majumdar L., Rocha W. R. M., et al. (2025). *ApJS*, **281**, 51 — the original INDRA paper; IRAS 2A ground truth.
- Rocha W. R. M., van Dishoeck E. F., Ressler M. E., et al. (2024). *A&A*, **683**, A124 — JWST MIRI-MRS analysis of IRAS 2A.
- Boogert A. C. A., Pontoppidan K. M., Knez C., et al. (2008). *ApJ*, **678**, 985 — Spitzer c2d ice survey.
- Weingartner J. C. & Draine B. T. (2001). *ApJ*, **548**, 296 — silicate grain properties used in the two-component Drude model.
- Leiden Ice Database (LIDA) — <https://icedb.strw.leidenuniv.nl>
