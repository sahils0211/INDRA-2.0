# Models

## In this directory

| File | Size | Contents |
|---|---|---|
| `lgbm_species_names_v38patch.pkl` | 8 KB | Ordered list of the 286 classifier labels |
| `lgbm_thresholds_recall90_v38patch.pkl` | 6 KB | Per-species decision thresholds, tuned to a 90% recall target |

## Not in this repository

| File | Size | Why excluded |
|---|---|---|
| `lgbm_classifiers_v38patch.pkl` | **463 MB** | Exceeds GitHub's 100 MB per-file limit |

This file holds all 286 trained LightGBM binary classifiers — 284 from the main
spectral clustering plus 2 OCN⁻ patch classifiers (PR-AUC 94.6% and 98.7%).

### Obtaining it

**Option 1 — copy the existing bundle.** It lives in the project's working
directory alongside the other `lgbm_*_v38patch.pkl` files. This is the fastest
route and gives bit-identical results to everything reported in
`docs/INDRA_Technical_Report.pdf`.

**Option 2 — retrain from scratch.** Requires the synthetic training arrays
(see `data/README.md`), then:

```bash
python src/02_training/indra_classifier_v38.py
```

Approximately 8–10 hours on a Ryzen 7 with 16 GB RAM. The script uses
LightGBM's internal parallelism with a single copy of the training matrix in
memory — do **not** wrap the outer loop in `joblib.Parallel`, which spawns one
1.3 GB copy per worker and overflows RAM.

**Option 3 — Git LFS.** If you want the bundle version-controlled:

```bash
git lfs install
git lfs track "models/lgbm_classifiers_*.pkl"
git add .gitattributes
```

Note that 463 MB will consume most of a free LFS quota.

## Loading

```python
import joblib

classifiers = joblib.load("models/lgbm_classifiers_v38patch.pkl")
names       = joblib.load("models/lgbm_species_names_v38patch.pkl")
thresholds  = joblib.load("models/lgbm_thresholds_recall90_v38patch.pkl")
```

## Important

These classifiers do **not** need retraining to run v3.9. The data leakage
corrected in v3.9 was entirely downstream of Stage 1 — it lived in the NNLS
selection and elimination logic, not in how the classifiers were trained. The
v3.8patch classifier bundle is the current deployed model.

Input must be optical depth **with silicate included**, shape `(n, 3000)`,
on the grid `np.linspace(5.0, 28.0, 3000)`. Feeding silicate-removed spectra
to the classifiers is a domain mismatch and degrades results substantially.
