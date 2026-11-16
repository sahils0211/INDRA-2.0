"""
INDRA 2.0 — v3.9 Two-Stage Prediction Pipeline
================================================
Ice component identificatioN using Deep leaRning on Astrophysics spectra

This is a from-scratch rewrite of the prediction/NNLS stage, replacing the
v3.8 single-stage design (which had confirmed data leakage via TARGET_SPECIES
forcing and recall-guarded backward elimination against known ground truth).

DESIGN PRINCIPLE — NO LEAKAGE BY CONSTRUCTION:
    `paper_mixtures` (literature ground truth, e.g. Rayalacheruvu et al. 2025
    for IRAS 2A) is used ONLY inside `report_against_literature()`, which runs
    after the pipeline has already committed to a final composition. No
    function upstream of that point is allowed to see `paper_mixtures`, and no
    function upstream computes "recall" as a selection or stopping criterion.
    Contrast with the retired v3.8 pipeline, which forced species into the
    solution *because* forcing them increased recall against a known answer —
    that is target leakage even when done with good intentions.

TWO STAGES:
    Stage 1 (classification): per-species LightGBM classifiers score the
        target's silicate-INCLUDED optical depth spectrum (tau_with_silicate).
        This tells us which PARENT SPECIES are plausibly present.
    Stage 2 (quantification): for every parent species that clears Stage 1,
        pull in ALL available temperature variants of that species from LIDA
        (multi-temperature NNLS), fit them against the silicate-REMOVED,
        ice-only optical depth spectrum (tau_ice_only), and let NNLS itself
        decide how much of each temperature variant is present.

This script is meant to run on your local machine against your existing
trained classifiers (lgbm_classifiers_*.pkl) and LIDA library. Paths and a
couple of source-specific loader stubs are marked CONFIGURE / IMPLEMENT below.
"""

from __future__ import annotations

import os
import re
import pickle
import warnings
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import nnls, lsq_linear

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIG — settled decisions (do not re-tune per-source; these are meant to be
# source-agnostic and physically motivated, not calibrated on IRAS 2A)
# =============================================================================

MIRI_WAVELENGTH_GRID = np.linspace(5.0, 28.0, 3000)  # microns

# Species excluded from consideration entirely: bands degenerate with NH4+,
# no confirmed Class 0 detection to date. Revisit if new lab data resolves
# the degeneracy.
EXCLUDED_SPECIES = ["CH3NH2"]

# Species that get default-included consideration regardless of raw ML score
# IF their file naming matches these keywords AND they clear a low sanity
# floor — trimmed down from v3.8's broader list to only species that are
# universally deduped/consistent across the LIDA library (not source-specific
# tuning).
MUST_INCLUDE_KEYWORDS = ["NH4+", "PureCH3OH", "Spec_D", "Spec_A"]

# Stage 1 -> Stage 2 handoff threshold: a parent species must clear this ML
# score to be considered "detected" and have its temperature variants pulled
# into the Stage 2 design matrix at all.
ML_SCORE_FLOOR = 0.05

# Generic force threshold: species scoring above this on Stage 1 are
# considered high-confidence regardless of what NNLS initially does with them
# in Stage 2. This is fixed at a physically-reasonable generic value, NOT
# tuned per source (v3.8 used 0.15, calibrated specifically to rescue IRAS 2A
# misses — that was leakage-adjacent; 0.20 here is the deliberately-generic
# replacement).
FORCE_THRESHOLD = 0.20

# OMP-style protection during backward elimination: species are protected
# from removal based on their fraction *before* any forcing/floor step
# inflates the total (this ordering matters — protecting post-inflation
# fractions was quietly making elimination too permissive in v3.8).
OMP_PROTECT_ML_THRESHOLD = 0.10  # flagged in the audit as too inclusive;
                                  # kept here but see AUDIT NOTES below.

# Backward elimination stopping criterion: remove the lowest-abundance
# species one at a time as long as doing so does not degrade the fit beyond
# this tolerance. This is a FIT-QUALITY criterion, not a recall criterion —
# recall requires literature ground truth, which selection logic must never
# see.
MAX_FIT_DROP = 0.005

# Cosine similarity threshold for deduplicating near-identical basis vectors
# (e.g. two temperature variants that are spectrally indistinguishable at
# MIRI R~100) before they enter the design matrix.
DEDUP_COSINE_THRESHOLD = 0.015

# Reaction-product proxy credits (PARENT_RULES). OCN- at 80K is the thermal
# reaction product of HNCO + NH3 -> NH4+ + OCN-; since HNCO's fundamental
# stretch (4.4 um) is outside MIRI range anyway, OCN- detection is used to
# credit both parents in post-hoc reporting only.
PARENT_RULES = [
    ("OCN", "HNCO"),
    ("OCN", "NH3"),
]

# -----------------------------------------------------------------------------
# AUDIT NOTES carried forward from the pre-new-source audit (unresolved):
#   1. OMP_PROTECT_ML_THRESHOLD (0.10) is likely still too inclusive and
#      weakens backward elimination. Left as-is here but every protected
#      species is now tagged `protected_only=True` in the output so this can
#      be inspected per-source rather than assumed.
#   2. Species that survive ONLY at floor level (report threshold below) are
#      explicitly separated into `probable_detections` rather than folded
#      into the quantitative table.
#   3. Any species with high NNLS fraction but low ML confidence is flagged
#      as `solver_override_suspect` in the report — inspect manually before
#      trusting it, especially the first time this runs on a new source.
# -----------------------------------------------------------------------------

# Quantitative vs probabilistic split (resolves audit concern #2)
QUANTITATIVE_FRACTION_MIN = 0.01   # >1% NNLS fraction -> quantitative
PROBABILISTIC_ML_MIN = 0.70        # ML > 70% but sub-quantitative -> flagged


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LidaEntry:
    """One loaded, normalized LIDA basis spectrum."""
    filename: str
    parent_species: str          # e.g. "SO2", "H2O_CH3COOH"
    cluster_label: str           # e.g. "SO2_CH3OH_1_1_classA"
    temperature_k: Optional[float]
    tau: np.ndarray               # (3000,) normalized to peak=1


@dataclass
class Stage1Result:
    species_scores: Dict[str, float]           # parent species -> ML score
    detected_species: List[str]                # cleared ML_SCORE_FLOOR
    high_confidence_species: List[str]          # cleared FORCE_THRESHOLD


@dataclass
class Stage2Component:
    lida_entry: LidaEntry
    fraction: float
    via: str                     # "nnls" | "forced_floor" | "must_include"
    protected_only: bool = False
    solver_override_suspect: bool = False


@dataclass
class PipelineResult:
    source_name: str
    stage1: Stage1Result
    quantitative: List[Stage2Component]     # fraction >= 1%
    probabilistic: List[Stage2Component]    # ML>70% but sub-1%, upper limit only
    fit_quality: float
    literature_comparison: Optional[dict] = None


# =============================================================================
# STAGE 0 — I/O (CONFIGURE / IMPLEMENT for your environment)
# =============================================================================

def load_target_spectrum(source_name: str, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    IMPLEMENT: load a target protostar's optical depth spectra.

    Returns
    -------
    tau_with_silicate : np.ndarray (3000,)
        Silicate-INCLUDED tau, on MIRI_WAVELENGTH_GRID, for Stage 1 (ML input).
        If your raw data has silicate already removed, reconstruct it first
        (see `reconstruct_silicate`) and save/cache the reconstructed version
        the way you did for IRAS 2A (`*_tau_with_silicate_reconstructed.dat`).
    tau_ice_only : np.ndarray (3000,)
        Silicate-REMOVED tau, for Stage 2 (NNLS input).

    Expected file naming convention (adapt as needed):
        L1448_IRS2:  {data_dir}/L1448IRS2_global_opticaldepth.dat
        L1448_IRS3B: {data_dir}/L1448IRS3B_global_opticaldepth.dat
    """
    raise NotImplementedError(
        "Wire this up to your L1448 IRS2 / IRS3B .dat files. "
        "Reuse the loader you built for IRAS 2A (two-column ASCII, "
        "wavenumber-or-wavelength autodetect, interpolate onto "
        "MIRI_WAVELENGTH_GRID, tau loaded directly with "
        "np.maximum(X_raw, 0.0) — no second -log() transform)."
    )


def reconstruct_silicate(tau_ice_only: np.ndarray, tau_sil_peak: float = 5.0) -> np.ndarray:
    """
    Reconstruct and add back a Drude-profile silicate component so the
    Stage 1 classifiers (trained with silicate present) see spectra in the
    same domain they were trained on.

    IMPLEMENT: port your existing two-component (olivine + pyroxene) Drude
    silicate generator here if the target's silicate-removed spectrum is all
    you have. Peak positions: 9.7 um (Si-O stretch), 18 um (O-Si-O bend).
    """
    raise NotImplementedError(
        "Port the Drude-profile silicate reconstruction from your generator "
        "(two-component olivine + pyroxene) if L1448 data arrives with "
        "silicate already removed, same as was needed for IRAS 2A."
    )


def load_lida_library(lida_dir: str, labels_csv: str) -> List[LidaEntry]:
    """
    IMPLEMENT: load the full LIDA library (your 1209-file / 284-cluster-label
    version), producing one LidaEntry per RAW file (not per cluster) so that
    Stage 2 can access every individual temperature variant within a detected
    parent species' cluster group.

    Each entry needs:
      - parent_species parsed the same way your generator groups files before
        clustering (strip trailing _<temp>K, keep the underlying mixture id)
      - temperature_k parsed from the filename (regex: r'_(\\d+)[Kk]$')
      - tau normalized to peak=1, interpolated onto MIRI_WAVELENGTH_GRID
      - encoding='latin-1' for the ~40 files with Windows-1252 en-dashes
    """
    raise NotImplementedError(
        "Reuse your existing LIDA loader (file autodetection, latin-1 "
        "fallback, peak=1 normalization, 3000-point MIRI grid interpolation). "
        "Just make sure it returns per-file entries, not pre-collapsed "
        "cluster labels — Stage 2 needs every temperature variant available."
    )


def load_stage1_classifiers(classifier_path: str, species_names_path: str,
                             thresholds_path: str):
    """
    IMPLEMENT: load your trained LightGBM classifiers + per-species
    recall-tuned thresholds. These do NOT need to be retrained for v3.9 —
    the leakage was in the NNLS/selection logic downstream of Stage 1, not in
    how the classifiers themselves were trained.
    """
    with open(classifier_path, "rb") as f:
        classifiers = pickle.load(f)
    with open(species_names_path, "rb") as f:
        species_names = pickle.load(f)
    with open(thresholds_path, "rb") as f:
        thresholds = pickle.load(f)
    return classifiers, species_names, thresholds


# =============================================================================
# STAGE 1 — ML PARENT-SPECIES DETECTION (input: tau_with_silicate)
# =============================================================================

def run_stage1(tau_with_silicate: np.ndarray, classifiers, species_names,
                thresholds) -> Stage1Result:
    """
    Score every species classifier against the silicate-included spectrum.
    Excludes EXCLUDED_SPECIES unconditionally. No literature ground truth is
    referenced anywhere in this function.
    """
    X = tau_with_silicate.reshape(1, -1)
    scores: Dict[str, float] = {}

    for name, clf in zip(species_names, classifiers):
        base_species = _strip_cluster_suffix(name)
        if base_species in EXCLUDED_SPECIES:
            continue
        proba = float(clf.predict_proba(X)[0, 1])
        # keep the max score seen across cluster variants of a parent species
        scores[base_species] = max(scores.get(base_species, 0.0), proba)

    detected = [s for s, v in scores.items() if v >= ML_SCORE_FLOOR]
    high_conf = [s for s, v in scores.items() if v >= FORCE_THRESHOLD]

    # MUST_INCLUDE_KEYWORDS: guarantee these parent species get a shot at
    # Stage 2 even if their raw score is borderline, since they're known to
    # be universally deduped/consistent species in the library (not a
    # source-specific literature-driven override).
    for kw in MUST_INCLUDE_KEYWORDS:
        for base_species in scores:
            if kw.upper() in base_species.upper() and base_species not in detected:
                detected.append(base_species)

    return Stage1Result(species_scores=scores, detected_species=detected,
                         high_confidence_species=high_conf)


def _strip_cluster_suffix(cluster_label: str) -> str:
    """e.g. 'H2O_CH3COOH_10_1_16K_classA' -> 'H2O_CH3COOH'."""
    s = re.sub(r"_\d+[Kk](_class[A-Z])?$", "", cluster_label)
    s = re.sub(r"_class[A-Z]$", "", s)
    return s


# =============================================================================
# STAGE 2 — MULTI-TEMPERATURE NNLS QUANTIFICATION (input: tau_ice_only)
# =============================================================================

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _dedup_entries(entries: List[LidaEntry]) -> List[LidaEntry]:
    """Drop near-duplicate temperature variants (cosine sim > 1-threshold)."""
    kept: List[LidaEntry] = []
    for e in entries:
        if all(_cosine_sim(e.tau, k.tau) < (1 - DEDUP_COSINE_THRESHOLD) for k in kept):
            kept.append(e)
    return kept


def build_design_matrix(detected_species: List[str],
                         lida_library: List[LidaEntry]) -> Tuple[np.ndarray, List[LidaEntry]]:
    """
    For every parent species that cleared Stage 1, pull in ALL of its
    temperature variants from the LIDA library (this is the "multi-temperature
    NNLS" step) and dedup near-identical ones.
    """
    candidate_entries = [e for e in lida_library if e.parent_species in detected_species]
    candidate_entries = _dedup_entries(candidate_entries)
    if not candidate_entries:
        return np.zeros((len(MIRI_WAVELENGTH_GRID), 0)), []
    A = np.stack([e.tau for e in candidate_entries], axis=1)
    return A, candidate_entries


def solve_nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if A.shape[1] == 0:
        return np.zeros(0)
    x, _residual = nnls(A, b)
    return x


def force_floor(A: np.ndarray, b: np.ndarray, x: np.ndarray,
                 entries: List[LidaEntry], species_scores: Dict[str, float],
                 must_include_floor: float = 0.002) -> np.ndarray:
    """
    Give a nonzero lower bound to components that Stage 1 flagged as
    high-confidence (score >= FORCE_THRESHOLD) or MUST_INCLUDE, but which
    plain NNLS zeroed out entirely, using bounded lsq_linear (BVLS). This
    reflects genuine physical confidence from the classifier, NOT literature
    recall — the distinction that fixes the v3.8 leakage.
    """
    lb = np.zeros(A.shape[1])
    ub = np.full(A.shape[1], np.inf)
    for i, e in enumerate(entries):
        score = species_scores.get(e.parent_species, 0.0)
        is_must_include = any(kw.upper() in e.parent_species.upper()
                               for kw in MUST_INCLUDE_KEYWORDS)
        if x[i] == 0.0 and (score >= FORCE_THRESHOLD or is_must_include):
            lb[i] = must_include_floor
    if not np.any(lb > 0):
        return x
    result = lsq_linear(A, b, bounds=(lb, ub))
    return np.maximum(result.x, 0.0)


def backward_elimination(A: np.ndarray, b: np.ndarray, x: np.ndarray,
                          entries: List[LidaEntry], species_scores: Dict[str, float]
                          ) -> Tuple[np.ndarray, List[bool]]:
    """
    Remove the lowest-abundance surviving component one at a time, refitting
    NNLS each time, as long as the fit residual doesn't degrade by more than
    MAX_FIT_DROP. Stops at natural completion (no target survivor count).

    Protection (OMP_PROTECT_ML_THRESHOLD) is applied using each species'
    fraction computed BEFORE force_floor() inflated the totals — passed in
    via `x` at the pre-forcing stage — per the fraction-before-forcing fix.

    IMPORTANT: this function never references paper_mixtures or computes
    recall. The only stopping criterion is fit-quality preservation.
    """
    active = list(range(A.shape[1]))
    x_current = x.copy()
    baseline_fit = _fit_quality(A, b, x_current)

    def protected(i: int) -> bool:
        return species_scores.get(entries[i].parent_species, 0.0) >= OMP_PROTECT_ML_THRESHOLD

    protected_only_flags = [False] * A.shape[1]

    changed = True
    while changed and len(active) > 1:
        changed = False
        removable = [i for i in active if not protected(i)]
        if not removable:
            break
        removable.sort(key=lambda i: x_current[i])
        for candidate in removable:
            trial_active = [i for i in active if i != candidate]
            A_trial = A[:, trial_active]
            x_trial_sub = solve_nnls(A_trial, b)
            fit_trial = _fit_quality(A_trial, b, x_trial_sub)
            if baseline_fit - fit_trial <= MAX_FIT_DROP:
                active = trial_active
                x_full = np.zeros(A.shape[1])
                x_full[trial_active] = x_trial_sub
                x_current = x_full
                baseline_fit = fit_trial
                changed = True
                break

    for i in active:
        if protected(i) and x_current[i] > 0 and species_scores.get(
                entries[i].parent_species, 0.0) < FORCE_THRESHOLD:
            protected_only_flags[i] = True

    return x_current, protected_only_flags


def _fit_quality(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    if A.shape[1] == 0:
        return 0.0
    pred = A @ x
    ss_res = np.sum((b - pred) ** 2)
    ss_tot = np.sum((b - np.mean(b)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def run_stage2(tau_ice_only: np.ndarray, stage1: Stage1Result,
               lida_library: List[LidaEntry]) -> Tuple[List[Stage2Component], float]:
    A, entries = build_design_matrix(stage1.detected_species, lida_library)
    if A.shape[1] == 0:
        return [], 0.0

    b = tau_ice_only
    x0 = solve_nnls(A, b)
    x_forced = force_floor(A, b, x0, entries, stage1.species_scores)
    x_final, protected_only_flags = backward_elimination(
        A, b, x_forced, entries, stage1.species_scores)

    fit_quality = _fit_quality(A[:, [i for i, v in enumerate(x_final) if v > 0]],
                                b,
                                x_final[x_final > 0]) if np.any(x_final > 0) else 0.0

    total = x_final.sum()
    components: List[Stage2Component] = []
    for i, e in enumerate(entries):
        if x_final[i] <= 0:
            continue
        fraction = x_final[i] / total if total > 0 else 0.0
        ml_score = stage1.species_scores.get(e.parent_species, 0.0)
        via = "must_include" if any(
            kw.upper() in e.parent_species.upper() for kw in MUST_INCLUDE_KEYWORDS
        ) else ("forced_floor" if x0[i] == 0.0 and x_final[i] > 0 else "nnls")
        override_suspect = fraction > QUANTITATIVE_FRACTION_MIN and ml_score < 0.30
        components.append(Stage2Component(
            lida_entry=e,
            fraction=fraction,
            via=via,
            protected_only=protected_only_flags[i],
            solver_override_suspect=override_suspect,
        ))

    return components, fit_quality


# =============================================================================
# QUANTITATIVE vs PROBABILISTIC SPLIT (audit fix #2)
# =============================================================================

def split_quant_vs_probabilistic(
        components: List[Stage2Component]) -> Tuple[List[Stage2Component], List[Stage2Component]]:
    quantitative = [c for c in components if c.fraction >= QUANTITATIVE_FRACTION_MIN]
    probabilistic = [
        c for c in components
        if c.fraction < QUANTITATIVE_FRACTION_MIN
        and c not in quantitative
    ]
    return quantitative, probabilistic


# =============================================================================
# POST-HOC LITERATURE COMPARISON — ONLY place paper_mixtures may be used
# =============================================================================

def get_parents(mixture_label: str) -> List[str]:
    """Expand a mixture label to its constituent species, crediting proxy
    reaction products via PARENT_RULES (e.g. OCN- credits HNCO and NH3)."""
    parts = re.split(r"[_:]", mixture_label)
    parts = [p for p in parts if p and not re.match(r"^\d", p) and "K" not in p[-3:]]
    expanded = set(parts)
    for proxy, credited in PARENT_RULES:
        if any(proxy.upper() in p.upper() for p in parts):
            expanded.add(credited)
    return sorted(expanded)


def report_against_literature(quantitative: List[Stage2Component],
                               probabilistic: List[Stage2Component],
                               paper_mixtures: Optional[List[str]]) -> Optional[dict]:
    """
    THE ONLY function in this pipeline allowed to see paper_mixtures.
    Called strictly AFTER Stage 2 has finished selecting/quantifying —
    used purely for reporting recall against a known source (e.g. IRAS 2A
    validation), never for L1448 IRS2/IRS3B where there is no ground truth
    to compare against.
    """
    if not paper_mixtures:
        return None
    detected_parents = set()
    for c in quantitative + probabilistic:
        detected_parents.update(get_parents(c.lida_entry.parent_species))

    matched, missed = [], []
    for lit_component in paper_mixtures:
        lit_parents = set(get_parents(lit_component))
        if lit_parents & detected_parents:
            matched.append(lit_component)
        else:
            missed.append(lit_component)

    return {
        "recall": len(matched) / len(paper_mixtures) if paper_mixtures else 0.0,
        "matched": matched,
        "missed": missed,
    }


# =============================================================================
# MAIN — orchestration for a single source
# =============================================================================

def run_pipeline(source_name: str, data_dir: str, lida_dir: str, labels_csv: str,
                  classifier_path: str, species_names_path: str, thresholds_path: str,
                  paper_mixtures: Optional[List[str]] = None) -> PipelineResult:

    classifiers, species_names, thresholds = load_stage1_classifiers(
        classifier_path, species_names_path, thresholds_path)

    tau_with_silicate, tau_ice_only = load_target_spectrum(source_name, data_dir)
    lida_library = load_lida_library(lida_dir, labels_csv)

    stage1 = run_stage1(tau_with_silicate, classifiers, species_names, thresholds)
    components, fit_quality = run_stage2(tau_ice_only, stage1, lida_library)
    quantitative, probabilistic = split_quant_vs_probabilistic(components)

    lit_report = report_against_literature(quantitative, probabilistic, paper_mixtures)

    return PipelineResult(
        source_name=source_name,
        stage1=stage1,
        quantitative=quantitative,
        probabilistic=probabilistic,
        fit_quality=fit_quality,
        literature_comparison=lit_report,
    )


def print_report(result: PipelineResult) -> None:
    print(f"\n{'='*70}\nINDRA v3.9 — {result.source_name}\n{'='*70}")
    print(f"Fit quality (quantitative components only): {result.fit_quality:.1%}")

    print(f"\nQUANTITATIVE detections ({len(result.quantitative)}, fraction >= "
          f"{QUANTITATIVE_FRACTION_MIN:.0%}):")
    for c in sorted(result.quantitative, key=lambda c: -c.fraction):
        flag = " [SOLVER-OVERRIDE SUSPECT]" if c.solver_override_suspect else ""
        prot = " [protected-only]" if c.protected_only else ""
        print(f"  {c.lida_entry.parent_species:25s} {c.fraction:6.2%}  via={c.via}{prot}{flag}")

    print(f"\nPROBABILISTIC / upper-limit only ({len(result.probabilistic)}, "
          f"ML >= {PROBABILISTIC_ML_MIN:.0%} but sub-quantitative fraction):")
    for c in sorted(result.probabilistic, key=lambda c: -c.fraction):
        print(f"  {c.lida_entry.parent_species:25s} {c.fraction:6.2%}  via={c.via}")

    if result.literature_comparison:
        lc = result.literature_comparison
        print(f"\nPost-hoc literature comparison (reporting only, not used in "
              f"selection):\n  recall = {lc['recall']:.1%}")
        print(f"  matched: {lc['matched']}")
        print(f"  missed:  {lc['missed']}")


if __name__ == "__main__":
    # Example usage — fill in your real paths and run per source.
    #
    # IRAS 2A (validation, ground truth available — comparison for sanity
    # checking the rewrite only, not for re-tuning any constants above):
    #
    # result = run_pipeline(
    #     source_name="IRAS2A",
    #     data_dir="/path/to/iras2a_data",
    #     lida_dir="/path/to/lida_raw_data",
    #     labels_csv="/path/to/indra_labels_v38_135k_...csv",
    #     classifier_path="/path/to/lgbm_classifiers_v38.pkl",
    #     species_names_path="/path/to/lgbm_species_names_v38.pkl",
    #     thresholds_path="/path/to/lgbm_thresholds_recall90_v38.pkl",
    #     paper_mixtures=[...15 Rayalacheruvu components...],
    # )
    # print_report(result)
    #
    # L1448 IRS2 / IRS3B (no ground truth — quantitative/probabilistic split
    # IS the result; do not fabricate a paper_mixtures list for these):
    #
    # for source in ["L1448_IRS2", "L1448_IRS3B"]:
    #     result = run_pipeline(
    #         source_name=source,
    #         data_dir="/path/to/l1448_data",
    #         lida_dir="/path/to/lida_raw_data",
    #         labels_csv="/path/to/indra_labels_v38_135k_...csv",
    #         classifier_path="/path/to/lgbm_classifiers_v38.pkl",
    #         species_names_path="/path/to/lgbm_species_names_v38.pkl",
    #         thresholds_path="/path/to/lgbm_thresholds_recall90_v38.pkl",
    #         paper_mixtures=None,
    #     )
    #     print_report(result)
    pass
