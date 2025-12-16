# Limitations and Scope Boundaries

## AMR Thesis Project: Explicit Limitations Documentation

This document provides comprehensive documentation of study limitations, scope boundaries, and negative results for the AMR thesis project. Transparent acknowledgment of limitations strengthens scientific credibility.

---

## Table of Contents

1. [Study Design Limitations](#1-study-design-limitations)
2. [Methodological Limitations](#2-methodological-limitations)
3. [Data Limitations](#3-data-limitations)
4. [Analytical Limitations](#4-analytical-limitations)
5. [Scope Boundaries](#5-scope-boundaries)
6. [Negative Results](#6-negative-results)

---

## 1. Study Design Limitations

### 1.1 No Temporal Inference

| Limitation | Impact | Acknowledgment |
|------------|--------|----------------|
| **Cross-sectional design** | Cannot establish temporal trends or resistance evolution | Results represent a snapshot; no inference about changes over time |
| **No longitudinal data** | Cannot assess resistance trajectory or persistence | Patterns may shift with time; validation requires repeat sampling |
| **No seasonal analysis** | Cannot account for seasonal variation in resistance | Environmental factors may vary seasonally |

**Standard Statement**:
> "This study employs a cross-sectional design. Results represent patterns at a single time point and do **not** support inference about temporal trends, resistance evolution, or predictive forecasting."

### 1.2 Dataset Dependency

| Limitation | Impact | Acknowledgment |
|------------|--------|----------------|
| **Single dataset** | Findings may not generalize to other populations | External validation required for broader application |
| **Specific regions** | Limited to Philippine regions sampled | Regional patterns may differ in other geographic contexts |
| **Specific sources** | Environmental/aquatic focus | Clinical isolates may show different patterns |

**Standard Statement**:
> "Results reflect patterns within the analyzed dataset from specific Philippine regions and environmental sources. Generalization to other populations, regions, or sample types requires external validation."

### 1.3 No Causal Inference

| Limitation | Impact | Acknowledgment |
|------------|--------|----------------|
| **Observational design** | Cannot establish causation | Associations do not imply causal relationships |
| **Confounding factors** | Uncontrolled variables may influence patterns | Results may be affected by unmeasured confounders |
| **No experimental manipulation** | Cannot test interventions | Descriptive analysis only |

**Standard Statement**:
> "All associations reported are observational. This study does **not** establish causal relationships between environmental factors and resistance patterns."

---

## 2. Methodological Limitations

### 2.1 Clustering Subjectivity

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Number of clusters** | Choice of k affects interpretation | Multiple k values explored; selection justified |
| **Linkage method** | Different methods may yield different clusters | Ward's method selected with rationale |
| **Distance metric** | Euclidean may not capture all similarity aspects | Ordinal encoding preserves biological meaning |

**Acknowledgment**:
> "The number of clusters is a parameter choice that influences interpretation. While k=[value] was selected based on [criteria], alternative clusterings may reveal different patterns."

### 2.2 Model Selection

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Limited model set** | Three models may not capture all patterns | Selected for interpretability and diversity |
| **No hyperparameter tuning** | Default parameters used | Focus on pattern discrimination, not optimization |
| **No cross-validation** | Single train-test split | Stratified split with fixed random state |

### 2.3 Feature Importance Interpretation

| Limitation | Description | Acknowledgment |
|------------|-------------|----------------|
| **Associative only** | Importance does not imply causation | High importance indicates discriminative power, not mechanism |
| **Model-dependent** | Different models rank features differently | Report from multiple models for robustness |
| **No biological validation** | Statistical importance vs. biological significance | Interpretation considers known mechanisms |

**Acknowledgment**:
> "Feature importance scores indicate statistical contribution to group separation. High importance does **not** imply biological causation or clinical significance."

---

## 3. Data Limitations

### 3.1 Sample Size Considerations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Overall sample size** | May limit statistical power | Report confidence intervals where applicable |
| **Class imbalance** | Some categories have few samples | Stratified sampling; macro-averaged metrics |
| **Regional variation** | Unequal sampling across regions | Acknowledge in regional comparisons |

### 3.2 Missing Data

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Missing AST values** | Some antibiotic results not available | Median imputation with coverage thresholds |
| **Missing metadata** | Some isolates lack complete metadata | Document missing rates; exclude from specific analyses |

**Standard Statement**:
> "Missing data were handled through median imputation for resistance values. Antibiotics tested in <70% of isolates were excluded. Isolates with >30% missing data were removed."

### 3.3 Data Quality

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Inter-laboratory variation** | AST methods may differ across labs | Standardized interpretation (S/I/R) |
| **Reporting inconsistencies** | Naming conventions varied | Controlled vocabularies for standardization |

---

## 4. Analytical Limitations

### 4.1 MDR Self-Consistency

| Limitation | Description | Acknowledgment |
|------------|-------------|----------------|
| **Circular relationship** | MDR derived from same features used for discrimination | Explicitly termed "self-consistency discrimination" |
| **Not predictive** | MDR discrimination does not predict future MDR | Results assess pattern alignment, not prediction |

**Standard Statement**:
> "MDR discrimination evaluates how consistently resistance fingerprints align with MDR status derived from the same features. This represents **self-consistency assessment**, not predictive capability."

### 4.2 Statistical Testing

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Multiple comparisons** | Many tests increase Type I error risk | Report uncorrected p-values with caution |
| **Small expected counts** | Chi-square assumptions may be violated | Note when expected counts < 5 |
| **Association vs. causation** | Significant associations are not causal | Interpret as observational findings |

### 4.3 Pattern Discrimination vs. Prediction

| What This Study Does | What This Study Does NOT Do |
|---------------------|----------------------------|
| Evaluates pattern consistency | Predict future resistance |
| Assesses discriminative capacity | Forecast clinical outcomes |
| Identifies associations | Establish causal mechanisms |
| Characterizes dataset patterns | Generalize to external populations |

---

## 5. Scope Boundaries

### 5.1 Explicit Scope Statement

> ⚠️ **CRITICAL DISCLAIMER**: This tool is intended for **exploratory pattern recognition and surveillance analysis only**. It should **NOT** be used for:
> - Clinical decision support
> - Treatment recommendations
> - Patient-level predictions
> - Regulatory submissions without further validation

### 5.2 What This Project Provides

| Capability | Description |
|------------|-------------|
| ✅ Pattern identification | Discover natural groupings in resistance data |
| ✅ Pattern discrimination | Evaluate how patterns align with known categories |
| ✅ Descriptive analysis | Characterize resistance profiles and associations |
| ✅ Visualization | Interactive exploration of resistance patterns |
| ✅ Hypothesis generation | Identify patterns for further investigation |

### 5.3 What This Project Does NOT Provide

| Non-Capability | Reason |
|----------------|--------|
| ❌ Clinical decision support | Not validated for clinical use |
| ❌ Treatment guidance | Outside scope; requires clinical validation |
| ❌ Outbreak prediction | No temporal modeling capability |
| ❌ Resistance forecasting | Cross-sectional design; no predictive models |
| ❌ Causal inference | Observational design only |

### 5.4 No Patient-Level Data

> "This system processes **environmental and aquatic samples only**. No patient-level identifiers are collected, processed, or analyzed. The tool is designed for surveillance research, not clinical care."

---

## 6. Negative Results

### 6.1 Expected Patterns Not Observed

[Document any expected findings that were not observed]

| Expected Finding | Observed Result | Possible Explanation |
|------------------|-----------------|---------------------|
| [Expected pattern] | [What was actually found] | [Possible reasons] |

### 6.2 Non-Significant Associations

[Document associations that were tested but not statistically significant]

| Association Tested | Result | Interpretation |
|-------------------|--------|----------------|
| [Variables] | p = [value], not significant | [What this means] |

### 6.3 Model Limitations Observed

[Document where models performed poorly]

| Model | Task | Performance | Notes |
|-------|------|-------------|-------|
| [Model] | [Task] | F1 = [low value] | [Possible reasons for poor performance] |

---

## Summary Statement for Thesis

### Comprehensive Limitations Acknowledgment

> "This study has several limitations that should be considered when interpreting results:
>
> 1. **Temporal scope**: The cross-sectional design precludes inference about temporal trends or resistance evolution.
>
> 2. **Generalizability**: Findings are specific to the analyzed dataset from Philippine environmental samples and may not generalize to other populations or contexts.
>
> 3. **Causal inference**: All associations are observational; no causal relationships can be established.
>
> 4. **Analytical constraints**: Supervised learning results represent pattern discrimination within existing data, not predictive capability for new samples.
>
> 5. **Clinical applicability**: This tool is for exploratory surveillance analysis only and should not be used for clinical decision support.
>
> Despite these limitations, the methodology provides a rigorous framework for AMR pattern recognition that can generate hypotheses for further investigation and support surveillance efforts."

---

## Document Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial limitations documentation |

---

*This document is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*

**See also**: [METHODOLOGY.md](METHODOLOGY.md) | [DOCUMENTATION.md](DOCUMENTATION.md)
