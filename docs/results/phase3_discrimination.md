# Phase 3 Results: Clustering and Pattern Discrimination

## Hierarchical Clustering and Supervised Learning Results

This document provides a structured template for documenting Phase 3 (clustering) and Phase 4 (supervised discrimination) results for the AMR thesis project.

---

## Table of Contents

1. [Clustering Results (Phase 3)](#1-clustering-results-phase-3)
2. [Supervised Discrimination Results (Phase 4)](#2-supervised-discrimination-results-phase-4)
3. [Feature Importance Analysis](#3-feature-importance-analysis)
4. [Cluster-Supervised Comparison](#4-cluster-supervised-comparison)

---

## 1. Clustering Results (Phase 3)

### 1.1 Clustering Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Linkage method | Ward | Minimizes within-cluster variance |
| Distance metric | Euclidean | Captures ordinal resistance differences |
| Number of clusters | [k] | [Selection rationale] |
| Imputation | Median | Robust to outliers |

### 1.2 Cluster Summary

| Cluster | N | Percentage | MDR Rate | Mean MAR | Characterization |
|---------|---|------------|----------|----------|------------------|
| 1 | [n] | [%] | [%] | [value] | [Brief description] |
| 2 | [n] | [%] | [%] | [value] | [Brief description] |
| 3 | [n] | [%] | [%] | [value] | [Brief description] |
| ... | ... | ... | ... | ... | ... |
| **Total** | **[N]** | **100%** | **[%]** | **[value]** | - |

### 1.3 Cluster Profiles

#### Cluster 1: [Descriptive Name]

| Antibiotic | Mean Resistance | Level |
|------------|-----------------|-------|
| [AB with highest] | [value] | High/Moderate/Low |
| [AB with next highest] | [value] | High/Moderate/Low |
| ... | ... | ... |

**Archetype**: [Description of resistance pattern]

#### Cluster 2: [Descriptive Name]

[Repeat for each cluster]

### 1.4 Cluster Quality Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Average cluster size | [N] | [Interpretation] |
| Size standard deviation | [value] | [Balance assessment] |
| Within-cluster variance | [value] | [Tightness assessment] |

### 1.5 Dendrogram Analysis

**Key Observations**:
- [Observation about merge heights]
- [Observation about cluster separation]
- [Natural cut point justification]

---

## 2. Supervised Discrimination Results (Phase 4)

> **Scope Statement**: Results reflect pattern discrimination within the analyzed dataset and are **not** predictive of performance on external datasets. Evaluation metrics quantify the consistency with which resistance fingerprints align with predefined categories.

### 2.1 Task A: Species Discrimination

#### Data Split

| Set | Samples | Percentage |
|-----|---------|------------|
| Training | [N] | 80% |
| Test | [N] | 20% |
| **Total** | **[N]** | **100%** |

#### Model Evaluation Metrics

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|-------|----------|-------------------|----------------|------------------|
| Logistic Regression | [value] | [value] | [value] | [value] |
| Random Forest | [value] | [value] | [value] | [value] |
| k-Nearest Neighbors | [value] | [value] | [value] | [value] |

**Best Model**: [Model name] (F1 = [value])

#### Confusion Matrix (Best Model)

| Actual \ Discriminated | Species A | Species B | Species C |
|------------------------|-----------|-----------|-----------|
| **Species A** | [TP] | [FN] | [FN] |
| **Species B** | [FP] | [TP] | [FN] |
| **Species C** | [FP] | [FP] | [TP] |

#### Interpretation

> "The [best model] demonstrates [strong/moderate/weak] discriminative capacity for species identification based on resistance fingerprints (F1 = [value]). [Additional observations about which species are well-distinguished vs. confused]."

### 2.2 Task B: MDR Discrimination

> **MDR Target Transparency**: The MDR label is derived from the same resistance features used as input. This task evaluates self-consistency discrimination—how consistently resistance fingerprints align with MDR status.

#### Data Split

| Set | Non-MDR | MDR | Total |
|-----|---------|-----|-------|
| Training | [N] | [N] | [N] |
| Test | [N] | [N] | [N] |
| **Total** | **[N]** | **[N]** | **[N]** |

#### Model Evaluation Metrics

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|-------|----------|-------------------|----------------|------------------|
| Logistic Regression | [value] | [value] | [value] | [value] |
| Random Forest | [value] | [value] | [value] | [value] |
| k-Nearest Neighbors | [value] | [value] | [value] | [value] |

**Best Model**: [Model name] (F1 = [value])

#### Confusion Matrix (Best Model)

| Actual \ Discriminated | Non-MDR | MDR |
|------------------------|---------|-----|
| **Non-MDR** | [TN] | [FP] |
| **MDR** | [FN] | [TP] |

#### Interpretation

> "MDR discrimination shows [strong/moderate/weak] self-consistency (F1 = [value]), indicating that resistance fingerprints [reliably/moderately/poorly] align with MDR status as defined by ≥3 resistant antibiotic classes."

---

## 3. Feature Importance Analysis

### 3.1 Species Discrimination - Top Antibiotics

| Rank | Antibiotic | Importance Score | Model | Biological Context |
|------|------------|------------------|-------|-------------------|
| 1 | [AB] | [value] | [model] | [Context] |
| 2 | [AB] | [value] | [model] | [Context] |
| 3 | [AB] | [value] | [model] | [Context] |
| ... | ... | ... | ... | ... |

### 3.2 MDR Discrimination - Top Antibiotics

| Rank | Antibiotic | Importance Score | Model | Biological Context |
|------|------------|------------------|-------|-------------------|
| 1 | [AB] | [value] | [model] | [Context] |
| 2 | [AB] | [value] | [model] | [Context] |
| 3 | [AB] | [value] | [model] | [Context] |
| ... | ... | ... | ... | ... |

### 3.3 Model Agreement Analysis

| Antibiotic | LR Rank | RF Rank | Appears in Both Top 5? |
|------------|---------|---------|------------------------|
| [AB1] | [rank] | [rank] | Yes/No |
| [AB2] | [rank] | [rank] | Yes/No |
| ... | ... | ... | ... |

**Interpretation**: [Observations about which antibiotics consistently show importance across models]

> **Note**: Importance scores indicate associative contribution to group separation, not predictive or causal relationships. Biological interpretation should consider known resistance mechanisms.

---

## 4. Cluster-Supervised Comparison

### 4.1 Clusters vs. MDR Status

| Cluster | Non-MDR | MDR | Total | MDR Rate | Purity |
|---------|---------|-----|-------|----------|--------|
| 1 | [n] | [n] | [N] | [%] | [value] |
| 2 | [n] | [n] | [N] | [%] | [value] |
| ... | ... | ... | ... | ... | ... |
| **Total** | **[N]** | **[N]** | **[N]** | **[%]** | - |

**Chi-Square Test**: χ² = [value], df = [df], p = [p-value]

**Interpretation**: [Significant/Not significant] association between clusters and MDR status. [Observations about which clusters are MDR-enriched/depleted]

### 4.2 Clusters vs. Species

| Cluster | Sp. A | Sp. B | Sp. C | Total | Dominant | Purity |
|---------|-------|-------|-------|-------|----------|--------|
| 1 | [n] | [n] | [n] | [N] | [Sp] | [value] |
| 2 | [n] | [n] | [n] | [N] | [Sp] | [value] |
| ... | ... | ... | ... | ... | ... | ... |

**Chi-Square Test**: χ² = [value], df = [df], p = [p-value]

**Interpretation**: [Significant/Not significant] association between clusters and species. [Observations about species-cluster alignment]

---

## Figure Captions

### Figure 3.1: Hierarchical Clustering Dendrogram

> "Dendrogram showing hierarchical clustering of resistance profiles (n=[N] isolates) using Ward linkage with Euclidean distance. Horizontal line indicates cut point for k=[k] clusters. Branch heights reflect merge distances; longer branches indicate more distinct groupings."

### Figure 3.2: Resistance Heatmap with Clustering

> "Heatmap illustrating hierarchical clustering of resistance profiles. Clusters were derived **solely from resistance features** (S=0, I=1, R=2); metadata variables are shown for **post-hoc interpretation only** and did not influence cluster formation."

### Figure 4.1: Confusion Matrix (Species Discrimination)

> "Confusion matrix for species discrimination using [model] classifier (n=[N] test samples). Results reflect pattern consistency within the analyzed dataset, not predictive performance on external data."

### Figure 4.2: Confusion Matrix (MDR Discrimination)

> "Confusion matrix for MDR discrimination using [model] classifier (n=[N] test samples). MDR discrimination evaluates self-consistency alignment between resistance fingerprints and MDR status."

---

## Document Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial results template |

---

*This document is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*

**See also**: [clustering.md](../methods/clustering.md) | [supervised_models.md](../methods/supervised_models.md) | [phase4_environment.md](phase4_environment.md)
