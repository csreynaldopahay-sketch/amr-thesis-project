# Clustering Methods Documentation

## Phase 3: Unsupervised Structure Identification

This document provides comprehensive documentation of hierarchical agglomerative clustering parameters, methods, and interpretation guidelines for the AMR thesis project.

---

## Table of Contents

1. [Parameter Transparency](#1-parameter-transparency)
2. [Clustering Algorithm Details](#2-clustering-algorithm-details)
3. [Metadata Handling (Avoiding Interpretive Leakage)](#3-metadata-handling-avoiding-interpretive-leakage)
4. [Cluster Quality Assessment](#4-cluster-quality-assessment)
5. [Interpretation Guidelines](#5-interpretation-guidelines)

---

## 1. Parameter Transparency

### 1.1 Clustering Parameter Summary Table

| Parameter | Value | Options Available | Rationale |
|-----------|-------|-------------------|-----------|
| **Linkage Method** | Ward | ward, complete, average, single | Ward's method minimizes within-cluster variance, producing compact clusters suitable for resistance pattern identification |
| **Distance Metric** | Euclidean | euclidean, manhattan | Euclidean distance with ordinal encoding (S=0, I=1, R=2) captures meaningful resistance pattern similarity |
| **Number of Clusters** | 5 | User-configurable (2-10) | Default based on typical AMR pattern diversity; optimized based on cluster quality metrics |
| **Missing Value Imputation** | Median | mean, median, most_frequent | Median is robust to outliers in resistance data |

### 1.2 Standardized Method Statement

> "Hierarchical agglomerative clustering was performed using **Ward linkage** with **Euclidean distance** to identify resistance profile structure. Clusters were derived **solely from encoded resistance features** (S=0, I=1, R=2); metadata variables (region, site, species, environment) were used **only for post-hoc interpretation** and did not influence cluster formation."

### 1.3 Parameter Selection Justification

| Parameter | Selection Rationale |
|-----------|---------------------|
| **Ward linkage** | Minimizes total within-cluster variance; produces compact, balanced clusters ideal for pattern discrimination |
| **Euclidean distance** | With ordinal encoding, captures meaningful differences between resistance profiles; S→I→R represents increasing resistance |
| **Median imputation** | Preserves central tendency while being robust to extreme values in resistance profiles |

---

## 2. Clustering Algorithm Details

### 2.1 Algorithm Description

```
HIERARCHICAL AGGLOMERATIVE CLUSTERING PROCEDURE:

1. INITIALIZATION
   - Each isolate starts as its own cluster (N clusters for N isolates)

2. ITERATION (repeat N-1 times):
   a. Compute pairwise distances between all clusters using Ward's criterion
   b. Identify the pair of clusters with minimum distance
   c. Merge the two closest clusters into a single cluster
   d. Update distance matrix

3. TERMINATION
   - Continue until all isolates belong to a single cluster
   - Store merge history in linkage matrix

4. CLUSTER ASSIGNMENT
   - Cut dendrogram at specified level (n_clusters)
   - Assign cluster labels to each isolate
```

### 2.2 Ward's Linkage Method

**Mathematical Definition**:
```
d(A ∪ B, C) = √[(nA + nC)/(nA + nB + nC) × d²(A,C) + 
               (nB + nC)/(nA + nB + nC) × d²(B,C) - 
               nC/(nA + nB + nC) × d²(A,B)]

Where:
- A, B = clusters being merged
- C = cluster to which distance is being calculated
- nX = number of isolates in cluster X
- d(X,Y) = distance between clusters X and Y
```

### 2.3 Cluster Assignment Criteria

| Method | Parameter | Description | Use Case |
|--------|-----------|-------------|----------|
| **maxclust** | n_clusters | Cut dendrogram to form exactly n clusters | When predetermined number of groups desired |
| **distance** | threshold | Cut dendrogram at specified distance threshold | When natural separation distance known |

### 2.4 Input Features (CRITICAL)

**Features INCLUDED** (used for clustering):
- All `{ANTIBIOTIC}_encoded` columns (e.g., AM_encoded, AMC_encoded, etc.)

**Features EXCLUDED** (NOT used for clustering):
| Excluded Variable | Reason |
|-------------------|--------|
| REGION | Metadata - post-hoc interpretation only |
| SITE | Metadata - post-hoc interpretation only |
| SPECIES | Metadata - post-hoc interpretation only |
| ENVIRONMENT | Metadata - post-hoc interpretation only |
| SAMPLE_SOURCE | Metadata - post-hoc interpretation only |
| MDR_FLAG | Derived feature - used for post-hoc comparison |
| MAR_INDEX | Derived feature - used for post-hoc characterization |
| CODE | Identifier - not a feature |

---

## 3. Metadata Handling (Avoiding Interpretive Leakage)

### 3.1 Critical Statement on Metadata Exclusion

> **IMPORTANT**: Metadata variables (region, site, species, environment, sample source) were **explicitly excluded** during cluster formation. Clusters are based **entirely on resistance fingerprints**. Metadata is used **only for post-hoc interpretation** to characterize and describe discovered clusters.

### 3.2 Why This Matters

| Issue | Consequence | Prevention |
|-------|-------------|------------|
| **Circular reasoning** | Cannot claim "clusters differ by region" if region influenced clustering | Exclude metadata from clustering input |
| **Interpretive leakage** | Confounds pattern discovery with prior categorization | Strict feature-metadata separation |
| **Validity threats** | Panel reviewers will question methodology | Document exclusion explicitly |

### 3.3 Post-Hoc Interpretation Process

1. **First**: Perform clustering using ONLY resistance features
2. **Then**: After cluster labels assigned, cross-tabulate with metadata
3. **Finally**: Report associations as **observations**, not influences

**Correct Language**:
> "Cluster 3 was **enriched** for isolates from water sources (chi-square p<0.05), suggesting an association between water environments and this resistance profile."

**Incorrect Language** (avoid):
> ~~"Water isolates clustered together due to their environmental origin."~~

---

## 4. Cluster Quality Assessment

### 4.1 Quality Metrics Computed

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Cluster sizes** | Number of isolates per cluster | Balanced sizes preferred |
| **Size standard deviation** | Variation in cluster sizes | Lower = more balanced |
| **Within-cluster variance** | Average variance within clusters | Lower = tighter clusters |
| **Inconsistency coefficients** | Statistical measure from scipy | Higher = more distinct clusters |

### 4.2 Cluster Number Selection

The pipeline evaluates cluster quality for k=2 to k=10 clusters:

| k | Evaluation Criteria |
|---|---------------------|
| 2 | Minimum meaningful grouping |
| 3-5 | Typical AMR pattern diversity |
| 6-10 | Fine-grained pattern detection |

**Selection Guidance**:
- Consider biological interpretability
- Balance between granularity and sample size per cluster
- Evaluate cluster profiles for meaningful resistance patterns

### 4.3 Dendrogram Interpretation

| Dendrogram Feature | Interpretation |
|--------------------|----------------|
| **Height of merge** | Distance at which clusters joined; higher = more dissimilar |
| **Branch length** | Duration of cluster stability before next merge |
| **Clear horizontal cuts** | Natural cluster boundaries |

---

## 5. Interpretation Guidelines

### 5.1 Cluster Profile Characterization

For each cluster, compute and report:

| Characteristic | Calculation | Purpose |
|----------------|-------------|---------|
| Mean resistance profile | Average encoded value per antibiotic | Define cluster archetype |
| High-resistance antibiotics | Mean > 1.5 | Identify dominant resistance patterns |
| Low-resistance antibiotics | Mean < 0.5 | Identify susceptibility patterns |
| MDR proportion | % of MDR isolates | Risk assessment |
| MAR index mean | Average MAR | Overall resistance level |

### 5.2 Resistance Level Classification

| Mean Resistance Score | Classification |
|-----------------------|----------------|
| > 1.5 | High resistance (approaching "R") |
| 1.0 - 1.5 | Moderate-high resistance |
| 0.5 - 1.0 | Moderate resistance |
| < 0.5 | Low resistance (approaching "S") |

### 5.3 Reporting Template

**Example Cluster Description**:
> "**Cluster 2** (n=45 isolates, 23% of dataset) exhibited a **high-resistance** archetype with elevated resistance to β-lactams (mean AM_encoded=1.8) and aminoglycosides (mean GM_encoded=1.6). This cluster showed **65% MDR prevalence** with mean MAR index of 0.42. Post-hoc analysis revealed enrichment in hospital effluent samples (chi-square p=0.003)."

### 5.4 Mandatory Caption Elements for Figures

Every clustering figure caption must include:

1. **Data source**: "Based on encoded resistance profiles from N isolates"
2. **Method**: "Hierarchical clustering with Ward linkage and Euclidean distance"
3. **Interpretation boundary**: "Metadata shown for post-hoc interpretation only"

**Example Caption**:
> "**Figure 3.1**: Heatmap illustrating hierarchical clustering of resistance profiles (n=187 isolates). Clusters were derived solely from resistance features using Ward linkage with Euclidean distance; metadata variables (region, species, environment) are shown in annotation tracks for **post-hoc interpretation only** and did not influence cluster formation."

---

## Document Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial clustering documentation |

---

*This document is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*

**See also**: [preprocessing.md](preprocessing.md) | [supervised_models.md](supervised_models.md) | [METHODOLOGY.md](../METHODOLOGY.md)
