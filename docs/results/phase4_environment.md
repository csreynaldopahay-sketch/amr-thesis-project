# Phase 4 Results: Environmental and Regional Analysis

## Regional Distribution and Multivariate Analysis Results

This document provides a structured template for documenting Phase 5 (regional and environmental analysis) results for the AMR thesis project.

---

## Table of Contents

1. [Regional Distribution Analysis](#1-regional-distribution-analysis)
2. [Environmental Distribution Analysis](#2-environmental-distribution-analysis)
3. [Principal Component Analysis Results](#3-principal-component-analysis-results)
4. [Statistical Associations](#4-statistical-associations)

---

## 1. Regional Distribution Analysis

> **Scope Statement**: Regional associations are **observational** and reflect patterns in the analyzed dataset. Results do not establish causation between geographic factors and resistance profiles.

### 1.1 Cluster Distribution by Region

| Cluster | BARMM | Region III | Region VIII | Total |
|---------|-------|------------|-------------|-------|
| Cluster 1 | [n] ([%]) | [n] ([%]) | [n] ([%]) | [N] |
| Cluster 2 | [n] ([%]) | [n] ([%]) | [n] ([%]) | [N] |
| Cluster 3 | [n] ([%]) | [n] ([%]) | [n] ([%]) | [N] |
| ... | ... | ... | ... | ... |
| **Total** | **[N]** | **[N]** | **[N]** | **[Total]** |

**Chi-Square Test**: χ² = [value], df = [df], p = [p-value]

**Interpretation**: [Significant/Not significant] association between clusters and regions.

### 1.2 Regional MDR Rates

| Region | Total Isolates | MDR Count | MDR Rate | Fold Enrichment |
|--------|----------------|-----------|----------|-----------------|
| BARMM | [N] | [n] | [%] | [×] |
| Region III | [N] | [n] | [%] | [×] |
| Region VIII | [N] | [n] | [%] | [×] |
| **Overall** | **[N]** | **[n]** | **[%]** | **1.0×** |

### 1.3 Regional Cluster Enrichment

| Region | Most Enriched Cluster | Enrichment | Interpretation |
|--------|----------------------|------------|----------------|
| BARMM | Cluster [X] | [×] | [Brief interpretation] |
| Region III | Cluster [X] | [×] | [Brief interpretation] |
| Region VIII | Cluster [X] | [×] | [Brief interpretation] |

---

## 2. Environmental Distribution Analysis

### 2.1 Cluster Distribution by Environment

| Cluster | Water | Fish | Hospital | Total |
|---------|-------|------|----------|-------|
| Cluster 1 | [n] ([%]) | [n] ([%]) | [n] ([%]) | [N] |
| Cluster 2 | [n] ([%]) | [n] ([%]) | [n] ([%]) | [N] |
| Cluster 3 | [n] ([%]) | [n] ([%]) | [n] ([%]) | [N] |
| ... | ... | ... | ... | ... |
| **Total** | **[N]** | **[N]** | **[N]** | **[Total]** |

**Chi-Square Test**: χ² = [value], df = [df], p = [p-value]

**Interpretation**: [Significant/Not significant] association between clusters and environments.

### 2.2 Environment MDR Rates

| Environment | Total Isolates | MDR Count | MDR Rate | Fold Enrichment |
|-------------|----------------|-----------|----------|-----------------|
| Water | [N] | [n] | [%] | [×] |
| Fish | [N] | [n] | [%] | [×] |
| Hospital | [N] | [n] | [%] | [×] |
| **Overall** | **[N]** | **[n]** | **[%]** | **1.0×** |

### 2.3 Sample Source Details

| Sample Source | Count | MDR Rate | Most Common Cluster |
|---------------|-------|----------|---------------------|
| Drinking Water | [N] | [%] | Cluster [X] |
| Lake Water | [N] | [%] | Cluster [X] |
| River Water | [N] | [%] | Cluster [X] |
| Fish Banak | [N] | [%] | Cluster [X] |
| Fish Gusaw | [N] | [%] | Cluster [X] |
| Fish Tilapia | [N] | [%] | Cluster [X] |
| Effluent Untreated | [N] | [%] | Cluster [X] |
| Effluent Treated | [N] | [%] | Cluster [X] |

---

## 3. Principal Component Analysis Results

### 3.1 Variance Explained

| Component | Variance Explained | Cumulative | Interpretation |
|-----------|-------------------|------------|----------------|
| PC1 | [%] | [%] | [Brief interpretation] |
| PC2 | [%] | [%] | [Brief interpretation] |
| PC3 | [%] | [%] | [Brief interpretation] |

**Total variance explained (PC1+PC2)**: [%]

> **Interpretation Guide:**
> - **If cumulative variance ≥ 60%**: The 2D projection provides a representative view of resistance structure. Interpretation of visual patterns is well-supported.
> - **If cumulative variance 50-60%**: The 2D projection captures moderate variance. Interpret patterns cautiously; some information is lost.
> - **If cumulative variance < 50%**: The 2D projection represents a simplified view. Full resistance space is multi-dimensional; these projections emphasize only the two dominant axes of variation. **Acknowledge this as a limitation in reporting.**

### 3.2 Component Loadings (Top Antibiotics)

#### PC1 Loadings

| Antibiotic | Loading | Direction |
|------------|---------|-----------|
| [AB with highest absolute] | [value] | Positive/Negative |
| [AB with second highest] | [value] | Positive/Negative |
| [AB with third highest] | [value] | Positive/Negative |

**PC1 Interpretation**: [Description of what PC1 captures - e.g., "β-lactam resistance variation"]

#### PC2 Loadings

| Antibiotic | Loading | Direction |
|------------|---------|-----------|
| [AB with highest absolute] | [value] | Positive/Negative |
| [AB with second highest] | [value] | Positive/Negative |
| [AB with third highest] | [value] | Positive/Negative |

**PC2 Interpretation**: [Description of what PC2 captures]

### 3.3 PCA Visualization Summary

| Color Coding Variable | Visual Separation | Notes |
|----------------------|-------------------|-------|
| Cluster | Clear/Moderate/Overlapping | [Observations] |
| Region | Clear/Moderate/Overlapping | [Observations] |
| MDR Status | Clear/Moderate/Overlapping | [Observations] |
| Species | Clear/Moderate/Overlapping | [Observations] |

---

## 4. Statistical Associations

### 4.1 Chi-Square Test Summary

| Association | χ² | df | p-value | Significant? | Effect |
|-------------|----|----|---------|--------------|--------|
| Cluster × Region | [value] | [df] | [p] | Yes/No | [Cramér's V if significant] |
| Cluster × Environment | [value] | [df] | [p] | Yes/No | [Cramér's V if significant] |
| Cluster × Species | [value] | [df] | [p] | Yes/No | [Cramér's V if significant] |
| Cluster × MDR | [value] | [df] | [p] | Yes/No | [Cramér's V if significant] |
| Species × Environment | [value] | [df] | [p] | Yes/No | [Cramér's V if significant] |
| Region × MDR | [value] | [df] | [p] | Yes/No | [Cramér's V if significant] |

### 4.2 Key Findings Summary

1. **Regional Patterns**: [Summary of regional distribution findings]

2. **Environmental Patterns**: [Summary of environmental distribution findings]

3. **PCA Insights**: [Summary of PCA findings]

4. **MDR Hotspots**: [Summary of MDR enrichment patterns]

---

## Figure Captions

### Figure 5.1: PCA Scatter Plot by Cluster

> "Principal Component Analysis of resistance profiles (n=[N] isolates, [M] antibiotics). **PC1 explains [X]% and PC2 explains [Y]% of total variance ([X+Y]% cumulative)**. Points colored by cluster assignment; cluster overlap indicates shared resistance pattern features. [If cumulative <50%: Note that the 2D projection captures less than 50% of total variance; patterns should be interpreted with caution.]"

### Figure 5.2: PCA Scatter Plot by Region

> "PCA visualization with regional coloring (n=[N] isolates). **PC1 and PC2 explain [X]% and [Y]% of total variance respectively ([X+Y]% cumulative)**. Separation patterns reflect regional variation in resistance profiles. Regional associations are observational and do not imply causation."

### Figure 5.3: PCA Biplot

> "PCA biplot showing isolate distribution and antibiotic loadings. **PC1+PC2 explain [X+Y]% of total variance**. Arrows indicate antibiotic contributions to principal components; longer arrows indicate stronger contributions. Antibiotics pointing in similar directions show correlated resistance patterns."

### Figure 5.4: Regional Distribution Bar Chart

> "Distribution of isolates across clusters by geographic region. Chi-square test indicates [significant/non-significant] association (χ²=[value], p=[p]). Results reflect observed patterns in the analyzed dataset."

### Figure 5.5: Environmental Distribution Chart

> "Distribution of isolates across clusters by environmental category. Enrichment values indicate fold-difference from overall dataset proportions."

### Figure 5.6: Scree Plot

> "Scree plot showing variance explained by principal components. **PC1: [X]%, PC2: [Y]% (cumulative: [X+Y]%)**. The red line shows cumulative variance; the 80% threshold indicates the number of components needed to capture most variance. [If PC1+PC2 <50%: The low cumulative variance for PC1+PC2 indicates that resistance patterns are distributed across multiple dimensions, and 2D visualizations should be interpreted with caution.]"

---

## Interpretation Notes

> **Scope**: Regional and environmental associations are **observational** and reflect patterns in the analyzed dataset. Results do not establish causation between geographic/environmental factors and resistance profiles. Temporal and confounding factors were not controlled.

### PCA Variance Interpretation

> **Important**: When cumulative variance explained by PC1+PC2 is less than 50%, this should be acknowledged as a limitation. The 2D plots represent simplified views of a higher-dimensional resistance space. While still useful for visualization, patterns observed may not fully represent the underlying data structure.

---

## Document Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial results template |
| 1.1 | 2024 | Added PCA explained variance reporting and interpretation guidelines |

---

*This document is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*

**See also**: [multivariate_analysis.md](../methods/multivariate_analysis.md) | [phase5_synthesis.md](phase5_synthesis.md)
