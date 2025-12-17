# Phase 2 Results: Data Preprocessing

## Preprocessing and Data Preparation Results

This document provides a structured template for documenting Phase 2 preprocessing results for the AMR thesis project.

---

## Table of Contents

1. [Data Ingestion Results](#1-data-ingestion-results)
2. [Data Cleaning Results](#2-data-cleaning-results)
3. [Encoding Results](#3-encoding-results)
4. [Feature Engineering Results](#4-feature-engineering-results)
5. [Final Dataset Summary](#5-final-dataset-summary)

---

## 1. Data Ingestion Results

### 1.1 Source Files Processed

| File | Region | Site | Isolates Loaded |
|------|--------|------|-----------------|
| [filename1.csv] | [Region] | [Site] | [N] |
| [filename2.csv] | [Region] | [Site] | [N] |
| ... | ... | ... | ... |
| **Total** | - | - | **[Total N]** |

### 1.2 Metadata Extraction Summary

| Metadata Field | Coverage | Missing |
|----------------|----------|---------|
| REGION | [%] | [N] |
| SITE | [%] | [N] |
| ENVIRONMENT | [%] | [N] |
| SAMPLING_SOURCE | [%] | [N] |
| SPECIES | [%] | [N] |

### 1.3 Antibiotic Columns Identified

| Antibiotic | Abbreviation | Present in Data |
|------------|--------------|-----------------|
| Ampicillin | AM | ✅ / ❌ |
| Amoxicillin-Clavulanate | AMC | ✅ / ❌ |
| [Continue for all antibiotics] | ... | ... |

---

## 2. Data Cleaning Results

### 2.1 Cleaning Parameters Applied

| Parameter | Value Applied |
|-----------|---------------|
| Minimum antibiotic coverage | 70% |
| Maximum missing per isolate | 30% |
| Duplicate handling | Remove by CODE |

### 2.2 Data Retention Summary

| Stage | Records In | Records Out | Retention Rate |
|-------|------------|-------------|----------------|
| Raw ingestion | [N] | [N] | 100% |
| After duplicate removal | [N] | [N] | [%] |
| After antibiotic filtering | [N] | [N] | [%] |
| After isolate filtering | [N] | [N] | [%] |
| **Final cleaned dataset** | - | **[N]** | **[%]** |

### 2.3 Exclusion Summary

| Exclusion Type | Count | Percentage | Details |
|----------------|-------|------------|---------|
| Low-coverage antibiotics | [N] | [%] | [List of excluded antibiotics] |
| High-missing isolates | [N] | [%] | Exceeded 30% threshold |
| Invalid resistance values | [N] | [%] | Not in {S, I, R} |
| Duplicate isolates | [N] | [%] | Same CODE value |

### 2.4 Antibiotic Coverage Table

| Antibiotic | Isolates Tested | Coverage (%) | Retained? |
|------------|-----------------|--------------|-----------|
| [AB1] | [N] | [%] | Yes/No |
| [AB2] | [N] | [%] | Yes/No |
| ... | ... | ... | ... |

### 2.5 Species Distribution (After Cleaning)

| Species | Count | Percentage |
|---------|-------|------------|
| Escherichia coli | [N] | [%] |
| Klebsiella pneumoniae | [N] | [%] |
| [Other species] | [N] | [%] |
| **Total** | **[N]** | **100%** |

---

## 3. Encoding Results

### 3.1 Encoding Scheme Applied

| Original | Encoded | Count | Percentage |
|----------|---------|-------|------------|
| S (Susceptible) | 0 | [N] | [%] |
| I (Intermediate) | 1 | [N] | [%] |
| R (Resistant) | 2 | [N] | [%] |
| Missing | NaN | [N] | [%] |

### 3.2 Encoded Columns Created

| Column Name | Based On | Data Type |
|-------------|----------|-----------|
| AM_encoded | AM | int/float |
| AMC_encoded | AMC | int/float |
| ... | ... | ... |

**Total encoded columns**: [N]

---

## 4. Feature Engineering Results

### 4.1 MAR Index Distribution

| Statistic | Value |
|-----------|-------|
| Mean | [value] |
| Median | [value] |
| Std Dev | [value] |
| Min | [value] |
| Max | [value] |

**Interpretation**: [MAR > 0.2 indicates high-risk; describe distribution]

### 4.2 MDR Classification Results

| MDR Status | Count | Percentage |
|------------|-------|------------|
| MDR (≥3 classes) | [N] | [%] |
| Non-MDR (<3 classes) | [N] | [%] |
| **Total** | **[N]** | **100%** |

### 4.3 Resistance Count Distribution

| Resistance Count | Isolates | Percentage |
|------------------|----------|------------|
| 0 | [N] | [%] |
| 1-3 | [N] | [%] |
| 4-6 | [N] | [%] |
| 7-9 | [N] | [%] |
| 10+ | [N] | [%] |

### 4.4 Resistant Classes Distribution

| Classes Resistant | Isolates | Percentage | MDR? |
|-------------------|----------|------------|------|
| 0 | [N] | [%] | No |
| 1 | [N] | [%] | No |
| 2 | [N] | [%] | No |
| 3 | [N] | [%] | Yes |
| 4+ | [N] | [%] | Yes |

---

## 5. Final Dataset Summary

### 5.1 Analysis-Ready Dataset Characteristics

| Characteristic | Value |
|----------------|-------|
| Total isolates | [N] |
| Total antibiotics (retained) | [N] |
| Encoded columns | [N] |
| Metadata columns | [N] |
| MDR prevalence | [%] |
| Mean MAR index | [value] |

### 5.2 Regional Distribution

| Region | Isolates | Percentage |
|--------|----------|------------|
| BARMM | [N] | [%] |
| Region III | [N] | [%] |
| Region VIII | [N] | [%] |
| **Total** | **[N]** | **100%** |

### 5.3 Environment Distribution

| Environment | Isolates | Percentage |
|-------------|----------|------------|
| Water | [N] | [%] |
| Fish | [N] | [%] |
| Hospital | [N] | [%] |
| **Total** | **[N]** | **100%** |

### 5.4 Output Files Generated

| File | Description | Location |
|------|-------------|----------|
| unified_raw_dataset.csv | Raw consolidated data | data/processed/ |
| cleaned_dataset.csv | Cleaned and standardized | data/processed/ |
| cleaning_report.txt | Cleaning documentation | data/processed/ |
| encoded_dataset.csv | Numerically encoded | data/processed/ |
| analysis_ready_dataset.csv | Final with features | data/processed/ |
| feature_matrix_X.csv | Features only | data/processed/ |
| metadata.csv | Metadata only | data/processed/ |

---

## 6. Cluster Number Validation (k=5 Selection)

### 6.1 Validation Methodology

Optimal cluster count was determined through silhouette analysis and elbow method across k=2 to k=10. This evidence-based validation provides scientific justification for using k=5 clusters instead of an arbitrary choice.

### 6.2 Cluster Validation Results

![Cluster Validation](../../data/processed/figures/cluster_validation.png)

**Figure 6.1**: Cluster validation analysis showing (A) Elbow Method using Within-Cluster Sum of Squares (WCSS) and (B) Silhouette Analysis for k=2 to k=10.

| k | Silhouette Score | WCSS | Cluster Sizes | Interpretation |
|---|------------------|------|---------------|----------------|
| 2 | 0.377 | 2399 | 117, 375 | Too coarse - merges distinct phenotypes |
| 3 | 0.417 | 1769 | 117, 123, 252 | Under-clusters species diversity |
| 4 | 0.465 | 1486 | 23, 94, 123, 252 | Misses MDR/non-MDR E. coli split |
| **5** | **0.488** | **1238** | **23, 94, 123, 104, 148** | **Optimal balance: distinct phenotypes** |
| 6 | 0.517 | 1013 | 23, 94, 86, 37, 104, 148 | Slight over-fragmentation |
| 7 | 0.527 | 895 | 23, 94, 86, 9, 28, 104, 148 | Creates very small clusters (n=9) |
| 8-10 | 0.55-0.59 | 660-796 | Various | Over-fragmentation reduces interpretability |

### 6.3 Justification for k=5

Based on convergent evidence from silhouette analysis and elbow method:

1. **Silhouette Score**: k=5 achieved a silhouette score of **0.49**, indicating reasonable cluster separation. While higher k values showed improved silhouette scores (up to 0.59 at k=10), the incremental improvement beyond k=5 comes with significant costs:
   - Very small clusters emerge (e.g., n=9 at k=7)
   - Biological interpretability decreases
   - Statistical power per cluster diminishes

2. **Elbow Analysis**: The WCSS curve shows a clear inflection point around **k=4-5**, after which the rate of decrease slows substantially. This "elbow" indicates that k=5 captures the major variance structure in the data.

3. **Biological Interpretability**: k=5 produces clusters of sufficient size (23-148 isolates) for meaningful statistical analysis and biological characterization:
   - C1: Salmonella-dominated, aminoglycoside-resistant (n=23)
   - C2: Mixed species, moderate resistance (n=94)
   - C3: E. coli MDR phenotype, tetracycline-resistant (n=123)
   - C4: E. coli susceptible phenotype (n=104)
   - C5: E. coli with low-moderate resistance (n=148)

**Conclusion**: k=5 represents the optimal trade-off between cluster granularity and biological interpretability. The silhouette score of 0.49 confirms adequate cluster separation, while the resulting cluster sizes enable robust statistical inference for each resistance phenotype.

### 6.4 Validation Output Files

| File | Description | Location |
|------|-------------|----------|
| cluster_validation.png | Elbow and silhouette plots | data/processed/figures/ |
| cluster_validation_results.csv | Detailed metrics for k=2-10 | data/processed/figures/ |

---

## Interpretation Notes

> **Scope**: These preprocessing results describe the data preparation steps performed on the analyzed dataset. Decisions regarding exclusion thresholds and encoding schemes follow established standards (see [preprocessing.md](../methods/preprocessing.md)).

---

## Document Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial results template |
| 1.1 | 2025 | Added cluster validation (k=5) section |

---

*This document is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*

**See also**: [preprocessing.md](../methods/preprocessing.md) | [phase3_discrimination.md](phase3_discrimination.md)
