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

## Interpretation Notes

> **Scope**: These preprocessing results describe the data preparation steps performed on the analyzed dataset. Decisions regarding exclusion thresholds and encoding schemes follow established standards (see [preprocessing.md](../methods/preprocessing.md)).

---

## Document Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial results template |

---

*This document is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*

**See also**: [preprocessing.md](../methods/preprocessing.md) | [phase3_discrimination.md](phase3_discrimination.md)
