# Phase 8 — Documentation & Reporting

## AMR Thesis Project: Technical Documentation

This document provides comprehensive technical documentation for the AMR Pattern Recognition pipeline, covering preprocessing decisions, clustering parameters, and model evaluation metrics.

> **See also**: [METHODOLOGY.md](METHODOLOGY.md) for comprehensive research methodology and analytical framework documentation.
>
> **See also**: [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive system architecture and design documentation.

---

## Table of Contents

1. [Preprocessing Decisions](#1-preprocessing-decisions)
2. [Clustering Parameters](#2-clustering-parameters)
3. [Model Evaluation Metrics](#3-model-evaluation-metrics)
4. [Terminology Glossary](#4-terminology-glossary)

---

## 1. Preprocessing Decisions

### 1.1 Data Ingestion (Phase 2.1)

**Purpose**: Load, merge, and consolidate AMR data from multiple CSV files.

| Decision | Rationale |
|----------|-----------|
| Parse isolate codes for metadata | Enables regional and environmental analysis by extracting national site, local site, sample source from standardized naming conventions |
| Extract region from filename | Preserves geographic context for regional analysis |
| Standardize antibiotic abbreviations | Ensures consistency across different data sources |

### 1.2 Data Cleaning (Phase 2.2 & 2.3)

**Purpose**: Clean and standardize resistance data for structure identification.

| Decision | Value | Rationale |
|----------|-------|-----------|
| Minimum antibiotic coverage | 50% | Antibiotics tested in fewer than 50% of isolates are excluded to ensure robust pattern discrimination |
| Maximum missing data per isolate | 50% | Isolates with >50% missing values are excluded to maintain data quality |
| Duplicate removal | By CODE | Removes exact duplicate entries based on isolate code |
| Resistance value standardization | S, I, R only | Converts variations (*R, SUSCEPTIBLE, etc.) to standard three-category format |

**Species Name Standardization**:
- Variants like "E. coli", "e.coli" → "Escherichia coli"
- "Klebsiella pneumoniae ssp pneumoniae" → "Klebsiella pneumoniae"

### 1.3 Resistance Encoding (Phase 2.4)

**Purpose**: Convert categorical resistance values to numerical format for structure identification and pattern discrimination.

| Original Value | Encoded Value | Description |
|---------------|---------------|-------------|
| S (Susceptible) | 0 | No resistance detected |
| I (Intermediate) | 1 | Intermediate resistance |
| R (Resistant) | 2 | Full resistance |

**Encoding Rationale**: Ordinal encoding preserves the biological meaning of resistance levels, enabling meaningful distance calculations for clustering.

### 1.4 Feature Engineering (Phase 2.5)

**Derived Features**:

| Feature | Formula | Description |
|---------|---------|-------------|
| MAR Index | Resistant antibiotics / Total tested | Multiple Antibiotic Resistance index |
| Resistance Count | Count of R values | Total number of resistant antibiotics |
| Resistant Classes Count | Count of resistant antibiotic classes | Number of distinct antibiotic classes showing resistance |
| MDR Flag | Classes ≥ 3 | Multi-Drug Resistant if resistant to ≥3 antibiotic classes |

**Antibiotic Class Definitions** (for MDR calculation):

| Class | Antibiotics |
|-------|-------------|
| Penicillins | AM, AMP |
| β-lactam/β-lactamase inhibitor | AMC, PRA |
| Cephalosporins-1st gen | CN, CF |
| Cephalosporins-3rd/4th gen | CPD, CTX, CFT, CPT |
| Cephamycins | CFO |
| Carbapenems | IPM, MRB |
| Aminoglycosides | AN, GM, N |
| Quinolones/Fluoroquinolones | NAL, ENR |
| Tetracyclines | DO, TE |
| Nitrofurans | FT |
| Phenicols | C |
| Folate pathway inhibitors | SXT |

---

## 2. Clustering Parameters

### 2.1 Hierarchical Clustering (Phase 3.1)

**Purpose**: Unsupervised structure identification to discover natural groupings in resistance patterns.

| Parameter | Default Value | Options | Rationale |
|-----------|---------------|---------|-----------|
| Linkage Method | Ward | ward, complete, average, single | Ward's method minimizes within-cluster variance, suitable for compact clusters |
| Distance Metric | Euclidean | euclidean, manhattan | Euclidean distance with ordinal encoding captures resistance pattern similarity |
| Number of Clusters | 5 | User-configurable | Default based on typical AMR pattern diversity; can be optimized |
| Missing Value Imputation | Median | mean, median, most_frequent | Median is robust to outliers in resistance data |

### 2.2 Cluster Assignment Criteria

| Method | Parameter | Description |
|--------|-----------|-------------|
| `maxclust` | n_clusters | Cut dendrogram to form specified number of clusters |
| `distance` | threshold | Cut dendrogram at specified distance threshold |

### 2.3 Cluster Quality Metrics

The pipeline analyzes cluster quality for k=2 to k=10 clusters:

- **Cluster sizes**: Distribution of isolates across clusters
- **Size standard deviation**: Measure of cluster balance
- **Inconsistency coefficients**: From scipy's inconsistent() function

---

## 3. Model Evaluation Metrics

### 3.1 Supervised Learning for Pattern Discrimination (Phase 4)

**Objective**: Evaluate how well resistance fingerprints discriminate known categories (species, MDR status).

> **Important Note**: This is pattern discrimination, NOT forecasting or prediction of future outcomes. Metrics quantify how consistently resistance patterns align with known categories.

### 3.2 Data Splitting

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Train-Test Split | 80%-20% | Standard split for model evaluation |
| Stratification | By target | Preserves class distribution in both sets |
| Random State | 42 | Reproducibility |

**Purpose of splitting**: Assess model generalization, avoid overfitting, support robustness of pattern discrimination.

### 3.3 Models Evaluated

| Model | Key Hyperparameters |
|-------|---------------------|
| Random Forest | n_estimators=100, random_state=42 |
| Support Vector Machine | kernel='rbf', random_state=42 |
| k-Nearest Neighbors | n_neighbors=5 |
| Logistic Regression | max_iter=1000, random_state=42 |
| Decision Tree | random_state=42 |
| Naive Bayes | Gaussian |

### 3.4 Model Evaluation Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Accuracy** | Overall correct classifications / Total samples | General performance measure |
| **Precision** | True Positives / (True Positives + False Positives) | How many predicted positives are actually positive |
| **Recall** | True Positives / (True Positives + False Negatives) | How many actual positives are correctly identified |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall |
| **Confusion Matrix** | Cross-tabulation of predicted vs actual | Detailed view of classification performance |

**Averaging Method**: Weighted average (accounts for class imbalance)

### 3.5 Feature Importance Analysis (Phase 4.6)

**Purpose**: Identify antibiotics contributing most to group separation.

| Model Type | Importance Method |
|------------|-------------------|
| Tree-based (RF, DT) | feature_importances_ |
| Linear (LR, SVM) | Absolute coefficient values |

**Interpretation Guidelines**:
- Higher importance scores indicate antibiotics that more consistently differentiate categories
- Findings should be related to:
  - AST (Antimicrobial Susceptibility Testing) results
  - MDR trends
  - Biological plausibility

### 3.6 Interpretation of Results

Model evaluation metrics should be interpreted as:

> "These metrics quantify how consistently resistance patterns align with known categories, NOT predictive performance for future samples."

---

## 4. Terminology Glossary

This section defines key terms used throughout the pipeline, aligned with the project's analytical framework.

### Pattern Discrimination
The use of supervised learning to evaluate how well resistance fingerprints distinguish between known categories (e.g., bacterial species, MDR vs non-MDR). This is distinct from prediction—it assesses pattern consistency within existing data.

### Model Evaluation
The process of quantifying model performance using metrics (accuracy, precision, recall, F1-score) to assess how well resistance patterns align with known categories. Used to compare different classification approaches and identify the most suitable model for pattern discrimination.

### Structure Identification
The process of discovering natural groupings and patterns within resistance data through unsupervised methods (hierarchical clustering). This reveals the underlying organization of resistance profiles without pre-defined categories, enabling identification of resistance archetypes and clusters.

### Resistance Fingerprint
A unique pattern of antibiotic susceptibility results for each isolate, encoded numerically (S=0, I=1, R=2) and used as input features for both structure identification (clustering) and pattern discrimination (supervised learning).

### Multi-Drug Resistant (MDR)
An isolate showing resistance to ≥3 antibiotic classes, as defined by CDC/CLSI guidelines.

### MAR Index
Multiple Antibiotic Resistance Index: the ratio of antibiotics an isolate is resistant to over total antibiotics tested. Values range from 0 (fully susceptible) to 1 (fully resistant).

---

## Document Version

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024 | AMR Research Team | Initial documentation |

---

*This documentation is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*
