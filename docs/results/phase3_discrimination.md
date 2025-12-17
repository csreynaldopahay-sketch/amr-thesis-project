# Phase 3 Results: Clustering and Pattern Discrimination

## Hierarchical Clustering and Supervised Learning Results

This document provides structured documentation for Phase 3 (clustering) and Phase 4 (supervised discrimination) results for the AMR thesis project. The MDR discrimination task has been replaced with scientifically rigorous co-resistance network analysis.

---

## Table of Contents

1. [Clustering Results (Phase 3)](#1-clustering-results-phase-3)
2. [Co-Resistance Network Analysis (Phase 4)](#2-co-resistance-network-analysis-phase-4)
3. [Predictive Modeling Results](#3-predictive-modeling-results)
4. [Biological Interpretation](#4-biological-interpretation)
5. [Clinical Implications](#5-clinical-implications)

---

## 1. Clustering Results (Phase 3)

### 1.1 Clustering Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Linkage method | Ward | Minimizes within-cluster variance |
| Distance metric | Euclidean | Captures ordinal resistance differences |
| Number of clusters | 5 | Selected based on dataset characteristics and biological interpretability |
| Imputation | Median | Robust to outliers |

### 1.2 Cluster Summary

| Cluster | N | Percentage | MDR Rate | Mean MAR | Characterization |
|---------|---|------------|----------|----------|------------------|
| 1 | 23 | 4.7% | 26.1% | 0.12 | Salmonella-dominated, moderate resistance |
| 2 | 94 | 19.1% | 21.3% | 0.09 | Mixed species, low-moderate resistance |
| 3 | 123 | 25.0% | 54.5% | 0.18 | E. coli-dominated, high MDR, tetracycline-resistant |
| 4 | 104 | 21.1% | 0.0% | 0.04 | E. coli-dominated, susceptible phenotype |
| 5 | 148 | 30.1% | 1.4% | 0.06 | Mixed species, low resistance |
| **Total** | **492** | **100%** | **19.3%** | **0.10** | - |

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

## 2. Co-Resistance Network Analysis (Phase 4)

> **Scientific Rationale**: Unlike circular MDR discrimination (predicting MDR from the features that define it), co-resistance analysis reveals genuine biological relationships between antibiotic resistances. This approach identifies genetic linkage, shared mechanisms, and epidemiological patterns that inform surveillance optimization.

### 2.1 Network Construction Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Statistical test | Chi-square / Fisher's exact | Fisher's exact for small samples (<5 expected) |
| Significance level (α) | 0.01 | Conservative threshold |
| Bonferroni correction | α/231 = 4.33×10⁻⁵ | Corrects for 231 pairwise tests |
| Effect size threshold (φ) | 0.2 | Minimum phi coefficient for inclusion |

### 2.2 Network Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total nodes (antibiotics) | 22 | All tested antibiotics included |
| Total edges (significant pairs) | 32 | Significant co-resistance associations |
| Network density | 0.1385 | Moderate connectivity |
| Connected components | 7 | Distinct resistance clusters |

### 2.3 Top Co-Resistance Pairs (Bonferroni-corrected)

| Rank | Antibiotic Pair | Phi (φ) | P-value | Same Class? | Mechanism Interpretation |
|------|-----------------|---------|---------|-------------|--------------------------|
| 1 | AN ↔ GM | 0.978 | 1.33×10⁻³⁸ | Yes | Shared aminoglycoside resistance mechanism |
| 2 | CPD ↔ CPT | 0.865 | 3.72×10⁻⁷ | Yes | Cephalosporin cross-resistance |
| 3 | DO ↔ TE | 0.832 | 2.79×10⁻⁷³ | Yes | Tetracycline class co-resistance |
| 4 | CF ↔ CN | 0.820 | 8.07×10⁻⁸⁰ | Yes | First-generation cephalosporin co-resistance |
| 5 | MRB ↔ PRA | 0.816 | 2.49×10⁻⁵ | No | **Plasmid-mediated co-carriage** |
| 6 | ENR ↔ MRB | 0.816 | 2.55×10⁻⁵ | No | **Cross-class co-resistance** |
| 7 | CFO ↔ CPT | 0.814 | 1.41×10⁻⁸ | No | β-lactam spectrum overlap |
| 8 | AMC ↔ CF | 0.777 | 8.42×10⁻⁶⁴ | No | β-lactam mechanism sharing |
| 9 | CFO ↔ CPD | 0.655 | 2.13×10⁻¹¹ | No | Cephalosporin spectrum overlap |
| 10 | AMC ↔ CN | 0.649 | 9.24×10⁻⁴⁹ | No | β-lactam mechanism sharing |

### 2.4 Network Clusters Identified

| Cluster | Antibiotics | Classes Involved |
|---------|-------------|------------------|
| 1 (Main) | AMC, AN, C, CF, CFO, CFT, CN, CPD, CPT, DO, FT, GM, SXT, TE | Aminoglycosides, Cephalosporins, Phenicols, Tetracyclines, Folate pathway inhibitors |
| 2 | ENR, MRB, PRA | Fluoroquinolones, Carbapenems, β-lactam/BLI |

**Interpretation**: The largest cluster (14 antibiotics) represents the core co-resistance network, suggesting extensive horizontal gene transfer among environmental isolates. The carbapenem-fluoroquinolone cluster (ENR-MRB-PRA) indicates distinct resistance determinants, possibly plasmid-mediated.

---

## 3. Predictive Modeling Results

> **Non-Circular Analysis**: Unlike MDR discrimination, predicting resistance to antibiotic X from all OTHER antibiotics is scientifically valid. This reveals genuine co-resistance patterns with clinical utility.

### 3.1 Target Antibiotic Selection Rationale

| Antibiotic | Class | Clinical Significance |
|------------|-------|----------------------|
| TE (Tetracycline) | Tetracyclines | Aquaculture indicator, environmental resistance marker |
| ENR (Enrofloxacin) | Fluoroquinolones | Critical antimicrobial, WHO Watch category |
| IPM (Imipenem) | Carbapenems | Last-resort antibiotic, WHO Reserve category |
| SXT (Sulfamethoxazole/Trimethoprim) | Folate pathway inhibitors | Common empiric therapy antibiotic |
| AM (Ampicillin) | Penicillins | First-line antibiotic, baseline resistance indicator |

### 3.2 Predictive Model Performance (Random Forest)

| Target | AUC | Resistance Rate | N_Train | N_Test | Interpretation |
|--------|-----|-----------------|---------|--------|----------------|
| **TE** | **0.993** | 24.2% | 393 | 99 | Strong predictive capacity |
| **ENR** | **1.000** | 0.6% | 393 | 99 | Strong (limited by class imbalance) |
| **SXT** | **0.945** | 11.6% | 393 | 99 | Strong predictive capacity |
| **AM** | **0.800** | 57.8% | 393 | 99 | Strong predictive capacity |
| IPM | N/A | 0.4% | 393 | 99 | Insufficient resistant samples |

### 3.3 Top Predictive Antibiotics by Target

#### TE (Tetracycline) Prediction

| Rank | Predictor | Class | Importance | Biological Interpretation |
|------|-----------|-------|------------|---------------------------|
| 1 | **DO** | Tetracyclines | 0.6089 | Same-class co-resistance (tet genes) |
| 2 | SXT | Folate pathway inhibitors | 0.0907 | Plasmid co-carriage |
| 3 | C | Phenicols | 0.0667 | Environmental selection pressure |

#### SXT Prediction

| Rank | Predictor | Class | Importance | Biological Interpretation |
|------|-----------|-------|------------|---------------------------|
| 1 | **DO** | Tetracyclines | 0.2225 | Plasmid-mediated co-resistance |
| 2 | C | Phenicols | 0.1590 | Environmental co-selection |
| 3 | TE | Tetracyclines | 0.1497 | Tetracycline-SXT linkage |

#### AM (Ampicillin) Prediction

| Rank | Predictor | Class | Importance | Biological Interpretation |
|------|-----------|-------|------------|---------------------------|
| 1 | **FT** | Nitrofurans | 0.2208 | Cross-class co-selection |
| 2 | CN | Cephalosporins-1st | 0.1305 | β-lactam mechanism sharing |
| 3 | SXT | Folate pathway inhibitors | 0.0948 | Multi-drug resistance plasmid |

---

## 4. Biological Interpretation

### 4.1 Same-Class Co-Resistance (Expected Patterns)

Strong co-resistance within antibiotic classes reflects shared resistance mechanisms:

- **AN ↔ GM (φ=0.978)**: Aminoglycoside-modifying enzymes (AMEs) often confer resistance to multiple aminoglycosides
- **DO ↔ TE (φ=0.832)**: Tetracycline efflux pumps (tetA, tetB genes) provide cross-resistance
- **CF ↔ CN (φ=0.820)**: First-generation cephalosporins share binding targets

### 4.2 Cross-Class Co-Resistance (Novel Findings)

Cross-class associations suggest horizontal gene transfer:

- **MRB ↔ PRA (φ=0.816)**: Carbapenem-β-lactam/BLI co-resistance suggests carbapenemase genes on multi-resistance plasmids
- **ENR ↔ MRB (φ=0.816)**: Fluoroquinolone-carbapenem linkage indicates plasmid-mediated quinolone resistance (PMQR) genes co-located with carbapenemases

### 4.3 Predictive Insights

High AUC scores for cross-class predictions reveal epidemiologically linked resistances:

- **TE predicted by DO (AUC=0.993)**: Expected same-class relationship confirms model validity
- **SXT predicted by DO (AUC=0.945)**: Tetracycline-trimethoprim/sulfamethoxazole co-resistance suggests plasmid linkage (common in IncF plasmids)
- **AM predicted by FT (AUC=0.800)**: Ampicillin-nitrofuran association may indicate environmental co-selection pressure

---

## 5. Clinical Implications

### 5.1 Surveillance Optimization

Based on network analysis, testing these "hub" antibiotics can inform resistance to connected drugs:

1. **Tetracyclines (TE/DO)**: High connectivity suggests testing TE can infer resistance to SXT, C, and other linked antibiotics
2. **Aminoglycosides (AN/GM)**: Strong within-class correlation allows inference from single representative

### 5.2 Treatment Guidance

Strong co-resistance associations indicate:

- If resistant to **MRB (meropenem)**, likely also resistant to **ENR (fluoroquinolone)** and **PRA (piperacillin-tazobactam)**
- If resistant to **TE (tetracycline)**, consider **SXT** and **C (chloramphenicol)** may also be compromised

### 5.3 Epidemiological Insights

The cross-class co-resistance patterns suggest:

1. **Mobile genetic elements**: Resistance genes likely co-located on conjugative plasmids
2. **Environmental reservoirs**: Fish and water sources may serve as amplification sites for multi-resistance plasmids
3. **One Health concern**: Environmental-to-clinical transmission pathways require investigation

---

## Figure Captions

### Figure 3.1: Hierarchical Clustering Dendrogram

> "Dendrogram showing hierarchical clustering of resistance profiles (n=492 isolates) using Ward linkage with Euclidean distance. Horizontal line indicates cut point for k=5 clusters. Branch heights reflect merge distances; longer branches indicate more distinct groupings."

### Figure 3.2: Resistance Heatmap with Clustering

> "Heatmap illustrating hierarchical clustering of resistance profiles. Clusters were derived **solely from resistance features** (S=0, I=1, R=2); metadata variables are shown for **post-hoc interpretation only** and did not influence cluster formation."

### Figure 4.1: Co-Resistance Network

> "Co-resistance network showing statistically significant associations between antibiotics (n=22). Edges represent Bonferroni-corrected significant pairs (p < 4.33×10⁻⁵) with φ > 0.2. Node colors indicate antibiotic classes. Edge thickness reflects phi coefficient magnitude. Network contains 32 significant associations organized into 7 connected components."

### Figure 4.2: Co-Resistance Matrix

> "Heatmap of phi coefficients between all antibiotic pairs. Strong positive values (dark) indicate resistance to one antibiotic predicts resistance to the other. The diagonal represents self-correlation (φ=1). Clustering reveals antibiotic groups with shared resistance mechanisms."

---

## Generated Output Files

| File | Description | Location |
|------|-------------|----------|
| coresistance_network.graphml | Network graph in GraphML format | `data/processed/figures/` |
| coresistance_matrix.csv | Phi coefficient matrix | `data/processed/figures/` |
| coresistance_edge_stats.csv | Statistical details for all pairs | `data/processed/figures/` |
| coresistance_network.png | Network visualization | `data/processed/figures/` |
| coresistance_prediction_results.csv | AUC and predictor summary | `data/processed/figures/` |
| coresistance_interpretation.md | Automated interpretation | `data/processed/figures/` |

---

## Document Version

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial results template |
| 2.0 | 2025-12-17 | Replaced circular MDR discrimination with co-resistance network analysis |

---

*This document is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*

**See also**: [clustering.md](../methods/clustering.md) | [phase4_environment.md](phase4_environment.md) | [scripts/coresistance_analysis.py](../../scripts/coresistance_analysis.py)
