# Methodology

## Antimicrobial Resistance Pattern Recognition and Surveillance Pipeline

This document provides a comprehensive methodology for the AMR (Antimicrobial Resistance) pattern recognition pipeline, detailing the research design, data collection procedures, analytical approaches, and validation strategies employed in this thesis project.

---

## Table of Contents

1. [Research Design](#1-research-design)
2. [Data Collection and Sources](#2-data-collection-and-sources)
3. [Data Preprocessing (Phase 2)](#3-data-preprocessing-phase-2)
4. [Unsupervised Structure Identification (Phase 3)](#4-unsupervised-structure-identification-phase-3)
5. [Supervised Learning for Pattern Discrimination (Phase 4)](#5-supervised-learning-for-pattern-discrimination-phase-4)
6. [Regional and Environmental Analysis (Phase 5)](#6-regional-and-environmental-analysis-phase-5)
7. [Integration and Synthesis (Phase 6)](#7-integration-and-synthesis-phase-6)
8. [Quality Control and Validation](#8-quality-control-and-validation)
9. [Software and Tools](#9-software-and-tools)
10. [Limitations and Considerations](#10-limitations-and-considerations)
11. [Ethical Considerations](#11-ethical-considerations)

---

## 1. Research Design

### 1.1 Study Objectives

This study implements a comprehensive analytical pipeline for antimicrobial resistance (AMR) surveillance with the following objectives:

1. **Primary Objective**: Develop and validate a data-driven approach for identifying resistance patterns in bacterial isolates from environmental and clinical samples across multiple Philippine regions.

2. **Secondary Objectives**:
   - Identify natural groupings (clusters) in resistance profiles using unsupervised learning methods
   - Evaluate how resistance fingerprints discriminate between known categories (bacterial species, MDR status)
   - Characterize regional and environmental factors associated with resistance patterns
   - Synthesize findings to identify dominant resistance archetypes and MDR-enriched patterns

### 1.2 Analytical Framework

The methodology follows a structured multi-phase approach:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AMR ANALYSIS PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 2: Data Preprocessing                                        │
│  ├── 2.1 Data Ingestion and Consolidation                          │
│  ├── 2.2 Data Cleaning                                              │
│  ├── 2.3 Missing Data Handling                                      │
│  ├── 2.4 Resistance Encoding                                        │
│  └── 2.5 Feature Engineering                                        │
│                                                                     │
│  Phase 3: Unsupervised Structure Identification                     │
│  ├── 3.1 Hierarchical Agglomerative Clustering                     │
│  ├── 3.2 Visualization of Resistance Patterns                      │
│  └── 3.3 Cluster Interpretation                                     │
│                                                                     │
│  Phase 4: Supervised Learning for Pattern Discrimination            │
│  ├── 4.1 Objective Definition                                       │
│  ├── 4.2 Data Splitting                                             │
│  ├── 4.3 Model Selection                                            │
│  ├── 4.4 Model Training                                             │
│  ├── 4.5 Model Evaluation                                           │
│  └── 4.6 Feature Importance Analysis                                │
│                                                                     │
│  Phase 5: Regional and Environmental Analysis                       │
│  ├── 5.1 Cluster Distribution Analysis                              │
│  └── 5.2 Principal Component Analysis (PCA)                         │
│                                                                     │
│  Phase 6: Integration and Synthesis                                 │
│  ├── 6.1 Cluster-Supervised Comparison                              │
│  ├── 6.2 Resistance Archetype Identification                        │
│  ├── 6.3 Species-Environment Association Analysis                   │
│  └── 6.4 MDR-Enriched Pattern Identification                        │
│                                                                     │
│  Phase 7: Interactive Dashboard (Streamlit)                         │
│                                                                     │
│  Phase 8: Documentation and Reporting                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Terminology

| Term | Definition |
|------|------------|
| **Pattern Discrimination** | Use of supervised learning to evaluate how well resistance fingerprints distinguish between known categories; distinct from prediction—assesses pattern consistency within existing data |
| **Structure Identification** | Process of discovering natural groupings in resistance data through unsupervised methods without pre-defined categories |
| **Resistance Fingerprint** | Unique pattern of antibiotic susceptibility results for each isolate, encoded numerically and used as input features |
| **Multi-Drug Resistant (MDR)** | Isolate showing resistance to ≥3 antibiotic classes (CDC/CLSI definition) |
| **MAR Index** | Multiple Antibiotic Resistance Index: ratio of resistant antibiotics to total tested |

---

## 2. Data Collection and Sources

### 2.1 Study Sites and Regions

Data were collected from three geographic regions in the Philippines:

| Region | Code | Sites |
|--------|------|-------|
| **BARMM** (Bangsamoro Autonomous Region in Muslim Mindanao) | M | APMC, Dayawan, Gadongan, Tuca Kialdan |
| **Region III - Central Luzon** | P | San Gabriel, San Roque |
| **Region VIII - Eastern Visayas** | O | Alegria, Larrazabal, OD Hospital |

### 2.2 Sample Sources

Environmental and clinical samples were collected from various sources:

| Code | Sample Source | Description |
|------|---------------|-------------|
| DW | Drinking Water | Potable water samples |
| LW | Lake Water | Surface water from lakes |
| RW | River Water | Surface water from rivers |
| FB | Fish Banak | Fish samples (Banak species) |
| FG | Fish Gusaw | Fish samples (Gusaw species) |
| FT | Fish Tilapia | Fish samples (Tilapia) |
| FK | Fish Kaolang | Fish samples (Kaolang species) |
| EWU | Effluent Water Untreated | Hospital effluent (pre-treatment) |
| EWT | Effluent Water Treated | Hospital effluent (post-treatment) |

### 2.3 Isolate Identification Convention

Each bacterial isolate follows a standardized naming convention:

```
[Species Prefix]_[National Site][Local Site][Sample Source][Replicate][Colony]

Example: EC_OADWR1C3
├── EC    = Escherichia coli
├── O     = Ormoc (National Site)
├── A     = Alegria (Local Site)
├── DW    = Drinking Water (Sample Source)
├── R1    = Replicate 1
└── C3    = Colony 3
```

### 2.4 Antimicrobial Susceptibility Testing (AST)

#### 2.4.1 Antibiotics Tested

A panel of 23 antibiotics across 12 classes was tested:

| Antibiotic Class | Antibiotics (Abbreviation) |
|------------------|---------------------------|
| Penicillins | Ampicillin (AM/AMP) |
| β-lactam/β-lactamase inhibitor combinations | Amoxicillin-Clavulanate (AMC), Piperacillin-Tazobactam (PRA) |
| Cephalosporins (1st generation) | Cephalothin (CN), Cefazolin (CF) |
| Cephalosporins (3rd/4th generation) | Cefpodoxime (CPD), Cefotaxime (CTX), Ceftriaxone (CFT), Cefepime (CPT) |
| Cephamycins | Cefoxitin (CFO) |
| Cephalosporin/BLI combinations | Ceftazidime-Avibactam (CZA) |
| Carbapenems | Imipenem (IPM), Meropenem (MRB) |
| Aminoglycosides | Amikacin (AN), Gentamicin (GM), Neomycin (N) |
| Quinolones/Fluoroquinolones | Nalidixic Acid (NAL), Enrofloxacin (ENR) |
| Tetracyclines | Doxycycline (DO), Tetracycline (TE) |
| Nitrofurans | Nitrofurantoin (FT) |
| Phenicols | Chloramphenicol (C) |
| Folate pathway inhibitors | Trimethoprim-Sulfamethoxazole (SXT) |

#### 2.4.2 Interpretation Categories

Results are categorized following CLSI guidelines:

| Category | Code | Description |
|----------|------|-------------|
| Susceptible | S | No resistance detected |
| Intermediate | I | Intermediate resistance |
| Resistant | R | Full resistance detected |

---

## 3. Data Preprocessing (Phase 2)

### 3.1 Data Ingestion and Consolidation (Phase 2.1)

#### 3.1.1 Objectives

- Load and merge AST data from multiple CSV files
- Extract metadata from filenames and isolate codes
- **Enforce explicit metadata standardization** for traceability and reproducibility
- Standardize antibiotic abbreviations across sources
- Create a unified raw dataset with validated metadata

#### 3.1.2 Required Metadata Columns

The following metadata columns are required at ingestion for methodological validity:

| Column | Description | Source |
|--------|-------------|--------|
| **REGION** | Geographic region (e.g., BARMM, Region VIII) | Extracted from filename |
| **SITE** | Specific sampling site within region | Extracted from filename |
| **ENVIRONMENT** | Environmental category (Water, Fish, Hospital) | Derived from sampling source |
| **SAMPLING_SOURCE** | Detailed sampling source (e.g., Drinking Water) | Parsed from isolate code |

#### 3.1.3 Procedures

1. **CSV File Parsing**: Each CSV file contains structured data with:
   - Header rows identifying CODE, ISOLATE ID, and summary columns
   - Antibiotic row listing tested antibiotics (AM, AMC, etc.)
   - MIC/INT row labels
   - Data rows with isolate-level AST results

2. **Metadata Extraction**:
   ```python
   # From filename: Region and Site
   "1NET_P2-AMR_Region VIII-Eastern Visayas - Copy - LOR-ALEGRIA.csv"
   → Region: "Region VIII - Eastern Visayas", Site: "ALEGRIA"
   
   # From isolate code: National site, Local site, Sample source, Environment
   "EC_OADWR1C3"
   → National: Ormoc, Local: Alegria, Source: Drinking Water, Environment: Water
   ```

3. **Environment Categorization**:

| Sampling Source | Environment Category |
|-----------------|---------------------|
| Drinking Water, Lake Water, River Water | Water |
| Fish Banak, Fish Gusaw, Fish Tilapia, Fish Kaolang | Fish |
| Effluent Water Untreated, Effluent Water Treated | Hospital |

4. **Antibiotic Standardization**: Variant names are mapped to standard abbreviations.

5. **Metadata Validation**: Coverage statistics are computed for all required columns.

#### 3.1.4 Output

- `unified_raw_dataset.csv`: Consolidated dataset with all isolates and validated metadata

### 3.2 Data Cleaning (Phase 2.2 & 2.3)

#### 3.2.1 Objectives

- **Validate resistance values** (only S, I, R allowed; no multi-label entries)
- Standardize species names and resistance values using **controlled vocabularies**
- **Detect and resolve duplicate isolates** with detailed logging
- Implement **formal missing data strategy** with transparent, defensible methodology
- Filter antibiotics and isolates based on coverage thresholds
- Generate comprehensive exclusion summary table

#### 3.2.2 Validation Rules

| Rule | Description | Action |
|------|-------------|--------|
| Valid Values | Only {S, I, R} allowed | Invalid values set to NULL |
| Multi-label Entries | No "S/R", "S,I", etc. | Invalid values set to NULL |

#### 3.2.3 Controlled Vocabularies

**Species Name Standardization:**

| Variant | Standardized Name |
|---------|-------------------|
| E. coli, e.coli | Escherichia coli |
| Klebsiella pneumoniae ssp pneumoniae | Klebsiella pneumoniae |
| Enterobacter cloacae complex | Enterobacter cloacae |

**Antibiotic Standardization:**

| Full Name | Standard Abbreviation |
|-----------|----------------------|
| Ampicillin | AM |
| Amoxicillin-Clavulanate | AMC |
| Gentamicin | GM |
| ... | ... |

#### 3.2.4 Formal Missing Data Strategy

This methodology converts implicit missing data handling into a transparent, defensible strategy.

| Parameter | Default Threshold | Rationale |
|-----------|-------------------|-----------|
| **Minimum antibiotic coverage** | ≥70% | Antibiotics tested in fewer than 70% of isolates are excluded to ensure robust pattern discrimination |
| **Maximum missing data per isolate** | ≤30% | Isolates with >30% missing AST values are excluded to maintain data quality |

**Procedure:**
1. Compute antibiotic test coverage (% of isolates tested for each antibiotic)
2. Retain antibiotics tested in ≥70% of isolates
3. Remove isolates exceeding 30% missing-value threshold
4. Generate exclusion summary table documenting all decisions

#### 3.2.5 Duplicate Detection and Resolution

Duplicates are identified by the `CODE` column and removed, keeping the first occurrence. All duplicate removals are logged with:
- Index and CODE of removed record
- Reason for removal

#### 3.2.6 Output

- `cleaned_dataset.csv`: Cleaned and standardized data
- `cleaning_report.txt`: Comprehensive documentation including:
  - Thresholds applied
  - Data retention summary
  - Validation summary
  - Antibiotic test coverage table
  - Exclusion summary table
  - Cleaning actions log

### 3.3 Resistance Encoding (Phase 2.4)

#### 3.3.1 Objective

Convert categorical resistance values to numerical format for quantitative analysis.

#### 3.3.2 Encoding Scheme

| Original Value | Encoded Value | Biological Interpretation |
|----------------|---------------|---------------------------|
| S (Susceptible) | 0 | No resistance detected |
| I (Intermediate) | 1 | Intermediate resistance |
| R (Resistant) | 2 | Full resistance |

#### 3.3.3 Rationale

Ordinal encoding preserves the biological meaning of resistance levels and enables:
- Meaningful distance calculations for clustering
- Numerical operations for supervised learning
- Calculation of resistance indices

#### 3.3.4 Output

- `encoded_dataset.csv`: Numerically encoded resistance data
- New columns: `{ANTIBIOTIC}_encoded` for each antibiotic

### 3.4 Feature Engineering (Phase 2.5)

#### 3.4.1 Objective

Derive clinically relevant features from encoded resistance data using **formalized definitions** with explicit citations.

#### 3.4.2 MAR Index (Multiple Antibiotic Resistance Index)

**Formula:**
```
MAR = a / b

Where:
  a = Number of antibiotics to which the isolate is resistant (R)
  b = Total number of antibiotics tested on the isolate
```

**Reference:** Krumperman PH. (1983). Multiple antibiotic resistance indexing of Escherichia coli to identify high-risk sources of fecal contamination of foods. *Applied and Environmental Microbiology*, 46(1), 165-170.

**Interpretation:**
- MAR > 0.2 indicates isolates from high-risk contamination sources
- MAR = 0 indicates fully susceptible isolate
- MAR = 1 indicates pan-resistant isolate

#### 3.4.3 MDR Classification

**Definition:** An isolate is classified as Multi-Drug Resistant (MDR) if it exhibits resistance to at least one agent in **≥3 antimicrobial categories**.

**Reference:** Magiorakos AP, et al. (2012). Multidrug-resistant, extensively drug-resistant and pandrug-resistant bacteria: an international expert proposal for interim standard definitions for acquired resistance. *Clinical Microbiology and Infection*, 18(3), 268-281. DOI: 10.1111/j.1469-0691.2011.03570.x

**Antimicrobial Categories:**

| Category | Antibiotics |
|----------|-------------|
| Penicillins | AM, AMP |
| β-lactam/BLI combinations | AMC, PRA |
| Cephalosporins (1st generation) | CN, CF |
| Cephalosporins (3rd/4th generation) | CPD, CTX, CFT, CPT |
| Cephamycins | CFO |
| Cephalosporin/BLI combinations | CZA |
| Carbapenems | IPM, MRB |
| Aminoglycosides | AN, GM, N |
| Quinolones/Fluoroquinolones | NAL, ENR |
| Tetracyclines | DO, TE |
| Nitrofurans | FT |
| Phenicols | C |
| Folate pathway inhibitors | SXT |

#### 3.4.4 Derived Features Summary

| Feature | Formula | Description |
|---------|---------|-------------|
| **MAR_INDEX_COMPUTED** | a / b (Krumperman, 1983) | Multiple Antibiotic Resistance index (0-1) |
| **RESISTANCE_COUNT** | Count where encoded = 2 | Total number of resistant antibiotics |
| **RESISTANT_CLASSES_COUNT** | Count of unique resistant classes | Number of antimicrobial categories with resistance |
| **MDR_FLAG** | Boolean: Classes ≥ 3 (Magiorakos, 2012) | Multi-Drug Resistant indicator |
| **MDR_CATEGORY** | "MDR" or "Non-MDR" | Categorical MDR status |
| **{AB}_RESISTANT** | Binary: 1 if R, 0 if S/I | Per-antibiotic binary resistance indicator |

#### 3.4.5 Structural Data Separation

To improve pipeline clarity and downstream modeling safety, the feature engineering phase produces physically separated outputs:

| Output | Description | Purpose |
|--------|-------------|---------|
| `analysis_ready_dataset.csv` | Full combined dataset | Complete data for reference |
| `feature_matrix_X.csv` | Encoded resistance values only | Input for clustering/ML models |
| `metadata.csv` | Sample identification and derived features | Interpretation and stratification |

This separation prevents accidental use of metadata as model features.

---

## 4. Unsupervised Structure Identification (Phase 3)

### 4.1 Hierarchical Agglomerative Clustering (Phase 3.1)

#### 4.1.1 Objective

Discover natural groupings in resistance patterns through unsupervised learning without pre-defined categories, enabling identification of resistance archetypes and clusters.

#### 4.1.2 Data Preparation

1. **Feature Selection**: Encoded resistance columns (`{AB}_encoded`)
2. **Missing Value Imputation**: Median imputation strategy
   - Rationale: Median is robust to outliers in resistance data

#### 4.1.3 Clustering Parameters

| Parameter | Default Value | Options | Rationale |
|-----------|---------------|---------|-----------|
| **Linkage Method** | Ward | ward, complete, average, single | Ward's method minimizes within-cluster variance, suitable for compact clusters |
| **Distance Metric** | Euclidean | euclidean, manhattan | Euclidean distance with ordinal encoding captures resistance pattern similarity |
| **Number of Clusters** | 5 | User-configurable | Default based on typical AMR pattern diversity |

#### 4.1.4 Clustering Algorithm

```
1. Initialize: Each isolate is its own cluster
2. Compute pairwise distances between all clusters
3. Merge the two closest clusters based on linkage criterion
4. Update distance matrix
5. Repeat steps 3-4 until desired number of clusters
6. Cut dendrogram at specified level to assign cluster labels
```

#### 4.1.5 Cluster Assignment Methods

| Method | Parameter | Description |
|--------|-----------|-------------|
| `maxclust` | n_clusters | Cut dendrogram to form specified number of clusters |
| `distance` | threshold | Cut dendrogram at specified distance threshold |

#### 4.1.6 Cluster Quality Assessment

The pipeline analyzes cluster quality for k=2 to k=10 clusters:

- **Cluster sizes**: Distribution of isolates across clusters
- **Size standard deviation**: Measure of cluster balance
- **Inconsistency coefficients**: Statistical measure of clustering validity

#### 4.1.7 Output

- `clustered_dataset.csv`: Dataset with `CLUSTER` column assignments
- Linkage matrix for dendrogram visualization

### 4.2 Visualization of Resistance Patterns (Phase 3.2)

#### 4.2.1 Generated Visualizations

| Visualization | Description | Purpose |
|---------------|-------------|---------|
| **Dendrogram** | Hierarchical tree structure | Visualize clustering hierarchy and optimal cut points |
| **Resistance Heatmap** | Color-coded matrix of resistance values | Identify resistance patterns across isolates |
| **Cluster Distribution** | Bar chart of cluster sizes | Assess cluster balance |
| **Cluster Profiles** | Mean resistance per cluster | Characterize cluster archetypes |

### 4.3 Cluster Interpretation (Phase 3.3)

#### 4.3.1 Cluster Summary Statistics

For each cluster, the following are computed:

- Number of isolates and percentage of total
- MDR proportion within cluster
- Mean MAR index
- Species composition
- Regional distribution
- Environmental/sample source distribution

---

## 5. Supervised Learning for Pattern Discrimination (Phase 4)

### 5.1 Objective (Phase 4.1)

Evaluate how well resistance fingerprints discriminate known categories through **two independent tasks**:

| Task | Input | Target | Type |
|------|-------|--------|------|
| **Task A: Species discrimination** | Resistance fingerprints | Species | Multi-class |
| **Task B: MDR discrimination** | Resistance fingerprints | MDR flag | Binary |

> **Important**: This is **pattern discrimination**, NOT forecasting or prediction of future outcomes. The metrics quantify how consistently resistance patterns align with known categories.

> **MDR Target Transparency**: The MDR label is derived from the SAME AST features used as input. MDR discrimination is treated as **self-consistency discrimination** - evaluating how consistently resistance fingerprints align with MDR status. This explicitly acknowledges the "predicted MDR from MDR" relationship.

### 5.2 Data Splitting (Phase 4.2) - LEAKAGE-SAFE

#### 5.2.1 Train-Test Split Discipline

**CRITICAL**: Train-test split is performed **BEFORE** any preprocessing to prevent data leakage.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Training Set | 80% | Sufficient data for model learning |
| Test Set | 20% | Independent evaluation of pattern consistency |
| Stratification | By target variable | Preserves class distribution in both sets |
| Random State | 42 (fixed and reported) | Reproducibility |

#### 5.2.2 Leakage-Safe Preprocessing Order

1. **Split FIRST**: 80/20 train-test split (stratified)
2. **Imputation**: Median strategy fit on TRAIN only, applied to both
3. **Scaling**: StandardScaler fit on TRAIN only, applied to both

This prevents information from the test set from influencing preprocessing.

#### 5.2.3 Strict Feature-Label Separation

- **Input matrix**: Resistance fingerprints ONLY (`{AB}_encoded` columns)
- **Excluded**: All metadata (region, site, environment, sample source)
- **Rationale**: Prevents silent contextual leakage

### 5.3 Model Selection (Phase 4.3) - RATIONALIZED SET

**Reduced to 3-4 models** with clear methodological justification:

| Model | Category | Key Hyperparameters | Purpose |
|-------|----------|---------------------|---------|
| **Logistic Regression** | Linear | max_iter=1000, random_state=42 | Linear baseline with coefficient interpretation |
| **Random Forest** | Tree-based | n_estimators=100, random_state=42 | Nonlinear model with Gini feature importance |
| **k-Nearest Neighbors** | Distance-based | n_neighbors=5 | Distance-based consistency check |

#### Model Category Descriptions

| Category | Description |
|----------|-------------|
| **Linear** | Models that learn linear decision boundaries. Interpretable through coefficient magnitudes. |
| **Tree-based** | Models that partition feature space using decision rules. Provide Gini-based importance. |
| **Distance-based** | Models that classify based on similarity to training instances. No parameters to interpret. |

### 5.4 Model Training (Phase 4.4)

#### 5.4.1 Feature Preprocessing (Applied AFTER Split)

1. **Missing Value Imputation**: Median strategy (fit on train only)
2. **Label Encoding**: Target variables encoded numerically
3. **Feature Scaling**: StandardScaler normalization (fit on train only)

#### 5.4.2 Input Features

- **Features**: Encoded resistance fingerprints (`{AB}_encoded` columns) ONLY
- **Targets**: Known labels (species identity or MDR category)
- **Excluded**: Metadata columns (no region, site, environment enters the model)

### 5.5 Model Evaluation (Phase 4.5)

#### 5.5.1 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / Total | Overall correct classifications |
| **Precision (Macro)** | Mean of per-class precision | Treats all classes equally |
| **Recall (Macro)** | Mean of per-class recall | Treats all classes equally |
| **F1-Score (Macro)** | Harmonic mean of macro P and R | Primary comparison metric |
| **Confusion Matrix** | Cross-tabulation per task | Detailed view per class |

#### 5.5.2 Averaging Method - MACRO

**Macro averaging** is used to prevent class imbalance bias:
```
Macro_Metric = (1/n_classes) × Σ(class_metric)
```

All classes are treated equally regardless of sample size.

#### 5.5.3 Interpretation Language Discipline

| Avoid | Use Instead |
|-------|-------------|
| "Model performs well" | "Model shows consistent alignment" |
| "Predicts accurately" | "Demonstrates discriminative capacity" |
| "High predictive accuracy" | "Strong pattern consistency" |

### 5.6 Feature Importance Analysis (Phase 4.6)

#### 5.6.1 Objective

Identify antibiotics showing **associative importance** for group separation.

#### 5.6.2 Importance Extraction Methods

| Model | Importance Method |
|-------|-------------------|
| **Random Forest** | Gini importance (mean decrease in impurity) |
| **Logistic Regression** | Absolute coefficient magnitude |
| **k-NN** | No native feature importance available |

#### 5.6.3 Antibiotic-Level Summary Table

| Column | Description |
|--------|-------------|
| Antibiotic | Antibiotic name |
| Importance Score | Numerical importance value |
| Task | Species or MDR |
| Interpretation | Biological context (e.g., "Common MDR marker") |

#### 5.6.4 Biological Restraint

- **Interpret as**: "Associative importance" - the antibiotic helps discriminate groups
- **Avoid**: Causal or mechanistic claims
- **Rationale**: High importance does NOT imply the antibiotic causes group membership

### 5.7 Robustness Checks

#### 5.7.1 Model Agreement Check

Compare top-ranked antibiotics across different models:
- Identifies overlapping important features
- Strengthens confidence in patterns when models agree

#### 5.7.2 Stability Across Random Seeds

- Re-run split with different seeds
- Report consistency qualitatively
- Shows robustness without heavy computation

### 5.8 Task Separation

#### 5.8.1 Separate Pipelines

Each task runs as an **independent experiment**:
- Separate train-test splits
- Separate preprocessing
- Separate model training
- Separate evaluation

#### 5.8.2 Separate Results Sections

For each task, report separately:
- Confusion matrices
- Metric tables
- Feature importance analyses

---

## 6. Regional and Environmental Analysis (Phase 5)

### 6.1 Cluster Distribution Analysis (Phase 5.1)

#### 6.1.1 Objectives

- Analyze how clusters distribute across geographic regions
- Identify environmental factors associated with specific clusters
- Test for statistical associations using chi-square tests

#### 6.1.2 Cross-Tabulation Analysis

Cross-tabulations are computed for:
- **Clusters × Regions**: Geographic distribution of resistance patterns
- **Clusters × Sample Sources**: Environmental associations
- **Clusters × Species**: Species composition of clusters

#### 6.1.3 Statistical Testing

**Chi-square test of independence** is applied to each cross-tabulation:

| Hypothesis | H₀ | H₁ |
|------------|----|----|
| Cluster-Region | Clusters and regions are independent | Clusters differ by region |
| Cluster-Environment | Clusters and sample sources are independent | Clusters differ by environment |
| Cluster-Species | Clusters and species are independent | Clusters differ by species |

Significance threshold: α = 0.05

### 6.2 Principal Component Analysis (Phase 5.2)

#### 6.2.1 Objective

Reduce dimensionality of resistance profiles and visualize patterns in a lower-dimensional space.

#### 6.2.2 Procedures

1. **Data Preparation**:
   - Extract encoded resistance columns
   - Impute missing values (median strategy)
   - Standardize features (StandardScaler)

2. **PCA Computation**:
   - Extract principal components (default: 2)
   - Compute explained variance ratios
   - Calculate component loadings

3. **Loading Interpretation**:
   - Identify antibiotics with highest absolute loadings per component
   - Interpret components in terms of resistance patterns

#### 6.2.3 Visualizations

| Plot Type | Description | Color Coding |
|-----------|-------------|--------------|
| **PCA Scatter Plot** | 2D projection of isolates | By cluster, region, or MDR status |
| **PCA Biplot** | Scatter plot with loading vectors | Shows antibiotic contributions |

---

## 7. Integration and Synthesis (Phase 6)

### 7.1 Cluster-Supervised Comparison (Phase 6.1)

#### 7.1.1 Objective

Evaluate how well unsupervised clusters align with supervised discrimination results.

#### 7.1.2 Methods

1. **Cross-tabulation** of clusters vs. MDR status and species
2. **Cluster Purity** calculation:
   ```
   Purity(cluster) = max(category_count) / cluster_size
   ```
3. **Chi-square tests** for cluster-category associations

#### 7.1.3 Interpretation

- Significant cluster-MDR association: Clustering captures MDR-related patterns
- High cluster purity: Clusters represent homogeneous groups

### 7.2 Resistance Archetype Identification (Phase 6.2)

#### 7.2.1 Definition

An **archetype** is a characteristic resistance profile that defines each cluster.

#### 7.2.2 Archetype Characterization

For each cluster:
1. **Mean resistance profile**: Average encoded value per antibiotic
2. **High resistance antibiotics**: Mean > 1.5 (between Intermediate and Resistant)
3. **Low resistance antibiotics**: Mean < 0.5 (mostly Susceptible)
4. **Overall resistance level classification**:

| Mean Resistance Score | Level |
|-----------------------|-------|
| > 1.5 | High resistance |
| > 1.0 | Moderate-high resistance |
| > 0.5 | Moderate resistance |
| ≤ 0.5 | Low resistance |

### 7.3 Species-Environment Association Analysis (Phase 6.3)

#### 7.3.1 Objective

Identify associations between bacterial species and environmental sources.

#### 7.3.2 Methods

1. **Cross-tabulation** of species vs. sample source and region
2. **Dominant environment** identification for each species
3. **Chi-square tests** for statistical significance

### 7.4 MDR-Enriched Pattern Identification (Phase 6.4)

#### 7.4.1 Objective

Identify clusters, regions, environments, and species with higher-than-expected MDR rates.

#### 7.4.2 Enrichment Analysis

For each grouping variable:
1. Calculate group-specific MDR rate
2. Compare to overall MDR rate
3. Compute **fold enrichment**:
   ```
   Fold_Enrichment = Group_MDR_Rate / Overall_MDR_Rate
   ```

#### 7.4.3 MDR Resistance Signature

Identifies antibiotics most associated with MDR by comparing mean resistance between MDR and Non-MDR isolates.

---

## 8. Quality Control and Validation

### 8.1 Data Quality Checks

| Check | Criteria | Action |
|-------|----------|--------|
| Missing values | >50% per antibiotic | Exclude antibiotic |
| Missing values | >50% per isolate | Exclude isolate |
| Duplicates | By CODE column | Remove duplicates |
| Invalid values | Not S, I, R, or null | Set to null |

### 8.2 Model Validation

| Validation Method | Purpose |
|-------------------|---------|
| Train-test split | Assess generalization |
| Stratified sampling | Maintain class balance |
| Multiple model comparison | Identify robust patterns |
| F1-score comparison | Account for class imbalance |

### 8.3 Statistical Significance

| Test | Purpose | Threshold |
|------|---------|-----------|
| Chi-square | Association testing | p < 0.05 |
| Cross-validation | Model stability | - |

---

## 9. Software and Tools

### 9.1 Programming Environment

| Component | Version/Details |
|-----------|-----------------|
| **Language** | Python 3.8+ |
| **Package Manager** | pip |

### 9.2 Core Libraries

| Library | Purpose | Key Functions |
|---------|---------|---------------|
| **pandas** | Data manipulation | DataFrame operations, CSV I/O |
| **numpy** | Numerical computing | Array operations, statistics |
| **scipy** | Scientific computing | Hierarchical clustering, chi-square tests |
| **scikit-learn** | Machine learning | Classification, PCA, preprocessing |
| **matplotlib** | Visualization | Static plots |
| **seaborn** | Statistical visualization | Heatmaps, enhanced plots |
| **streamlit** | Interactive dashboard | Web-based exploration |
| **joblib** | Model persistence | Save/load trained models |

### 9.3 Pipeline Execution

```bash
# Full pipeline
python main.py

# Interactive dashboard
streamlit run app/streamlit_app.py
```

---

## 10. Limitations and Considerations

### 10.1 Study Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Sample size variation across sites | May affect regional comparisons | Chi-square tests with appropriate significance levels |
| Missing AST data | Reduced effective sample size | Threshold-based exclusion, median imputation |
| Cross-sectional design | Cannot infer temporal trends | Focus on pattern identification rather than trends |
| Environmental sampling bias | May not represent true diversity | Standardized sampling protocols |

### 10.2 Methodological Considerations

1. **Pattern Discrimination vs. Prediction**: Results reflect how patterns align with known categories, not predictive accuracy for new samples.

2. **Clustering Subjectivity**: Number of clusters is a parameter that affects interpretation; multiple values should be explored.

3. **Feature Importance Interpretation**: High importance scores indicate discriminative power, not necessarily clinical significance.

4. **Statistical Tests**: Chi-square tests assume sufficient sample sizes; small cells may affect validity.

### 10.3 Scope Boundaries

> ⚠️ **Disclaimer**: This tool is intended for **exploratory pattern recognition and surveillance analysis only**. It should NOT be used for clinical decision support. No patient-level identifiers are processed.

---

## 11. Ethical Considerations

### 11.1 Data Handling

- **Anonymization**: No patient-level identifiers are collected or processed
- **Environmental Focus**: Primary focus on environmental and water samples
- **Data Security**: Processed data stored locally; no external transmission

### 11.2 Research Purpose

This methodology is developed for **academic research purposes** as part of a thesis project on antimicrobial resistance surveillance.

---

## References

1. **CLSI**: Clinical and Laboratory Standards Institute. Performance Standards for Antimicrobial Susceptibility Testing.

2. **CDC**: Centers for Disease Control and Prevention. Antibiotic Resistance Threats in the United States.

3. **Magiorakos AP, et al.** (2012). Multidrug-resistant, extensively drug-resistant and pandrug-resistant bacteria. Clinical Microbiology and Infection.

4. **scipy.cluster.hierarchy**: SciPy Hierarchical Clustering Documentation.

5. **scikit-learn**: Machine Learning in Python, Pedregosa et al., JMLR 12, 2011.

---

## Document Version

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024 | AMR Research Team | Initial methodology document |

---

*This methodology document is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*
