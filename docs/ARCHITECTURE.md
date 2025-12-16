# AMR Thesis Project: Comprehensive Architectural Design

## Antimicrobial Resistance Pattern Recognition and Surveillance Pipeline

This document provides a comprehensive architectural design for the AMR (Antimicrobial Resistance) pattern recognition pipeline, covering system architecture, component design, data flow, interfaces, and deployment considerations.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Executive Summary](#2-executive-summary)
3. [System Overview](#3-system-overview)
4. [High-Level Architecture](#4-high-level-architecture)
5. [Data Flow Architecture](#5-data-flow-architecture)
6. [Component Architecture](#6-component-architecture)
7. [Component Breakdown Table](#7-component-breakdown-table)
8. [Data Model and Schemas](#8-data-model-and-schemas)
9. [Module Interfaces](#9-module-interfaces)
10. [Technology Stack](#10-technology-stack)
11. [Deployment Architecture](#11-deployment-architecture)
12. [Non-Functional Considerations](#12-non-functional-considerations)
13. [Design Rationale & Justifications](#13-design-rationale--justifications)
14. [Quality Attributes](#14-quality-attributes)
15. [Appendices](#appendices)

---

## 1. Architecture Overview

### Narrative Description

The AMR Thesis Project implements a **data-driven analytical system** designed for antimicrobial resistance (AMR) surveillance, pattern recognition, and exploratory analysis. The system is specifically architected as a **research platform** for analysis and exploration—**not** for real-time prediction or clinical decision-making.

**System Purpose and Intent**

The architecture supports a complete analytical workflow from raw antimicrobial susceptibility testing (AST) data to interactive exploration and reporting. The system enables researchers to:

- Consolidate and preprocess multi-source AST data from environmental and aquatic samples
- Apply unsupervised learning techniques to discover natural resistance pattern groupings
- Evaluate pattern discrimination capabilities through supervised learning (as read-only analysis, not prediction)
- Analyze regional and environmental associations with resistance patterns
- Interactively explore and visualize results through a web-based dashboard

**Architectural Philosophy**

The design prioritizes **reproducibility**, **modularity**, and **transparency**—qualities essential for academic research and thesis-level work. All analytical models are treated as **read-only artifacts** at deployment; no real-time model updates or predictions occur during dashboard interaction. This separation ensures:

1. **Scientific Reproducibility**: Fixed random states, documented preprocessing decisions, and versioned outputs
2. **Clear Separation of Concerns**: Distinct layers for data, processing, and presentation
3. **Academic Appropriateness**: Realistic scope suitable for thesis-level implementation
4. **Auditability**: Complete documentation of methodological decisions and transformations

**Key Architectural Constraints**

| Constraint | Description |
|------------|-------------|
| **No Real-Time Prediction** | System performs exploratory analysis only; no clinical decision support |
| **Read-Only Models** | Trained models are artifacts loaded for evaluation, not updated at runtime |
| **Local Deployment** | Designed for workstation-level deployment, not cloud infrastructure |
| **Batch Processing** | Data flows through sequential phases; no streaming analytics |
| **Environmental Data Only** | No patient-level identifiers; focuses on environmental/aquatic samples |

---

## 2. Executive Summary

### 2.1 Purpose

The AMR Thesis Project implements a comprehensive analytical pipeline for antimicrobial resistance (AMR) surveillance and pattern recognition. The system processes antimicrobial susceptibility testing (AST) data from bacterial isolates collected across multiple Philippine regions, enabling researchers to:

- Identify natural groupings (clusters) in resistance profiles
- Evaluate pattern discrimination capabilities using supervised learning
- Analyze regional and environmental factors associated with resistance patterns
- Visualize and interact with analysis results through an interactive dashboard

### 2.2 Scope

This architectural design covers:

- **Data Layer**: Data ingestion, storage, and transformation
- **Processing Layer**: Analysis algorithms and machine learning models
- **Presentation Layer**: Visualization and interactive dashboard
- **Infrastructure**: Deployment and runtime environment

### 2.3 Architectural Goals

| Goal | Description |
|------|-------------|
| **Modularity** | Independent, loosely-coupled components that can be developed and tested separately |
| **Extensibility** | Easy addition of new analysis methods, visualizations, and data sources |
| **Reproducibility** | Deterministic results with configurable random states and versioned outputs |
| **Usability** | Clear APIs and interactive interfaces for researchers |
| **Maintainability** | Clean code structure with comprehensive documentation |

---

## 3. System Overview

### 3.1 System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL CONTEXT                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────────────────────────────────┐   │
│  │   Laboratory    │     │         AMR THESIS PROJECT                   │   │
│  │   AST Results   │────▶│  ┌─────────────────────────────────────┐    │   │
│  │   (CSV Files)   │     │  │                                     │    │   │
│  └─────────────────┘     │  │     Pattern Recognition Pipeline    │    │   │
│                          │  │                                     │    │   │
│  ┌─────────────────┐     │  │  • Data Preprocessing               │    │   │
│  │    Research     │◀───▶│  │  • Structure Identification        │    │   │
│  │    Personnel    │     │  │  • Pattern Discrimination          │    │   │
│  │                 │     │  │  • Regional Analysis                │    │   │
│  └─────────────────┘     │  │  • Interactive Dashboard            │    │   │
│                          │  │                                     │    │   │
│  ┌─────────────────┐     │  └─────────────────────────────────────┘    │   │
│  │   Generated     │◀────│                                             │   │
│  │   Reports &     │     │  Outputs: CSV files, visualizations,        │   │
│  │   Datasets      │     │  trained models, statistical reports        │   │
│  └─────────────────┘     └─────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 System Boundaries

| Boundary | In Scope | Out of Scope |
|----------|----------|--------------|
| **Data Sources** | CSV files with AST results | Direct laboratory instrument interfaces |
| **Processing** | Batch analysis pipeline | Real-time streaming analytics |
| **Users** | Research personnel | Clinical decision support |
| **Deployment** | Local workstation | Cloud-based infrastructure |

### 3.3 Key Stakeholders

| Stakeholder | Role | Interests |
|-------------|------|-----------|
| **Researchers** | Primary users | Data analysis, pattern discovery |
| **Thesis Advisors** | Reviewers | Methodology validation, reproducibility |
| **Laboratory Staff** | Data providers | Data format compatibility |
| **Future Maintainers** | Developers | Code clarity, documentation |

---

## 4. High-Level Architecture

### 4.1 Architectural Style

The system follows a **Layered Architecture** combined with a **Pipeline Pattern**:

- **Layered Architecture**: Separates concerns into distinct layers (data, processing, presentation)
- **Pipeline Pattern**: Sequential processing phases with well-defined inputs and outputs

### 4.2 System Layers Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Streamlit Dashboard (app/streamlit_app.py)                         │    │
│  │  • Interactive data exploration                                      │    │
│  │  • Visualization rendering                                           │    │
│  │  • Analysis result display                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                          ANALYSIS LAYER                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐    │
│  │  Clustering    │  │  Supervised    │  │  Integration & Synthesis   │    │
│  │  (Phase 3)     │  │  Learning      │  │  (Phase 6)                 │    │
│  │                │  │  (Phase 4)     │  │                            │    │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘    │
│  ┌────────────────┐  ┌────────────────┐                                    │
│  │  Regional      │  │  Visualization │                                    │
│  │  Analysis      │  │  (Phase 3.2)   │                                    │
│  │  (Phase 5)     │  │                │                                    │
│  └────────────────┘  └────────────────┘                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                        PREPROCESSING LAYER                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────┐  │
│  │  Data          │  │  Data          │  │  Resistance    │  │  Feature │  │
│  │  Ingestion     │  │  Cleaning      │  │  Encoding      │  │  Eng.    │  │
│  │  (Phase 2.1)   │  │  (Phase 2.2-3) │  │  (Phase 2.4)   │  │  (2.5)   │  │
│  └────────────────┘  └────────────────┘  └────────────────┘  └──────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                           DATA LAYER                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Raw CSV Files → Processed Datasets → Analysis Results → Models     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Processing Pipeline Sequence

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        AMR ANALYSIS PIPELINE FLOW                             │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│  │Phase 2.1│───▶│Phase 2.2│───▶│Phase 2.4│───▶│Phase 2.5│───▶│Phase 3  │    │
│  │Ingestion│    │Cleaning │    │Encoding │    │Features │    │Cluster  │    │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └────┬────┘    │
│                                                                    │         │
│                                                                    ▼         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│  │Phase 7  │◀───│Phase 6  │◀───│Phase 5  │◀───│Phase 4  │◀───│Phase 3.2│    │
│  │Dashboard│    │Synthesis│    │Regional │    │Supervised│   │Visualize│    │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Flow Architecture

### 5.1 End-to-End Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Raw CSV Files (*.csv)                                                         │
│       │                                                                         │
│       ▼                                                                         │
│   ┌────────────────────────────────────────────────────────────────────┐       │
│   │                    DATA INGESTION (Phase 2.1)                       │       │
│   │  • Load multiple CSV files from directory                           │       │
│   │  • Parse isolate codes for metadata extraction                      │       │
│   │  • Extract region/site from filenames                               │       │
│   │  • Standardize antibiotic abbreviations                             │       │
│   └────────────────────────────────────────────────────────────────────┘       │
│       │                                                                         │
│       ▼                                                                         │
│   unified_raw_dataset.csv                                                       │
│       │                                                                         │
│       ▼                                                                         │
│   ┌────────────────────────────────────────────────────────────────────┐       │
│   │                    DATA CLEANING (Phase 2.2-2.3)                    │       │
│   │  • Standardize species names                                        │       │
│   │  • Standardize resistance values (S, I, R)                          │       │
│   │  • Remove duplicate isolates                                        │       │
│   │  • Filter antibiotics by coverage threshold (≥50%)                  │       │
│   │  • Remove isolates with excessive missing data (>50%)               │       │
│   └────────────────────────────────────────────────────────────────────┘       │
│       │                                                                         │
│       ▼                                                                         │
│   cleaned_dataset.csv + cleaning_report.txt                                     │
│       │                                                                         │
│       ▼                                                                         │
│   ┌────────────────────────────────────────────────────────────────────┐       │
│   │                  RESISTANCE ENCODING (Phase 2.4)                    │       │
│   │  • Encode resistance values: S=0, I=1, R=2                          │       │
│   │  • Generate resistance fingerprints                                 │       │
│   │  • Create encoded columns (*_encoded)                               │       │
│   └────────────────────────────────────────────────────────────────────┘       │
│       │                                                                         │
│       ▼                                                                         │
│   encoded_dataset.csv                                                           │
│       │                                                                         │
│       ▼                                                                         │
│   ┌────────────────────────────────────────────────────────────────────┐       │
│   │                  FEATURE ENGINEERING (Phase 2.5)                    │       │
│   │  • Compute MAR Index (resistant/tested)                             │       │
│   │  • Compute resistance count                                         │       │
│   │  • Count resistant antibiotic classes                               │       │
│   │  • Determine MDR status (≥3 classes)                                │       │
│   │  • Create binary resistance indicators                              │       │
│   └────────────────────────────────────────────────────────────────────┘       │
│       │                                                                         │
│       ▼                                                                         │
│   analysis_ready_dataset.csv ───────────────────────────────────────────────   │
│       │                                                                    │    │
│       ├──────────────────────┬──────────────────────┬───────────────────┐ │    │
│       ▼                      ▼                      ▼                   ▼ │    │
│   ┌──────────┐          ┌──────────┐          ┌──────────┐       ┌──────┴───┐ │
│   │ Phase 3  │          │ Phase 4  │          │ Phase 5  │       │ Phase 7  │ │
│   │Clustering│          │Supervised│          │Regional  │       │Dashboard │ │
│   └────┬─────┘          └────┬─────┘          └────┬─────┘       └──────────┘ │
│        │                     │                     │                          │
│        ▼                     ▼                     ▼                          │
│   clustered_      *.joblib models    figures/*.png                           │
│   dataset.csv                                                                 │
│        │                                                                       │
│        └────────────────────────┬────────────────────────────┘                │
│                                 ▼                                              │
│                    ┌────────────────────────┐                                  │
│                    │  INTEGRATION (Phase 6) │                                  │
│                    │  • Cluster-supervised  │                                  │
│                    │    comparison          │                                  │
│                    │  • Resistance archetypes│                                 │
│                    │  • Species-environment │                                  │
│                    │  • MDR-enriched patterns│                                 │
│                    └────────────────────────┘                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Transformation Summary

| Phase | Input | Transformation | Output |
|-------|-------|----------------|--------|
| **2.1** | Raw CSV files | Parse, extract metadata, consolidate | `unified_raw_dataset.csv` |
| **2.2-2.3** | Unified raw dataset | Clean, standardize, filter | `cleaned_dataset.csv`, `cleaning_report.txt` |
| **2.4** | Cleaned dataset | Encode resistance values (S=0, I=1, R=2) | `encoded_dataset.csv` |
| **2.5** | Encoded dataset | Compute derived features (MAR, MDR) | `analysis_ready_dataset.csv` |
| **3.1** | Analysis-ready dataset | Hierarchical clustering | `clustered_dataset.csv`, linkage matrix |
| **3.2** | Clustered dataset | Generate plots | PNG visualizations in figures/ |
| **4** | Analysis-ready dataset | Train classifiers | `*.joblib` model files |
| **5** | Clustered dataset | PCA, cross-tabulation | PNG plots, statistical results |
| **6** | All results | Synthesize findings | Summary reports and interpretations |

---

## 6. Component Architecture

### 6.1 Component Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           COMPONENT DIAGRAM                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  src/                                                                           │
│  ├── preprocessing/                                                             │
│  │   ├── data_ingestion.py ──────────────────────────────────────────────┐     │
│  │   │   • create_unified_dataset()                                      │     │
│  │   │   • parse_isolate_code()                                          │     │
│  │   │   • process_csv_file()                                            │     │
│  │   │                                                                   │     │
│  │   ├── data_cleaning.py ◀──────────────────────────────────────────────┤     │
│  │   │   • clean_dataset()                                               │     │
│  │   │   • standardize_species_name()                                    │     │
│  │   │   • filter_antibiotics_by_coverage()                              │     │
│  │   │                                                                   │     │
│  │   ├── resistance_encoding.py ◀────────────────────────────────────────┤     │
│  │   │   • create_encoded_dataset()                                      │     │
│  │   │   • encode_resistance_profile()                                   │     │
│  │   │   • get_resistance_fingerprint()                                  │     │
│  │   │                                                                   │     │
│  │   └── feature_engineering.py ◀────────────────────────────────────────┘     │
│  │       • prepare_analysis_ready_dataset()                                     │
│  │       • compute_mar_index()                                                  │
│  │       • determine_mdr_status()                                               │
│  │                                                                              │
│  ├── clustering/                                                                │
│  │   └── hierarchical_clustering.py ◀────────────────────────────────────┐     │
│  │       • run_clustering_pipeline()                                     │     │
│  │       • perform_hierarchical_clustering()                             │     │
│  │       • get_cluster_summary()                                         │     │
│  │                                                                       │     │
│  ├── visualization/                                                      │     │
│  │   └── visualization.py ◀──────────────────────────────────────────────┤     │
│  │       • generate_all_visualizations()                                 │     │
│  │       • create_resistance_heatmap()                                   │     │
│  │       • create_dendrogram()                                           │     │
│  │                                                                       │     │
│  ├── supervised/                                                         │     │
│  │   └── supervised_learning.py ◀────────────────────────────────────────┤     │
│  │       • run_mdr_discrimination()                                      │     │
│  │       • run_all_models()                                              │     │
│  │       • get_feature_importance()                                      │     │
│  │                                                                       │     │
│  └── analysis/                                                           │     │
│      ├── regional_environmental.py ◀─────────────────────────────────────┤     │
│      │   • run_regional_environmental_analysis()                         │     │
│      │   • perform_pca()                                                 │     │
│      │   • analyze_cluster_distribution()                                │     │
│      │                                                                   │     │
│      └── integration_synthesis.py ◀──────────────────────────────────────┘     │
│          • run_integration_synthesis()                                          │
│          • identify_resistance_archetypes()                                     │
│          • identify_mdr_enriched_patterns()                                     │
│                                                                                 │
│  app/                                                                           │
│  └── streamlit_app.py                                                           │
│      • main() - Dashboard entry point                                           │
│      • Analysis view components                                                 │
│                                                                                 │
│  main.py                                                                        │
│  └── run_full_pipeline()                                                        │
│      • Orchestrates all phases                                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Detailed Component Specifications

#### 6.2.1 Data Ingestion Component

**Module**: `src/preprocessing/data_ingestion.py`

**Purpose**: Load, merge, and add metadata to CSV files from different regions and sites.

**Key Functions**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `create_unified_dataset()` | Main entry point for data ingestion | Directory path, output path | DataFrame |
| `load_all_csv_files()` | Load and consolidate all CSVs | Directory path | DataFrame |
| `process_csv_file()` | Process single CSV file | File path | DataFrame |
| `parse_isolate_code()` | Extract metadata from isolate code | Code string | Dict of metadata |
| `extract_region_from_filename()` | Extract region/site from filename | Filename | (region, site) tuple |

**Dependencies**: pandas, numpy, os, re, pathlib

---

#### 6.2.2 Data Cleaning Component

**Module**: `src/preprocessing/data_cleaning.py`

**Purpose**: Clean and standardize resistance data for structure identification.

**Key Functions**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `clean_dataset()` | Main cleaning function | DataFrame | (DataFrame, report dict) |
| `standardize_species_name()` | Standardize bacterial species names | Name string | Standardized name |
| `standardize_resistance_value()` | Convert to S/I/R | Value | Standardized value |
| `filter_antibiotics_by_coverage()` | Filter by coverage threshold | DataFrame, cols, threshold | List of retained cols |
| `generate_cleaning_report()` | Generate text report | Report dict | Report text |

**Configuration Constants**:
- `SPECIES_STANDARDIZATION`: Dictionary mapping variant names to standard names
- `ANTIBIOTIC_STANDARDIZATION`: Dictionary mapping antibiotic name variants

---

#### 6.2.3 Resistance Encoding Component

**Module**: `src/preprocessing/resistance_encoding.py`

**Purpose**: Encode AST results and create resistance fingerprints.

**Key Functions**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `create_encoded_dataset()` | Main encoding function | DataFrame | (DataFrame, info dict) |
| `encode_resistance_value()` | Encode single value | S/I/R | 0/1/2 |
| `encode_resistance_profile()` | Encode all resistance values | DataFrame, cols | Encoded DataFrame |
| `get_resistance_fingerprint()` | Generate fingerprint string | Row, cols | Fingerprint string |

**Constants**:
- `RESISTANCE_ENCODING = {'S': 0, 'I': 1, 'R': 2}`

---

#### 6.2.4 Feature Engineering Component

**Module**: `src/preprocessing/feature_engineering.py`

**Purpose**: Compute MAR index, MDR flag, and other derived features.

**Key Functions**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `prepare_analysis_ready_dataset()` | Main feature engineering | DataFrame | (DataFrame, feature matrix, metadata, info) |
| `compute_mar_index()` | Calculate MAR index | Row, cols | Float (0-1) |
| `compute_resistance_count()` | Count resistant antibiotics | Row, cols | Integer |
| `count_resistant_classes()` | Count resistant classes | Row, cols | Integer |
| `determine_mdr_status()` | Check MDR (≥3 classes) | Row, cols | Boolean |

**Constants**:
- `ANTIBIOTIC_CLASSES`: Dictionary mapping antibiotics to their therapeutic classes

---

#### 6.2.5 Clustering Component

**Module**: `src/clustering/hierarchical_clustering.py`

**Purpose**: Hierarchical agglomerative clustering for structure identification.

**Key Functions**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `run_clustering_pipeline()` | Main clustering pipeline | DataFrame, feature_cols, n_clusters | (DataFrame, linkage, info) |
| `prepare_clustering_data()` | Prepare data with imputation | DataFrame, cols | (np.array, DataFrame) |
| `perform_hierarchical_clustering()` | Execute clustering algorithm | Data array | Linkage matrix |
| `assign_clusters()` | Assign cluster labels | Linkage matrix, n_clusters | Labels array |
| `get_cluster_profiles()` | Calculate mean profiles | DataFrame, cols | Profile DataFrame |
| `get_cluster_summary()` | Summary statistics | DataFrame | Summary dict |

**Algorithm Parameters**:
- Default linkage method: Ward
- Default distance metric: Euclidean
- Default n_clusters: 5
- Imputation strategy: Median

---

#### 6.2.6 Visualization Component

**Module**: `src/visualization/visualization.py`

**Purpose**: Generate heatmaps, dendrograms, and resistance pattern visualizations.

**Key Functions**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `generate_all_visualizations()` | Generate all plots | DataFrame, cols, linkage, output_dir | Dict of paths |
| `create_resistance_heatmap()` | Create heatmap | DataFrame, cols | Figure |
| `create_dendrogram()` | Create dendrogram | Linkage matrix | Figure |
| `create_clustered_heatmap_with_dendrogram()` | Combined visualization | DataFrame, cols, linkage | Figure |
| `create_cluster_profile_heatmap()` | Mean profiles heatmap | Profiles DataFrame | Figure |
| `create_mdr_distribution_plot()` | MDR by group | DataFrame | Figure |
| `create_mar_distribution_plot()` | MAR index by group | DataFrame | Figure |

**Color Scheme**:
- Susceptible (S=0): Green (#4CAF50)
- Intermediate (I=1): Yellow (#FFC107)
- Resistant (R=2): Red (#F44336)

---

#### 6.2.7 Supervised Learning Component

**Module**: `src/supervised/supervised_learning.py`

**Purpose**: Supervised learning for pattern discrimination (NOT prediction).

**Key Functions**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `run_mdr_discrimination()` | MDR discrimination analysis | DataFrame, cols | Results dict |
| `run_supervised_pipeline()` | Main supervised pipeline | DataFrame, cols, target | Results dict |
| `prepare_data_for_classification()` | Split and preprocess | DataFrame, cols, target | Train/test splits |
| `run_all_models()` | Train and evaluate all models | X, y splits | Model results dict |
| `evaluate_model()` | Calculate metrics | Model, X_test, y_test | Metrics dict |
| `get_feature_importance()` | Extract importance scores | Model, feature names | Importance dict |
| `interpret_feature_importance()` | Biological interpretation | Importance dict | Interpretation dict |
| `save_model()` / `load_model()` | Model persistence | Model, path | - |

**Models Evaluated**:
- Random Forest (n_estimators=100)
- Support Vector Machine (kernel='rbf')
- k-Nearest Neighbors (n_neighbors=5)
- Logistic Regression (max_iter=1000)
- Decision Tree
- Naive Bayes (Gaussian)

---

#### 6.2.8 Regional Environmental Analysis Component

**Module**: `src/analysis/regional_environmental.py`

**Purpose**: Cluster distribution analysis and PCA.

**Key Functions**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `run_regional_environmental_analysis()` | Main analysis pipeline | DataFrame, cols, output_dir | Results dict |
| `analyze_cluster_distribution()` | Cross-tabulation analysis | DataFrame | Distribution dict |
| `cross_tabulate_clusters()` | Create cross-tabulation | DataFrame, cluster_col, group_col | Crosstab DataFrame |
| `perform_pca()` | Principal component analysis | DataFrame, cols | (X_pca, PCA, info) |
| `create_pca_plot()` | PCA scatter plot | X_pca, DataFrame, color_col | Figure |
| `create_pca_biplot()` | PCA biplot with loadings | X_pca, PCA, feature_names | Figure |

---

#### 6.2.9 Integration and Synthesis Component

**Module**: `src/analysis/integration_synthesis.py`

**Purpose**: Integrate results from all phases and synthesize findings.

**Key Functions**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `run_integration_synthesis()` | Main integration | DataFrame, cols, supervised_results | Results dict |
| `compare_clusters_with_supervised()` | Compare unsupervised vs supervised | DataFrame | Comparison dict |
| `identify_resistance_archetypes()` | Identify cluster archetypes | DataFrame, cols | Archetypes dict |
| `identify_species_environment_associations()` | Species-environment analysis | DataFrame | Associations dict |
| `identify_mdr_enriched_patterns()` | Find MDR-enriched groups | DataFrame, cols | MDR patterns dict |

---

#### 6.2.10 Dashboard Component

**Module**: `app/streamlit_app.py`

**Purpose**: Interactive web dashboard for data exploration and visualization.

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `main()` | Dashboard entry point and layout |
| `load_data()` | Load uploaded or default data |
| `get_antibiotic_cols()` | Identify antibiotic columns |
| `create_heatmap()` | Generate heatmap for display |
| `create_cluster_summary()` | Generate cluster statistics |
| `perform_pca_analysis()` | Run PCA for visualization |
| `create_pca_plot()` | Generate PCA scatter plot |

**Dashboard Pages**:
1. **Overview** - Dataset summary and preview
2. **Resistance Heatmap** - Visual resistance profiles
3. **Cluster Analysis** - Cluster statistics and distribution
4. **PCA Analysis** - Dimensionality reduction visualization
5. **Regional Distribution** - Geographic analysis
6. **Model Evaluation** - Supervised learning results
7. **Integration & Synthesis** - Combined analysis findings

---

## 7. Component Breakdown Table

This section provides a consolidated summary of all system components, their responsibilities, inputs, and outputs.

### 7.1 Layer-by-Layer Component Summary

#### Data Ingestion Layer

| Component | Responsibility | Input | Output |
|-----------|----------------|-------|--------|
| `data_ingestion.py` | Load, parse, and consolidate CSV files; extract metadata from filenames and isolate codes | Raw CSV files (*.csv) | `unified_raw_dataset.csv` |

#### Data Preprocessing & Validation Layer

| Component | Responsibility | Input | Output |
|-----------|----------------|-------|--------|
| `data_cleaning.py` | Standardize values, remove duplicates, filter by coverage thresholds, generate cleaning report | `unified_raw_dataset.csv` | `cleaned_dataset.csv`, `cleaning_report.txt` |
| `resistance_encoding.py` | Encode S/I/R to 0/1/2, generate resistance fingerprints | `cleaned_dataset.csv` | `encoded_dataset.csv` |
| `feature_engineering.py` | Compute MAR index, MDR status, resistant class counts, binary indicators | `encoded_dataset.csv` | `analysis_ready_dataset.csv`, feature matrix, metadata |

#### Analysis & Modeling Layer

| Component | Responsibility | Input | Output |
|-----------|----------------|-------|--------|
| `hierarchical_clustering.py` | Unsupervised structure identification using Ward's hierarchical clustering | Feature matrix | `clustered_dataset.csv`, linkage matrix, clustering info |
| `supervised_learning.py` | Pattern discrimination evaluation (species, MDR); feature importance analysis | Feature matrix, labels | Model files (*.joblib), evaluation metrics, importance scores |
| `regional_environmental.py` | PCA, cluster distribution analysis, chi-square tests | Clustered dataset | PCA results, distribution statistics, figures |
| `integration_synthesis.py` | Synthesize findings, identify archetypes, MDR-enriched patterns | All analysis results | Integration report, archetype definitions |

#### Visualization & Interaction Layer

| Component | Responsibility | Input | Output |
|-----------|----------------|-------|--------|
| `visualization.py` | Generate heatmaps, dendrograms, distribution plots | Processed data, linkage matrix | PNG figures in `figures/` directory |
| `streamlit_app.py` | Interactive dashboard for exploration and visualization | All processed datasets, models | Web interface (port 8501) |

#### Persistence / Artifact Storage Layer

| Component | Responsibility | Input | Output |
|-----------|----------------|-------|--------|
| File System (`data/processed/`) | Store processed datasets at each pipeline stage | CSV files | Versioned CSV files |
| Model Storage (`data/models/`) | Persist trained classifiers with preprocessing info | Trained models, scalers, encoders | `*.joblib` serialized files |
| Figure Storage (`data/processed/figures/`) | Store generated visualizations | Matplotlib figures | PNG image files |

#### Deployment Layer

| Component | Responsibility | Input | Output |
|-----------|----------------|-------|--------|
| `main.py` | Orchestrate full pipeline execution | Raw CSV directory | All processed outputs |
| Streamlit Server | Host interactive dashboard | Processed datasets | Web application |

### 7.2 Inter-Component Data Flow Matrix

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                      COMPONENT INTERACTION MATRIX                                  │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  FROM / TO          │ Ingestion │ Cleaning │ Encoding │ Features │ Clustering    │
│  ───────────────────┼───────────┼──────────┼──────────┼──────────┼──────────────│
│  Raw CSV Files      │     ●     │          │          │          │               │
│  Data Ingestion     │           │    ●     │          │          │               │
│  Data Cleaning      │           │          │    ●     │          │               │
│  Resistance Encoding│           │          │          │    ●     │               │
│  Feature Engineering│           │          │          │          │      ●        │
│                                                                                    │
│  FROM / TO          │ Supervised│ Regional │ Integration│ Visualize│ Dashboard   │
│  ───────────────────┼───────────┼──────────┼───────────┼──────────┼─────────────│
│  Feature Engineering│     ●     │          │           │          │      ●       │
│  Clustering         │           │    ●     │     ●     │    ●     │      ●       │
│  Supervised Learning│           │          │     ●     │          │      ●       │
│  Regional Analysis  │           │          │     ●     │    ●     │      ●       │
│  Integration        │           │          │           │          │      ●       │
│                                                                                    │
│  Legend: ● = Data flows from row component to column component                     │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Data Model and Schemas

### 8.1 Input Data Schema

**Raw CSV Structure**:

```
Row 3: CODE | ISOLATE ID | [metadata] | ... | SCORED RESISTANCE | NO. ANTIBIOTIC TESTED | MAR INDEX
Row 4: ESBL | AM | AMC | CPT | CN | ... (antibiotic names)
Row 5: MIC  | INT| INT | INT | INT| ... (value type indicators)
Row 6+: Data rows with isolate-level AST results
```

### 8.2 Processed Data Schemas

#### 8.2.1 unified_raw_dataset.csv

| Column | Type | Description |
|--------|------|-------------|
| CODE | string | Unique isolate identifier |
| ISOLATE_ID | string | Species name |
| REGION | string | Geographic region |
| SITE | string | Collection site |
| NATIONAL_SITE | string | Parsed national site code |
| LOCAL_SITE | string | Parsed local site |
| SAMPLE_SOURCE | string | Environmental source |
| REPLICATE | integer | Replicate number |
| COLONY | integer | Colony number |
| ESBL | string | ESBL status |
| SOURCE_FILE | string | Original filename |
| {AB} | string | S/I/R for each antibiotic |
| SCORED_RESISTANCE | float | Original resistance score |
| NUM_ANTIBIOTICS_TESTED | integer | Count of antibiotics tested |
| MAR_INDEX | float | Original MAR index |

#### 8.2.2 analysis_ready_dataset.csv (additional columns)

| Column | Type | Description |
|--------|------|-------------|
| {AB}_encoded | integer | Encoded resistance (0/1/2) |
| resistance_fingerprint | string | Concatenated fingerprint |
| MAR_INDEX_COMPUTED | float | Computed MAR index (0-1) |
| RESISTANCE_COUNT | integer | Count of resistant antibiotics |
| RESISTANT_CLASSES_COUNT | integer | Count of resistant classes |
| MDR_FLAG | boolean | Multi-drug resistant status |
| MDR_CATEGORY | string | "MDR" or "Non-MDR" |
| {AB}_RESISTANT | integer | Binary resistance indicator (0/1) |

#### 8.2.3 clustered_dataset.csv (additional columns)

| Column | Type | Description |
|--------|------|-------------|
| CLUSTER | integer | Assigned cluster label (1-n) |

### 8.3 Model Persistence Schema

**Model File Structure (.joblib)**:

```python
# Saved using joblib.dump()
# Components from sklearn.preprocessing
{
    'model': sklearn_estimator,         # Trained classifier (e.g., RandomForestClassifier)
    'scaler': StandardScaler_instance,  # Fitted sklearn.preprocessing.StandardScaler
    'label_encoder': LabelEncoder_instance  # Fitted sklearn.preprocessing.LabelEncoder
}
```

---

## 9. Module Interfaces

### 9.1 Preprocessing Pipeline Interface

```python
# Phase 2.1: Data Ingestion
def create_unified_dataset(input_dir: str, output_path: str = None) -> pd.DataFrame:
    """
    Load and consolidate all CSV files.
    
    Args:
        input_dir: Directory containing raw CSV files
        output_path: Optional path to save unified dataset
    
    Returns:
        DataFrame with unified raw data
    """

# Phase 2.2-2.3: Data Cleaning
def clean_dataset(
    df: pd.DataFrame,
    min_antibiotic_coverage: float = 50.0,
    max_isolate_missing: float = 50.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean and standardize the dataset.
    
    Args:
        df: Raw unified dataset
        min_antibiotic_coverage: Minimum % of isolates for antibiotic retention
        max_isolate_missing: Maximum % missing data allowed per isolate
    
    Returns:
        Tuple of (cleaned DataFrame, cleaning report dict)
    """

# Phase 2.4: Resistance Encoding
def create_encoded_dataset(
    df: pd.DataFrame,
    min_antibiotic_coverage: float = 50.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode resistance values numerically.
    
    Args:
        df: Cleaned dataset
        min_antibiotic_coverage: Coverage threshold
    
    Returns:
        Tuple of (encoded DataFrame, encoding info dict)
    """

# Phase 2.5: Feature Engineering
def prepare_analysis_ready_dataset(
    df: pd.DataFrame,
    antibiotic_cols: List[str] = None,
    output_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Add derived features to the dataset.
    
    Args:
        df: Encoded dataset
        antibiotic_cols: List of encoded antibiotic columns
        output_path: Optional save path
    
    Returns:
        Tuple of (full DataFrame, feature matrix, metadata, feature info dict)
    """
```

### 9.2 Analysis Pipeline Interface

```python
# Phase 3.1: Clustering
def run_clustering_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 5,
    linkage_method: str = 'ward',
    distance_metric: str = 'euclidean'
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Perform hierarchical clustering.
    
    Args:
        df: Analysis-ready dataset
        feature_cols: Encoded resistance columns
        n_clusters: Number of clusters to form
        linkage_method: Linkage algorithm ('ward', 'complete', 'average', 'single')
        distance_metric: Distance metric ('euclidean', 'manhattan')
    
    Returns:
        Tuple of (clustered DataFrame, linkage matrix, clustering info dict)
    """

# Phase 3.2: Visualization
def generate_all_visualizations(
    df: pd.DataFrame,
    feature_cols: List[str],
    linkage_matrix: np.ndarray,
    output_dir: str
) -> Dict[str, str]:
    """
    Generate all visualization outputs.
    
    Args:
        df: Clustered dataset
        feature_cols: Encoded resistance columns
        linkage_matrix: Clustering linkage matrix
        output_dir: Directory to save figures
    
    Returns:
        Dictionary mapping visualization names to file paths
    """

# Phase 4: Supervised Learning
def run_mdr_discrimination(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """
    Evaluate MDR pattern discrimination.
    
    Args:
        df: Analysis-ready dataset
        feature_cols: Encoded resistance columns
    
    Returns:
        Results dictionary with model metrics and feature importance
    """

# Phase 5: Regional Analysis
def run_regional_environmental_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str = None
) -> Dict:
    """
    Perform regional and environmental analysis.
    
    Args:
        df: Clustered dataset
        feature_cols: Encoded resistance columns
        output_dir: Optional output directory for figures
    
    Returns:
        Results dictionary with distributions and PCA results
    """

# Phase 6: Integration
def run_integration_synthesis(
    df: pd.DataFrame,
    feature_cols: List[str],
    supervised_results: Dict = None
) -> Dict:
    """
    Integrate and synthesize all analysis results.
    
    Args:
        df: Clustered dataset
        feature_cols: Encoded resistance columns
        supervised_results: Results from supervised learning (optional)
    
    Returns:
        Comprehensive integration results dictionary
    """
```

### 9.3 Pipeline Orchestration Interface

```python
# main.py
def run_full_pipeline(
    data_dir: str = None,
    output_dir: str = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute the complete AMR analysis pipeline.
    
    Args:
        data_dir: Directory containing raw CSV files
        output_dir: Directory for processed outputs
    
    Returns:
        Tuple of (final clustered DataFrame, integration results dict)
    """
```

---

## 10. Technology Stack

### 10.1 Technology Stack Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TECHNOLOGY STACK                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      PRESENTATION LAYER                              │   │
│  │  Streamlit >= 1.24.0                                                │   │
│  │  • Web framework for interactive dashboards                         │   │
│  │  • Reactive UI components                                           │   │
│  │  • Built-in caching and session state                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      VISUALIZATION LAYER                             │   │
│  │  Matplotlib >= 3.6.0                                                │   │
│  │  • Static plot generation                                           │   │
│  │  • Heatmaps, dendrograms, scatter plots                             │   │
│  │                                                                      │   │
│  │  Seaborn >= 0.12.0                                                  │   │
│  │  • Statistical visualization                                        │   │
│  │  • Enhanced plot aesthetics                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     MACHINE LEARNING LAYER                           │   │
│  │  scikit-learn >= 1.1.0                                              │   │
│  │  • Classification algorithms (RF, SVM, KNN, LR, DT, NB)             │   │
│  │  • Preprocessing (scaling, imputation)                              │   │
│  │  • Model evaluation metrics                                         │   │
│  │  • PCA and dimensionality reduction                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SCIENTIFIC COMPUTING LAYER                        │   │
│  │  SciPy >= 1.9.0                                                     │   │
│  │  • Hierarchical clustering (linkage, fcluster)                      │   │
│  │  • Statistical tests (chi-square)                                   │   │
│  │  • Distance calculations                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DATA LAYER                                    │   │
│  │  Pandas >= 1.5.0                                                    │   │
│  │  • DataFrame operations                                             │   │
│  │  • CSV I/O                                                          │   │
│  │  • Data manipulation                                                │   │
│  │                                                                      │   │
│  │  NumPy >= 1.23.0                                                    │   │
│  │  • Array operations                                                 │   │
│  │  • Numerical computing                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       UTILITY LAYER                                  │   │
│  │  joblib >= 1.2.0                                                    │   │
│  │  • Model serialization/persistence                                  │   │
│  │                                                                      │   │
│  │  Python >= 3.8                                                      │   │
│  │  • Runtime environment                                              │   │
│  │  • pathlib, re, os (standard library)                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Dependency Matrix

| Dependency | Version | Purpose | Components Using |
|------------|---------|---------|------------------|
| pandas | >=1.5.0 | Data manipulation | All modules |
| numpy | >=1.23.0 | Numerical operations | All modules |
| scipy | >=1.9.0 | Clustering, statistics | clustering, analysis |
| scikit-learn | >=1.1.0 | ML algorithms | supervised, preprocessing |
| matplotlib | >=3.6.0 | Visualization | visualization, analysis |
| seaborn | >=0.12.0 | Statistical plots | visualization |
| streamlit | >=1.24.0 | Dashboard | app |
| joblib | >=1.2.0 | Model persistence | supervised |

---

## 11. Deployment Architecture

### 11.1 Local Deployment Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LOCAL DEPLOYMENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Workstation                                                           │
│  ├── Python 3.8+ Environment                                                │
│  │   ├── Virtual Environment (recommended)                                  │
│  │   └── Installed Dependencies (requirements.txt)                          │
│  │                                                                          │
│  ├── Project Directory                                                      │
│  │   ├── amr-thesis-project/                                               │
│  │   │   ├── src/               # Source modules                           │
│  │   │   ├── app/               # Dashboard application                    │
│  │   │   ├── docs/              # Documentation                            │
│  │   │   ├── data/              # Data directory (created at runtime)      │
│  │   │   │   ├── processed/     # Processed datasets                       │
│  │   │   │   ├── models/        # Trained models                           │
│  │   │   │   └── figures/       # Visualizations                           │
│  │   │   ├── main.py            # Pipeline entry point                     │
│  │   │   └── requirements.txt   # Dependencies                             │
│  │   │                                                                      │
│  │   └── Input CSV Files (*.csv)                                           │
│  │                                                                          │
│  └── Runtime Processes                                                      │
│      ├── python main.py           # Batch pipeline                         │
│      └── streamlit run app/       # Dashboard server (port 8501)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Installation Procedure

```bash
# 1. Clone repository
git clone <repository-url>
cd amr-thesis-project

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import pandas; import sklearn; import streamlit; print('OK')"
```

### 11.3 Execution Modes

| Mode | Command | Purpose |
|------|---------|---------|
| **Full Pipeline** | `python main.py` | Run complete analysis |
| **Dashboard** | `streamlit run app/streamlit_app.py` | Interactive exploration |
| **Module Test** | `python -m src.preprocessing.data_ingestion` | Test individual module |

### 11.4 Directory Structure at Runtime

```
amr-thesis-project/
├── data/
│   ├── processed/
│   │   ├── unified_raw_dataset.csv
│   │   ├── cleaned_dataset.csv
│   │   ├── cleaning_report.txt
│   │   ├── encoded_dataset.csv
│   │   ├── analysis_ready_dataset.csv
│   │   ├── clustered_dataset.csv
│   │   └── figures/
│   │       ├── resistance_heatmap.png
│   │       ├── dendrogram.png
│   │       ├── clustered_heatmap.png
│   │       ├── cluster_profiles.png
│   │       ├── mdr_distribution.png
│   │       ├── mar_distribution.png
│   │       ├── pca_by_cluster.png
│   │       ├── pca_by_region.png
│   │       ├── pca_by_mdr.png
│   │       └── pca_biplot.png
│   └── models/
│       ├── mdr_classifier.joblib
│       └── species_classifier.joblib
├── src/
├── app/
├── docs/
└── *.csv (input files)
```

### 11.5 Model Handling (Training vs Inference)

This system treats all analytical models as **read-only artifacts** at deployment. The architecture clearly separates training-time and inference-time operations:

| Phase | Mode | Operations | Output |
|-------|------|------------|--------|
| **Training** | Batch Pipeline (`main.py`) | Data preprocessing, model fitting, cross-validation | `*.joblib` model files |
| **Inference** | Dashboard (`streamlit_app.py`) | Load pre-trained models, apply to loaded data | Visualizations, metrics display |

**Training Pipeline (Offline):**
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Model Serialization
                                                        │
                                                        ▼
                                              data/models/*.joblib
```

**Dashboard Inference (Online):**
```
                              data/models/*.joblib
                                      │
                                      ▼
Processed Data → Load Models → Apply Transforms → Display Results
```

**User Interaction Boundaries:**
- Users interact **only** through the Streamlit dashboard or CLI
- No model retraining occurs during dashboard interaction
- Dashboard displays pre-computed results and static visualizations
- Users can upload new data for visualization but models remain fixed

---

## 12. Non-Functional Considerations

This section details how the architecture addresses non-functional requirements essential for a thesis-level academic project.

### 12.1 Modularity and Maintainability

| Aspect | Implementation | Benefit |
|--------|----------------|---------|
| **Separation of Concerns** | Distinct modules for preprocessing, clustering, supervised learning, analysis, and visualization | Changes in one area don't affect others |
| **Single Responsibility** | Each module has one primary purpose | Easier to understand, test, and maintain |
| **Standard Interfaces** | Well-defined function signatures with type hints | Clear contracts between components |
| **Configuration Isolation** | Parameters defined at module level; easily adjustable | Flexibility without code changes |

**Module Independence Diagram:**
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODULE DEPENDENCY GRAPH                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐           │
│  │   data_         │────▶│   data_         │────▶│   resistance_   │           │
│  │   ingestion     │     │   cleaning      │     │   encoding      │           │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘           │
│                                                         │                       │
│                                                         ▼                       │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐           │
│  │   visualization │◀────│   hierarchical_ │◀────│   feature_      │           │
│  │                 │     │   clustering    │     │   engineering   │           │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘           │
│                                                         │                       │
│                                                         ▼                       │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐           │
│  │   streamlit_    │◀────│   integration_  │◀────│   supervised_   │           │
│  │   app           │     │   synthesis     │     │   learning      │           │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘           │
│                                                                                 │
│  Note: Arrows indicate data flow direction, not import dependencies             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Reproducibility

| Requirement | Implementation | Documentation |
|-------------|----------------|---------------|
| **Deterministic Results** | Fixed `random_state=42` across all stochastic operations | Reported in METHODOLOGY.md |
| **Versioned Outputs** | Timestamped output directories; CSV snapshots at each phase | Output file naming convention |
| **Documented Parameters** | All algorithm parameters logged in cleaning report and metadata | `cleaning_report.txt`, clustering info dict |
| **Environment Specification** | `requirements.txt` with pinned minimum versions | Reproducible environment setup |
| **Audit Trail** | Cleaning report documents all transformation decisions | `cleaning_report.txt` |

**Reproducibility Checklist:**
- [x] Random states fixed and reported
- [x] Data transformations documented
- [x] Coverage thresholds explicitly stated
- [x] Model hyperparameters logged
- [x] Output files versioned

### 12.3 Scalability (Conceptual)

While the system is designed for workstation-level deployment, the architecture supports conceptual scalability:

| Dimension | Current Implementation | Scalability Path |
|-----------|------------------------|------------------|
| **Data Volume** | DataFrame-based; suitable for thousands of isolates | Chunked processing or Dask for larger datasets |
| **Processing Parallelism** | `n_jobs=-1` for Random Forest utilizes all CPU cores | Can extend to other sklearn parallelizable operations |
| **Storage** | File-based CSV/joblib; efficient for thesis scale | Database integration possible (SQLite, PostgreSQL) |
| **User Concurrency** | Single-user Streamlit dashboard | Multi-user deployment via Streamlit Cloud or containerization |

> **Note:** Enterprise-scale scalability is out of scope for this thesis-level project.

### 12.4 Ethical and Security Considerations

#### Data Privacy
| Concern | Mitigation |
|---------|------------|
| **No Patient Identifiers** | System processes environmental/aquatic samples only |
| **Anonymized Isolate Codes** | Codes contain location/sample info, no personal data |
| **Local Processing** | No external data transmission; all processing on-premises |

#### Data Integrity
| Concern | Mitigation |
|---------|------------|
| **Input Validation** | Data cleaning validates all resistance values (S/I/R only) |
| **Transformation Audit** | Cleaning report documents all data modifications |
| **Reproducible Outputs** | Fixed random states ensure consistent results |

#### Ethical Use Boundaries
> ⚠️ **Critical Disclaimer**: This system is intended for **exploratory pattern recognition and surveillance analysis only**. It should **NOT** be used for:
> - Clinical decision support
> - Treatment recommendations
> - Diagnostic purposes
> - Patient-level risk assessment

---

## 13. Design Rationale & Justifications

This section consolidates the key architectural decisions and their justifications.

### 13.1 Architectural Style Selection

**Decision**: Layered Architecture combined with Pipeline Pattern

| Alternative Considered | Reason for Rejection | Selected Approach |
|------------------------|---------------------|-------------------|
| Microservices | Overkill for thesis-level scope; adds deployment complexity | Modular monolith with clear layer boundaries |
| Event-Driven | Unnecessary for batch processing; no real-time requirements | Sequential pipeline with defined data transformations |
| Model-View-Controller | Dashboard-only pattern; doesn't capture full pipeline | Layered architecture spanning all phases |

**Justification**: The combination of layered architecture and pipeline pattern provides:
1. Clear separation between data, processing, and presentation concerns
2. Sequential processing stages that align with the research methodology
3. Academic appropriateness with realistic implementation scope
4. Ease of debugging and validation at each stage

### 13.2 Technology Stack Decisions

| Decision | Justification | Alternatives Rejected |
|----------|---------------|----------------------|
| **Python 3.8+** | Industry standard for data science; extensive ecosystem | R (less general-purpose), Julia (smaller ecosystem) |
| **pandas** | De facto standard for tabular data; excellent documentation | Polars (newer, less mature), raw NumPy (less convenient) |
| **scikit-learn** | Comprehensive, well-documented ML library; academic credibility | TensorFlow/PyTorch (overkill for classical ML), statsmodels (less comprehensive) |
| **Streamlit** | Rapid dashboard development; Python-native; minimal frontend expertise required | Dash (more complex), Flask+templates (more manual work) |
| **Ward's Linkage** | Produces compact, balanced clusters; well-suited for resistance patterns | Complete linkage (chaining tendency), K-means (requires k specification, different assumptions) |

### 13.3 Data Processing Decisions

| Decision | Rationale |
|----------|-----------|
| **Ordinal Encoding (S=0, I=1, R=2)** | Preserves biological ordering of resistance levels; enables meaningful distance calculations |
| **Median Imputation** | Robust to outliers in resistance data; preserves central tendency |
| **70% Coverage Threshold** | Balances data quality with retention; ensures robust pattern discrimination |
| **30% Missing Data Threshold** | Excludes isolates with excessive gaps; maintains analysis integrity |

### 13.4 Model Handling Philosophy

**Decision**: Models as read-only artifacts at deployment

**Justification**:
1. **Reproducibility**: Fixed models ensure consistent results across dashboard sessions
2. **Auditability**: Training decisions are documented and versioned
3. **Scope Clarity**: System is for exploration, not real-time prediction
4. **Simplicity**: Avoids complex model versioning and update mechanisms

### 13.5 Separation of Concerns

| Layer | Responsibility | Isolation Mechanism |
|-------|----------------|---------------------|
| **Data Ingestion** | Load and consolidate raw data | Separate module; outputs CSV checkpoint |
| **Data Cleaning** | Standardize and validate | Separate module; generates cleaning report |
| **Feature Engineering** | Compute derived features | Separate module; outputs feature matrix and metadata |
| **Analysis** | Clustering, supervised learning, PCA | Separate modules per analysis type |
| **Visualization** | Generate static plots | Standalone module; reads processed data |
| **Dashboard** | Interactive exploration | Loads all outputs; no processing logic |

---

## 14. Quality Attributes

### 14.1 Reliability

| Attribute | Implementation |
|-----------|----------------|
| **Error Handling** | Try-catch blocks with informative error messages |
| **Missing Data** | Graceful handling via imputation (median strategy) |
| **Edge Cases** | Filters for insufficient samples (e.g., <2 for stratified split) |

### 14.2 Maintainability

| Attribute | Implementation |
|-----------|----------------|
| **Modularity** | Separate modules for each processing phase |
| **Documentation** | Comprehensive docstrings and markdown documentation |
| **Code Style** | Consistent naming conventions and structure |

### 14.3 Usability

| Attribute | Implementation |
|-----------|----------------|
| **CLI Interface** | Simple `python main.py` execution |
| **GUI Interface** | Interactive Streamlit dashboard |
| **Progress Feedback** | Console output for each pipeline phase |
| **Output Files** | Well-organized directory structure |

### 14.4 Performance

| Attribute | Implementation |
|-----------|----------------|
| **Memory Efficiency** | DataFrame operations (vectorized) |
| **Computation** | scikit-learn optimized algorithms |
| **Parallelism** | n_jobs=-1 for Random Forest (all cores) |

---

## Appendices

### A. File Inventory

| File Path | Purpose | Lines of Code (approx) |
|-----------|---------|------------------------|
| `main.py` | Pipeline orchestration | 220 |
| `src/preprocessing/data_ingestion.py` | CSV loading and parsing | 365 |
| `src/preprocessing/data_cleaning.py` | Data standardization | 530 |
| `src/preprocessing/resistance_encoding.py` | Value encoding | 310 |
| `src/preprocessing/feature_engineering.py` | Feature computation | 395 |
| `src/clustering/hierarchical_clustering.py` | Clustering algorithms | 375 |
| `src/visualization/visualization.py` | Plot generation | 495 |
| `src/supervised/supervised_learning.py` | ML model training | 715 |
| `src/analysis/regional_environmental.py` | PCA and distributions | 425 |
| `src/analysis/integration_synthesis.py` | Result synthesis | 740 |
| `app/streamlit_app.py` | Dashboard application | 715 |

### B. Glossary

| Term | Definition |
|------|------------|
| **AST** | Antimicrobial Susceptibility Testing |
| **MAR Index** | Multiple Antibiotic Resistance Index (resistant/tested) |
| **MDR** | Multi-Drug Resistant (resistant to ≥3 antibiotic classes) |
| **Pattern Discrimination** | Evaluating how well resistance patterns distinguish categories |
| **Structure Identification** | Discovering natural groupings via unsupervised learning |
| **Resistance Fingerprint** | Encoded pattern of susceptibility results for an isolate |
| **Cluster Archetype** | Characteristic resistance profile defining each cluster |
| **Ward's Method** | Hierarchical clustering linkage that minimizes within-cluster variance |

### C. Related Documents

| Document | Description |
|----------|-------------|
| [DOCUMENTATION.md](DOCUMENTATION.md) | Technical documentation with preprocessing decisions |
| [METHODOLOGY.md](METHODOLOGY.md) | Research methodology and analytical framework |
| [RUNNING_THE_SYSTEM.md](RUNNING_THE_SYSTEM.md) | Step-by-step execution guide |
| [README.md](../README.md) | Project overview and quick start |

### D. Antibiotic Classes Reference

| Class | Antibiotics |
|-------|-------------|
| Penicillins | AM, AMP |
| β-lactam/β-lactamase inhibitor | AMC, PRA |
| Cephalosporins (1st gen) | CN, CF |
| Cephalosporins (3rd/4th gen) | CPD, CTX, CFT, CPT |
| Cephamycins | CFO |
| Cephalosporin/BLI combinations | CZA |
| Carbapenems | IPM, MRB |
| Aminoglycosides | AN, GM, N |
| Quinolones/Fluoroquinolones | NAL, ENR |
| Tetracyclines | DO, TE |
| Nitrofurans | FT |
| Phenicols | C |
| Folate pathway inhibitors | SXT |

---

## Document Information

| Attribute | Value |
|-----------|-------|
| **Version** | 1.0 |
| **Created** | 2024 |
| **Authors** | AMR Research Team |
| **Status** | Final |

---

*This architectural design document is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*
