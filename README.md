# AMR Thesis Project: Antimicrobial Resistance Pattern Recognition

An analytical pipeline for antimicrobial resistance (AMR) pattern recognition and surveillance, designed for the analysis of bacterial isolates from environmental and clinical samples.

## Project Overview

This project implements a comprehensive data science pipeline for AMR surveillance, including:

- **Phase 2**: Data preprocessing (ingestion, cleaning, encoding, feature engineering)
- **Phase 3**: Unsupervised structure identification (hierarchical clustering, visualization)
- **Phase 4**: Supervised learning for pattern discrimination
- **Phase 5**: Regional and environmental analysis (PCA, distribution analysis)
- **Phase 6**: Integration and synthesis
- **Phase 7**: Interactive Streamlit dashboard
- **Phase 8**: Documentation (see [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md))

## Isolate Code Convention

The isolate codes follow this naming convention:

- **National Site**: O (Ormoc), P (Pampanga), M (Marawi/BARMM)
- **Local Site**: A (Alegria), L (Larrazabal), G (Gabriel), R (Roque), D (Dayawan), T (Tuca Kialdan)
- **Sample Source**: DW (Drinking Water), LW (Lake Water), FB (Fish Banak), FG (Fish Gusaw), RW (River Water), FT (Fish Tilapia), EWU (Effluent Water Untreated), EWT (Effluent Water Treated), FK (Fish Kaolang)
- **Replicate**: 1, 2, 3
- **Colony**: 1, 2, 3, etc.

Example: `EC_OADWR1C3` = Escherichia coli from Ormoc, Alegria site, Drinking Water, Replicate 1, Colony 3

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/amr-thesis-project.git
cd amr-thesis-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Full Pipeline

Execute the main pipeline to process all data:

```bash
python main.py
```

This will:
1. Load and consolidate all CSV files
2. Clean and preprocess the data
3. Encode resistance values (S=0, I=1, R=2)
4. Compute MAR index and MDR status
5. Perform hierarchical clustering
6. Run supervised learning models
7. Generate visualizations

### Running Individual Modules

Each phase can be run independently:

```python
# Data ingestion
from src.preprocessing.data_ingestion import create_unified_dataset
df = create_unified_dataset('data/', 'data/processed/unified.csv')

# Clustering
from src.clustering.hierarchical_clustering import run_clustering_pipeline
df_clustered, linkage_matrix, info = run_clustering_pipeline(df, feature_cols)

# Supervised learning
from src.supervised.supervised_learning import run_mdr_discrimination
results = run_mdr_discrimination(df, feature_cols)
```

### Interactive Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:
- Data overview and exploration
- Resistance heatmaps
- Cluster analysis
- PCA visualization
- Regional distribution analysis
- Model evaluation summaries

## Project Structure

```
amr-thesis-project/
├── app/
│   └── streamlit_app.py          # Interactive dashboard
├── data/
│   ├── raw/                      # Raw CSV files
│   ├── processed/                # Processed datasets
│   └── models/                   # Saved ML models
├── src/
│   ├── preprocessing/
│   │   ├── data_ingestion.py     # Phase 2.1
│   │   ├── data_cleaning.py      # Phase 2.2-2.3
│   │   ├── resistance_encoding.py # Phase 2.4
│   │   └── feature_engineering.py # Phase 2.5
│   ├── clustering/
│   │   └── hierarchical_clustering.py # Phase 3.1
│   ├── visualization/
│   │   └── visualization.py      # Phase 3.2
│   ├── supervised/
│   │   └── supervised_learning.py # Phase 4
│   └── analysis/
│       └── regional_environmental.py # Phase 5
├── tests/                        # Test files
├── main.py                       # Main pipeline script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Data Encoding

### Resistance Values
- **S (Susceptible)** → 0
- **I (Intermediate)** → 1
- **R (Resistant)** → 2

### MDR Definition
Multi-Drug Resistant (MDR) isolates are defined as those resistant to ≥3 antibiotic classes.

### MAR Index
Multiple Antibiotic Resistance (MAR) Index = Number of antibiotics resistant / Total antibiotics tested

## Output Files

After running the pipeline, the following files are generated:

- `unified_raw_dataset.csv`: Consolidated raw data from all sources
- `cleaned_dataset.csv`: Cleaned and standardized data
- `cleaning_report.txt`: Documentation of cleaning decisions
- `encoded_dataset.csv`: Numerically encoded resistance data
- `analysis_ready_dataset.csv`: Final dataset with all computed features
- `clustered_dataset.csv`: Dataset with cluster assignments
- `figures/`: Directory containing all visualizations

## Key Terminology

This project uses the following standardized terminology:

- **Pattern Discrimination**: Supervised learning to evaluate how resistance patterns distinguish known categories
- **Model Evaluation**: Quantifying model performance to assess pattern consistency
- **Structure Identification**: Unsupervised discovery of natural groupings in resistance data

For detailed terminology definitions, see [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

## Disclaimer

⚠️ **This tool is intended for exploratory pattern recognition and surveillance analysis only.** It should not be used for clinical decision support. No patient-level identifiers are processed.

## Antibiotic Classes

The pipeline recognizes the following antibiotic classes for MDR calculation:

- Penicillins (AM, AMP)
- β-lactam/β-lactamase inhibitor combinations (AMC, PRA)
- Cephalosporins - 1st generation (CN, CF)
- Cephalosporins - 3rd/4th generation (CPD, CTX, CFT, CPT)
- Cephamycins (CFO)
- Carbapenems (IPM, MRB)
- Aminoglycosides (AN, GM, N)
- Quinolones/Fluoroquinolones (NAL, ENR)
- Tetracyclines (DO, TE)
- Nitrofurans (FT)
- Phenicols (C)
- Folate pathway inhibitors (SXT)

## License

This project is developed for academic research purposes.

## Contact

For questions about this project, please contact the research team.
