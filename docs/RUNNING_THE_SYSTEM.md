# Step-by-Step Procedure for Running the AMR Thesis Project

This guide provides detailed instructions for setting up and running the AMR Pattern Recognition system.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Running the Full Pipeline](#3-running-the-full-pipeline)
4. [Running Individual Modules](#4-running-individual-modules)
5. [Launching the Interactive Dashboard](#5-launching-the-interactive-dashboard)
6. [Output Files](#6-output-files)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

Before running the system, ensure you have the following installed:

### Required Software

| Software | Version | Description |
|----------|---------|-------------|
| Python | 3.8 or higher | Programming language runtime |
| pip | Latest | Python package manager |
| Git | Latest | Version control (optional, for cloning) |

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Disk Space**: At least 500 MB for dependencies and output files

### Verify Python Installation

Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:

```bash
python --version
```

or

```bash
python3 --version
```

You should see output like `Python 3.8.x` or higher.

---

## 2. Installation

### Step 2.1: Clone or Download the Repository

**Option A: Clone with Git**

```bash
git clone https://github.com/your-repo/amr-thesis-project.git
cd amr-thesis-project
```

**Option B: Download ZIP**

1. Download the repository as a ZIP file
2. Extract to your desired location
3. Open terminal and navigate to the extracted folder:
   ```bash
   cd path/to/amr-thesis-project
   ```

### Step 2.2: Create a Virtual Environment (Recommended)

Creating a virtual environment keeps dependencies isolated:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt.

### Step 2.3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:
- pandas, numpy (data manipulation)
- scipy (scientific computing)
- scikit-learn (machine learning)
- matplotlib, seaborn (visualization)
- streamlit (dashboard)
- joblib (model persistence)

### Step 2.4: Verify Installation

```bash
python -c "import pandas; import sklearn; import streamlit; print('All dependencies installed successfully!')"
```

---

## 3. Running the Full Pipeline

The full pipeline processes all phases of the AMR analysis automatically.

### Step 3.1: Prepare Your Data

Ensure your CSV data files are in the project root directory or a `data/raw/` folder. The expected format:
- CSV files with resistance data
- Column names following the naming convention (see [README.md](../README.md))

### Step 3.2: Execute the Pipeline

From the project root directory, run:

```bash
python main.py
```

### Step 3.3: Monitor Progress

The pipeline will display progress through each phase:

1. **Phase 2.1**: Data Ingestion and Consolidation
2. **Phase 2.2-2.3**: Data Cleaning
3. **Phase 2.4**: Resistance Encoding
4. **Phase 2.5**: Feature Engineering
5. **Phase 3.1**: Hierarchical Clustering
6. **Phase 3.2**: Visualization Generation
7. **Phase 3.3**: Cluster Interpretation
8. **Phase 4**: Supervised Learning (MDR Discrimination)
9. **Phase 5**: Regional & Environmental Analysis
10. **Phase 6**: Integration & Synthesis

### Step 3.4: Review Output

After completion, you will see:
- Summary statistics printed to the console
- Output files saved to `data/processed/`
- Visualizations saved to `data/processed/figures/`

---

## 4. Running Individual Modules

You can run specific pipeline phases independently for targeted analysis.

### Step 4.1: Data Ingestion Only

```python
# Create a Python script or use Python interactive mode
python
```

```python
from src.preprocessing.data_ingestion import create_unified_dataset

# Specify input and output paths
df = create_unified_dataset('data/raw/', 'data/processed/unified.csv')
print(f"Loaded {len(df)} isolates")
```

### Step 4.2: Data Cleaning Only

```python
from src.preprocessing.data_cleaning import clean_dataset, generate_cleaning_report
import pandas as pd

# Load raw data
df_raw = pd.read_csv('data/processed/unified_raw_dataset.csv')

# Clean the data
df_clean, cleaning_report = clean_dataset(df_raw)
df_clean.to_csv('data/processed/cleaned_dataset.csv', index=False)

# Generate report
generate_cleaning_report(cleaning_report, 'data/processed/cleaning_report.txt')
```

### Step 4.3: Clustering Only

```python
from src.clustering.hierarchical_clustering import run_clustering_pipeline
import pandas as pd

# Load analysis-ready data
df = pd.read_csv('data/processed/analysis_ready_dataset.csv')
feature_cols = [c for c in df.columns if c.endswith('_encoded')]

# Run clustering
df_clustered, linkage_matrix, info = run_clustering_pipeline(df, feature_cols, n_clusters=5)
df_clustered.to_csv('data/processed/clustered_dataset.csv', index=False)
```

### Step 4.4: Supervised Learning Only

```python
from src.supervised.supervised_learning import run_mdr_discrimination, save_model
import pandas as pd

# Load clustered data
df = pd.read_csv('data/processed/clustered_dataset.csv')
feature_cols = [c for c in df.columns if c.endswith('_encoded')]

# Run MDR discrimination
results = run_mdr_discrimination(df, feature_cols)

# Save the best model
save_model(
    results['best_model']['model_object'],
    results['scaler'],
    results['label_encoder'],
    'data/models/mdr_classifier.joblib'
)
```

### Step 4.5: Visualization Only

```python
from src.visualization.visualization import generate_all_visualizations
import pandas as pd

# Load clustered data
df = pd.read_csv('data/processed/clustered_dataset.csv')
feature_cols = [c for c in df.columns if c.endswith('_encoded')]

# Generate visualizations
generate_all_visualizations(df, feature_cols, None, 'data/processed/figures')
```

---

## 5. Launching the Interactive Dashboard

The Streamlit dashboard provides an interactive interface for exploring the analysis results.

### Step 5.1: Ensure Data is Available

The dashboard works best with processed data. Either:
- Run the full pipeline first (`python main.py`), OR
- Have your own analysis-ready CSV file to upload

### Step 5.2: Start the Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Step 5.3: Access the Dashboard

After running the command, you will see output like:

```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Open your web browser and navigate to `http://localhost:8501`.

### Step 5.4: Using the Dashboard

1. **Upload Data** (if needed): Use the sidebar file uploader to load your CSV dataset
2. **Select Analysis Type**: Choose from:
   - Overview: Dataset summary and statistics
   - Resistance Heatmap: Visual representation of resistance patterns
   - Cluster Analysis: Cluster distribution and characteristics
   - PCA Analysis: Principal component analysis visualization
   - Regional Distribution: Geographic analysis
   - Model Evaluation: Supervised learning results
   - Integration & Synthesis: Combined analysis findings

### Step 5.5: Stop the Dashboard

Press `Ctrl+C` in the terminal to stop the Streamlit server.

---

## 6. Output Files

After running the pipeline, the following files are generated:

### Data Files (in `data/processed/`)

| File | Description |
|------|-------------|
| `unified_raw_dataset.csv` | Consolidated raw data from all sources |
| `cleaned_dataset.csv` | Cleaned and standardized data |
| `cleaning_report.txt` | Documentation of cleaning decisions |
| `encoded_dataset.csv` | Numerically encoded resistance data |
| `analysis_ready_dataset.csv` | Final dataset with computed features |
| `clustered_dataset.csv` | Dataset with cluster assignments |

### Visualizations (in `data/processed/figures/`)

| File | Description |
|------|-------------|
| `resistance_heatmap.png` | Heatmap of resistance profiles |
| `dendrogram.png` | Hierarchical clustering dendrogram |
| `cluster_distribution.png` | Bar chart of cluster sizes |
| `pca_plot.png` | PCA visualization |
| `regional_analysis.png` | Regional distribution charts |

### Models (in `data/models/`)

| File | Description |
|------|-------------|
| `mdr_classifier.joblib` | Trained MDR classification model |

---

## 7. Troubleshooting

### Common Issues and Solutions

#### Issue: `ModuleNotFoundError: No module named 'xxx'`

**Solution**: Install missing dependencies:
```bash
pip install -r requirements.txt
```

#### Issue: `FileNotFoundError: No CSV files found`

**Solution**: Ensure your data files are in the correct location:
- Place CSV files in the project root directory, OR
- Create a `data/raw/` folder and place files there

#### Issue: `PermissionError` when saving files

**Solution**: 
- Ensure the output directories exist and are writable
- On Windows, run the terminal as Administrator if needed
- Check that no other program has the files open

#### Issue: Streamlit dashboard not loading

**Solution**:
1. Ensure Streamlit is installed: `pip install streamlit`
2. Check if another process is using port 8501:
   ```bash
   streamlit run app/streamlit_app.py --server.port 8502
   ```
3. Clear browser cache or try a different browser

#### Issue: `MemoryError` during processing

**Solution**:
- Close other applications to free RAM
- Process data in smaller batches
- Increase system swap/virtual memory

#### Issue: Virtual environment not activating

**Windows Solution**:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```

**macOS/Linux Solution**:
```bash
source venv/bin/activate
```

### Getting Help

If you encounter issues not covered here:

1. Check the [technical documentation](DOCUMENTATION.md) for detailed explanations
2. Review the [README.md](../README.md) for project overview
3. Contact the research team for support

---

## Quick Reference

### Essential Commands

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Run full pipeline | `python main.py` |
| Start dashboard | `streamlit run app/streamlit_app.py` |
| Activate virtual environment (Windows) | `venv\Scripts\activate` |
| Activate virtual environment (macOS/Linux) | `source venv/bin/activate` |
| Deactivate virtual environment | `deactivate` |

---

*This guide is part of Phase 8 — Documentation & Reporting for the AMR Thesis Project.*
