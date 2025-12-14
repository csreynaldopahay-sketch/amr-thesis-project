"""
Supervised Learning Module for AMR Thesis Project
Phase 4 - Supervised Learning for Pattern Discrimination

OBJECTIVE (4.1):
    Evaluate how well resistance fingerprints discriminate known categories:
    - Bacterial species
    - MDR vs non-MDR groups
    Note: This is pattern discrimination, NOT forecasting/prediction of future outcomes.

DATA SPLITTING (4.2):
    - 80%-20% train-test split (default)
    - Purpose: Assess model generalization, avoid overfitting, support robustness
    - Framed as model validation, not prediction

MODEL SELECTION (4.3):
    - Random Forest
    - Support Vector Machine
    - k-Nearest Neighbors
    - Logistic Regression
    - Decision Tree
    - Naive Bayes

MODEL TRAINING (4.4):
    - Inputs: Resistance fingerprints (encoded antibiotic susceptibility)
    - Targets: Known labels (species or MDR category)

MODEL EVALUATION (4.5):
    - Accuracy, Precision, Recall, F1-score, Confusion matrix
    - Interpretation: Metrics quantify how consistently resistance patterns
      align with known categories, NOT predictive performance for future samples.

MODEL INTERPRETATION (4.6):
    - Feature importance analysis identifies antibiotics contributing most
      to group separation
    - Findings should be related to AST results, MDR trends, biological plausibility
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings('ignore')


# Model configurations
MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True),
    'k-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

# Antibiotic class mapping for biological plausibility interpretation (Phase 4.6)
ANTIBIOTIC_CLASSES = {
    'AM': 'Penicillins', 'AMP': 'Penicillins',
    'AMC': 'β-lactam/β-lactamase inhibitor', 'PRA': 'β-lactam/β-lactamase inhibitor',
    'CN': 'Cephalosporins-1st', 'CF': 'Cephalosporins-1st',
    'CPD': 'Cephalosporins-3rd/4th', 'CTX': 'Cephalosporins-3rd/4th',
    'CFT': 'Cephalosporins-3rd/4th', 'CPT': 'Cephalosporins-3rd/4th',
    'CFO': 'Cephamycins',
    'IPM': 'Carbapenems', 'MRB': 'Carbapenems',
    'AN': 'Aminoglycosides', 'GM': 'Aminoglycosides', 'N': 'Aminoglycosides',
    'NAL': 'Quinolones/Fluoroquinolones', 'ENR': 'Quinolones/Fluoroquinolones',
    'DO': 'Tetracyclines', 'TE': 'Tetracyclines',
    'FT': 'Nitrofurans',
    'C': 'Phenicols',
    'SXT': 'Folate pathway inhibitors'
}


def prepare_data_for_classification(df: pd.DataFrame,
                                    feature_cols: List[str],
                                    target_col: str,
                                    test_size: float = 0.2,
                                    random_state: int = 42) -> Tuple:
    """
    Prepare data for supervised learning (Phase 4.2 & 4.4).
    
    Applies 80%-20% train-test split (default) for model validation.
    Purpose: Assess model generalization, avoid overfitting, support robustness
    of pattern discrimination. This is framed as model validation, NOT prediction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features and target
    feature_cols : list
        List of feature column names (resistance fingerprints only)
    target_col : str
        Name of target column (known labels: species or MDR category)
    test_size : float
        Proportion of data for testing (default 0.2 = 20%)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, label_encoder, scaler)
    """
    # Remove rows with missing target
    df_valid = df[df[target_col].notna()].copy()
    
    # Filter out classes with fewer than 2 samples (required for stratified split)
    class_counts = df_valid[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    
    if (class_counts < 2).any():
        removed_classes = class_counts[class_counts < 2].index.tolist()
        warnings.warn(
            f"Removed {len(removed_classes)} class(es) with fewer than 2 samples "
            f"(required for stratified split): {removed_classes}",
            category=UserWarning
        )
    
    df_valid = df_valid[df_valid[target_col].isin(valid_classes)].copy()
    
    if len(df_valid) == 0:
        raise ValueError("No samples remaining after filtering classes with fewer than 2 samples.")
    
    # Prepare features
    existing_cols = [c for c in feature_cols if c in df_valid.columns]
    X = df_valid[existing_cols].copy()
    
    # Handle missing values in features
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Prepare target
    y = df_valid[target_col].copy()
    
    # Encode target if categorical
    label_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler


def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """
    Train a single model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to train
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    
    Returns:
    --------
    sklearn estimator
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   label_encoder=None) -> Dict:
    """
    Evaluate model performance (Phase 4.5).
    
    Computes: Accuracy, Precision, Recall, F1-score, Confusion matrix.
    
    Interpretation framing: These metrics quantify how consistently 
    resistance patterns align with known categories, NOT predictive 
    performance for future samples.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    label_encoder : LabelEncoder, optional
        Label encoder for decoding classes
    
    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Classification report
    if label_encoder is not None:
        target_names = label_encoder.classes_.tolist()
    else:
        target_names = [str(c) for c in np.unique(y_test)]
    
    results['classification_report'] = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    
    return results


def get_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """
    Extract feature importance from model (Phase 4.6).
    
    Identifies antibiotics contributing most to group separation.
    These findings should be related to:
    - AST (Antimicrobial Susceptibility Testing) results
    - MDR trends
    - Biological plausibility
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    feature_names : list
        List of feature names (antibiotic names)
    
    Returns:
    --------
    dict
        Feature importance scores (higher = more contribution to discrimination)
    """
    importance = {}
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        for name, imp in zip(feature_names, model.feature_importances_):
            importance[name] = float(imp)
    elif hasattr(model, 'coef_'):
        # Linear models
        coef = np.abs(model.coef_)
        if len(coef.shape) > 1:
            coef = coef.mean(axis=0)
        for name, imp in zip(feature_names, coef):
            importance[name] = float(imp)
    
    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    return importance


def _get_resistance_interpretation(resistance_rate: float) -> str:
    """
    Get human-readable interpretation of resistance rate.
    
    Parameters:
    -----------
    resistance_rate : float
        Proportion of isolates showing resistance (0.0-1.0)
    
    Returns:
    --------
    str
        Interpretation category (High/Moderate/Low resistance)
    """
    if resistance_rate > 0.5:
        return 'High resistance'
    elif resistance_rate > 0.2:
        return 'Moderate resistance'
    else:
        return 'Low resistance'


def interpret_feature_importance(feature_importance: Dict[str, float],
                                  df: pd.DataFrame = None,
                                  target_col: str = None) -> Dict:
    """
    Interpret feature importance findings (Phase 4.6).
    
    Relates feature importance to:
    - AST results: Which antibiotics are most discriminative
    - MDR trends: Connection to multi-drug resistance patterns
    - Biological plausibility: Interpretation context
    
    Parameters:
    -----------
    feature_importance : dict
        Feature importance scores from get_feature_importance()
    df : pd.DataFrame, optional
        Original dataframe for computing additional statistics
    target_col : str, optional
        Target column name for context
    
    Returns:
    --------
    dict
        Interpretation results including top discriminators and context
    """
    interpretation = {
        'top_discriminators': [],
        'antibiotic_classes_involved': set(),
        'interpretation_notes': []
    }
    
    # Get top 5 discriminators
    top_features = list(feature_importance.items())[:5]
    
    for antibiotic, score in top_features:
        ab_class = ANTIBIOTIC_CLASSES.get(antibiotic, 'Unknown class')
        interpretation['top_discriminators'].append({
            'antibiotic': antibiotic,
            'importance_score': score,
            'antibiotic_class': ab_class
        })
        interpretation['antibiotic_classes_involved'].add(ab_class)
    
    # Convert set to list for JSON serialization
    interpretation['antibiotic_classes_involved'] = list(
        interpretation['antibiotic_classes_involved']
    )
    
    # Add interpretation notes
    interpretation['interpretation_notes'].append(
        "Feature importance scores indicate antibiotics contributing most to group separation."
    )
    interpretation['interpretation_notes'].append(
        "Higher scores suggest these antibiotics show more consistent resistance patterns "
        "within categories (e.g., species or MDR status)."
    )
    
    # MDR-related interpretation if applicable
    if target_col and 'MDR' in target_col.upper():
        interpretation['interpretation_notes'].append(
            "For MDR discrimination, top features indicate antibiotics whose resistance "
            "patterns most consistently differentiate MDR from non-MDR isolates."
        )
    
    # Compute resistance rates if dataframe is provided
    if df is not None:
        resistance_stats = {}
        for ab, _ in top_features:
            col_name = f"{ab}_encoded" if f"{ab}_encoded" in df.columns else ab
            if col_name in df.columns:
                # Count resistant (value=2) vs total
                resistance_rate = (df[col_name] == 2).mean()
                resistance_stats[ab] = {
                    'resistance_rate': float(resistance_rate),
                    'interpretation': _get_resistance_interpretation(resistance_rate)
                }
        interpretation['ast_results'] = resistance_stats
    
    return interpretation


def run_all_models(X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray,
                   feature_names: List[str],
                   label_encoder=None) -> Dict:
    """
    Train and evaluate all models.
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Training and test features
    y_train, y_test : np.ndarray
        Training and test labels
    feature_names : list
        List of feature names
    label_encoder : LabelEncoder, optional
        Label encoder
    
    Returns:
    --------
    dict
        Results for all models
    """
    results = {}
    
    for name, model in MODELS.items():
        print(f"  Training {name}...")
        
        # Create fresh model instance
        model_instance = model.__class__(**model.get_params())
        
        # Train
        trained_model = train_model(model_instance, X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(trained_model, X_test, y_test, label_encoder)
        metrics['model_name'] = name
        
        # Feature importance
        metrics['feature_importance'] = get_feature_importance(trained_model, feature_names)
        
        results[name] = {
            'model': trained_model,
            'metrics': metrics
        }
    
    return results


def compare_models(results: Dict) -> pd.DataFrame:
    """
    Compare performance of all models.
    
    Parameters:
    -----------
    results : dict
        Results from run_all_models
    
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    comparison = []
    
    for name, result in results.items():
        metrics = result['metrics']
        comparison.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        })
    
    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
    
    return df_comparison


def run_supervised_pipeline(df: pd.DataFrame,
                            feature_cols: List[str],
                            target_col: str,
                            test_size: float = 0.2) -> Dict:
    """
    Main supervised learning pipeline (Phase 4).
    
    This pipeline implements supervised learning for pattern discrimination,
    evaluating how well resistance fingerprints discriminate known categories.
    
    Note: This is NOT for forecasting or predicting future outcomes. The metrics
    quantify how consistently resistance patterns align with known categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names (resistance fingerprints)
    target_col : str
        Name of target column (species or MDR category)
    test_size : float
        Test set proportion (default 0.2 = 80%-20% split)
    
    Returns:
    --------
    dict
        Complete pipeline results
    """
    print("=" * 50)
    print("PHASE 4: Supervised Learning for Pattern Discrimination")
    print("=" * 50)
    
    print(f"\n1. Target variable: {target_col}")
    
    # Prepare data
    print(f"2. Preparing data (test size: {test_size*100:.0f}%)...")
    X_train, X_test, y_train, y_test, label_encoder, scaler = prepare_data_for_classification(
        df, feature_cols, target_col, test_size
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    if label_encoder:
        print(f"   Classes: {list(label_encoder.classes_)}")
    
    # Get feature names
    feature_names = [c.replace('_encoded', '') for c in feature_cols 
                     if c in df.columns]
    
    # Train and evaluate models
    print("\n3. Training and evaluating models...")
    model_results = run_all_models(
        X_train, X_test, y_train, y_test, feature_names, label_encoder
    )
    
    # Compare models
    print("\n4. Model Comparison:")
    comparison_df = compare_models(model_results)
    print(comparison_df.to_string(index=False))
    
    # Get best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = model_results[best_model_name]['model']
    
    print(f"\n5. Best performing model: {best_model_name}")
    print(f"   F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
    
    # Feature importance from best model
    print("\n6. Feature Importance (Top 10):")
    feature_imp = model_results[best_model_name]['metrics']['feature_importance']
    for i, (feat, imp) in enumerate(list(feature_imp.items())[:10]):
        print(f"   {i+1}. {feat}: {imp:.4f}")
    
    # Phase 4.6: Model Interpretation
    print("\n7. Model Interpretation (Phase 4.6):")
    interpretation = interpret_feature_importance(feature_imp, df, target_col)
    
    print("   Top discriminating antibiotics by class:")
    for disc in interpretation['top_discriminators']:
        print(f"   - {disc['antibiotic']} ({disc['antibiotic_class']}): {disc['importance_score']:.4f}")
    
    if 'ast_results' in interpretation:
        print("\n   AST Results Context:")
        for ab, stats in interpretation['ast_results'].items():
            print(f"   - {ab}: {stats['resistance_rate']*100:.1f}% resistance rate ({stats['interpretation']})")
    
    print("\n   Interpretation Notes:")
    for note in interpretation['interpretation_notes']:
        print(f"   * {note}")
    
    # Complete results
    pipeline_results = {
        'target_variable': target_col,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'classes': label_encoder.classes_.tolist() if label_encoder else None,
        'model_results': {name: result['metrics'] for name, result in model_results.items()},
        'comparison': comparison_df.to_dict('records'),
        'best_model': {
            'name': best_model_name,
            'metrics': model_results[best_model_name]['metrics'],
            'model_object': best_model
        },
        'scaler': scaler,
        'label_encoder': label_encoder,
        'interpretation': interpretation
    }
    
    return pipeline_results


def run_species_discrimination(df: pd.DataFrame,
                               feature_cols: List[str]) -> Dict:
    """
    Run supervised learning for species discrimination.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    
    Returns:
    --------
    dict
        Pipeline results for species discrimination
    """
    print("\n" + "=" * 50)
    print("SPECIES DISCRIMINATION ANALYSIS")
    print("=" * 50)
    
    return run_supervised_pipeline(df, feature_cols, 'ISOLATE_ID')


def run_mdr_discrimination(df: pd.DataFrame,
                           feature_cols: List[str]) -> Dict:
    """
    Run supervised learning for MDR discrimination.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    
    Returns:
    --------
    dict
        Pipeline results for MDR discrimination
    """
    print("\n" + "=" * 50)
    print("MDR DISCRIMINATION ANALYSIS")
    print("=" * 50)
    
    return run_supervised_pipeline(df, feature_cols, 'MDR_CATEGORY')


def save_model(model, scaler, label_encoder, output_path: str):
    """
    Save trained model and preprocessors.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    scaler : StandardScaler
        Fitted scaler
    label_encoder : LabelEncoder
        Fitted label encoder
    output_path : str
        Path to save the model
    """
    import joblib
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    
    joblib.dump(model_data, output_path)
    print(f"Model saved to: {output_path}")


def load_model(model_path: str) -> Dict:
    """
    Load saved model and preprocessors.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
    
    Returns:
    --------
    dict
        Dictionary with model, scaler, and label_encoder
    """
    import joblib
    return joblib.load(model_path)


if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    analysis_path = project_root / "data" / "processed" / "analysis_ready_dataset.csv"
    
    if analysis_path.exists():
        df = pd.read_csv(analysis_path)
        feature_cols = [c for c in df.columns if c.endswith('_encoded')]
        
        # Run MDR discrimination
        if 'MDR_CATEGORY' in df.columns:
            mdr_results = run_mdr_discrimination(df, feature_cols)
            
            # Save best model
            models_dir = project_root / "data" / "models"
            models_dir.mkdir(exist_ok=True)
            
            save_model(
                mdr_results['best_model']['model_object'],
                mdr_results['scaler'],
                mdr_results['label_encoder'],
                str(models_dir / "mdr_classifier.joblib")
            )
        
        # Run species discrimination
        if 'ISOLATE_ID' in df.columns and df['ISOLATE_ID'].nunique() > 1:
            species_results = run_species_discrimination(df, feature_cols)
            
            save_model(
                species_results['best_model']['model_object'],
                species_results['scaler'],
                species_results['label_encoder'],
                str(models_dir / "species_classifier.joblib")
            )
    else:
        print(f"Analysis-ready dataset not found at {analysis_path}")
