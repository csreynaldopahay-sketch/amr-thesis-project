"""
Supervised Learning Module for AMR Thesis Project
Phase 4 - Pattern Discrimination using Machine Learning Models
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


def prepare_data_for_classification(df: pd.DataFrame,
                                    feature_cols: List[str],
                                    target_col: str,
                                    test_size: float = 0.2,
                                    random_state: int = 42) -> Tuple:
    """
    Prepare data for supervised learning.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features and target
    feature_cols : list
        List of feature column names
    target_col : str
        Name of target column
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, label_encoder, scaler)
    """
    # Remove rows with missing target
    df_valid = df[df[target_col].notna()].copy()
    
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
    Evaluate model performance.
    
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
    Extract feature importance from model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    feature_names : list
        List of feature names
    
    Returns:
    --------
    dict
        Feature importance scores
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
    Main supervised learning pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    target_col : str
        Name of target column
    test_size : float
        Test set proportion
    
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
        'label_encoder': label_encoder
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
