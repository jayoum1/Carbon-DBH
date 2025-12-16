"""
Neural Network DBH Growth Model

This module provides an alternative to the XGBoost model (dbh_growth_model.py) using
a feedforward neural network. The primary motivation is to address the fixed-point
behavior observed in tree-based models during iterative multi-year simulation.

WHY NEURAL NETWORKS:
    Tree-based models (XGBoost) create piecewise constant functions with discrete
    decision boundaries. During iterative simulation, many trees converge to fixed
    points where the model predicts zero growth. This may not reflect realistic
    biological growth patterns.

    Neural networks, with their smooth activation functions, produce continuous
    predictions that may better capture gradual growth trajectories over multiple
    years. However, this comes with a tradeoff: neural networks may have different
    predictive accuracy compared to XGBoost, and the choice between accuracy and
    trajectory realism is an active area of evaluation.

MODEL ARCHITECTURE:
    - Input layer: Same features as XGBoost model
    - Hidden layer 1: 64 units with ReLU activation
    - Hidden layer 2: 32 units with ReLU activation
    - Output layer: 1 unit (DBH prediction)
    - Regularization: L2 (alpha ~ 1e-4 to 1e-3)
    - Early stopping: Enabled to prevent overfitting
    - Feature scaling: StandardScaler (fit on training data only)

FEATURES:
    - PrevDBH_cm: Previous year's DBH (required)
    - Species_*: One-hot encoded species features
    - Plot_*: One-hot encoded plot features
    - GapYears: Years between measurements (default: 1.0)
    - Group_*: Tree group features (if available)

TARGET:
    - DBH_cm: This year's DBH (predicted from PrevDBH_cm and features)

Usage Example:
    >>> from src.models.dbh_growth_nn import predict_dbh_next_year_nn
    >>> next_dbh = predict_dbh_next_year_nn(
    ...     prev_dbh_cm=25.0,
    ...     species='red oak',
    ...     plot='Upper'
    ... )
    >>> print(f"Predicted DBH next year: {next_dbh:.2f} cm")
"""

import pandas as pd
import numpy as np
import sys
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CARBON_ALL_PLOTS_ENCODED, MODELS_DIR

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model file paths
MODEL_PATH = MODELS_DIR / "dbh_growth_nn_model.pkl"
SCALER_PATH = MODELS_DIR / "dbh_growth_nn_scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "dbh_growth_nn_features.txt"

# Global variables for caching loaded model
_loaded_model = None
_loaded_scaler = None
_loaded_feature_names = None


def load_and_prepare_data():
    """
    Load the processed dataset and prepare it for training.
    
    This function mirrors the XGBoost model's data loading logic.
    
    Returns:
        df: DataFrame with PrevDBH_cm created and ready for modeling
    """
    print("Loading dataset...")
    df = pd.read_csv(str(CARBON_ALL_PLOTS_ENCODED))
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    
    # Sort by TreeID and Year to ensure proper ordering
    df = df.sort_values(['TreeID', 'Year']).copy()
    
    # Create PrevDBH_cm from DBH_cm (shift within each TreeID group)
    if 'PrevDBH_cm' not in df.columns:
        print("Creating PrevDBH_cm column...")
        df['PrevDBH_cm'] = df.groupby('TreeID')['DBH_cm'].shift()
        print(f"Created PrevDBH_cm. Rows with PrevDBH_cm: {df['PrevDBH_cm'].notna().sum():,}")
    
    return df


def select_features(df):
    """
    Select features for the model.
    
    This function uses the same feature selection logic as the XGBoost model.
    
    Features included:
        - PrevDBH_cm: Previous year's DBH (required)
        - Species_*: One-hot encoded species columns
        - Plot_*: One-hot encoded plot columns
        - GapYears: Years between measurements (if available)
        - Group_*: Tree group (softwood/hardwood)
        - GrowthType_*: Growth type indicators (if available)
    
    Parameters:
        df: DataFrame with all columns
    
    Returns:
        X: Feature matrix (DataFrame)
        y: Target vector (Series)
        feature_names: List of feature names
    """
    print("\nSelecting features...")
    
    # Target variable
    y = df['DBH_cm'].copy()
    
    # Start with PrevDBH_cm (required feature)
    feature_cols = ['PrevDBH_cm']
    
    # Add one-hot encoded species columns
    species_cols = [col for col in df.columns if col.startswith('Species_')]
    feature_cols.extend(species_cols)
    print(f"  Added {len(species_cols)} species features")
    
    # Add one-hot encoded plot columns
    plot_cols = [col for col in df.columns if col.startswith('Plot_')]
    feature_cols.extend(plot_cols)
    print(f"  Added {len(plot_cols)} plot features")
    
    # Add GapYears if available (years between measurements)
    if 'GapYears' in df.columns:
        feature_cols.append('GapYears')
        print("  Added GapYears feature")
    
    # Add Group columns (softwood/hardwood)
    group_cols = [col for col in df.columns if col.startswith('Group_')]
    feature_cols.extend(group_cols)
    if group_cols:
        print(f"  Added {len(group_cols)} group features")
    
    # Add GrowthType columns if available
    growthtype_cols = [col for col in df.columns if col.startswith('GrowthType_')]
    feature_cols.extend(growthtype_cols)
    if growthtype_cols:
        print(f"  Added {len(growthtype_cols)} growth type features")
    
    # Extract feature matrix
    X = df[feature_cols].copy()
    
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"Feature names: {feature_cols}")
    
    return X, y, feature_cols


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a neural network model with feature scaling and early stopping.
    
    Architecture:
        Input → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1)
    
    Parameters:
        X: Feature matrix (DataFrame)
        y: Target vector (Series)
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        model: Trained MLPRegressor model
        scaler: Fitted StandardScaler
        X_train, X_test, y_train, y_test: Train/test splits
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK DBH GROWTH MODEL")
    print("="*60)
    
    # Filter rows where both PrevDBH_cm and DBH_cm are not null
    valid_mask = X['PrevDBH_cm'].notna() & y.notna()
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    
    # Fill any remaining NaN values in features with 0 (for one-hot encoded columns, NaN = 0)
    X_clean = X_clean.fillna(0.0)
    
    print(f"Rows with valid PrevDBH_cm and DBH_cm: {len(X_clean):,}")
    print(f"Rows dropped (missing PrevDBH_cm or DBH_cm): {(~valid_mask).sum():,}")
    print(f"NaN values in features after filtering: {X_clean.isna().sum().sum()}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Create and fit scaler on training data only
    print("\nFitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Feature scaling complete")
    
    # Create neural network model
    print("\nTraining neural network model...")
    print("  Architecture: Input → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1)")
    print("  Regularization: L2 (alpha=1e-3)")
    print("  Early stopping: Enabled")
    
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),  # Two hidden layers: 64 and 32 units
        activation='relu',
        solver='adam',
        alpha=1e-3,  # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,  # Use 10% of training data for validation
        n_iter_no_change=20,  # Stop if no improvement for 20 iterations
        random_state=random_state,
        verbose=True
    )
    
    model.fit(X_train_scaled, y_train)
    
    print("✓ Model trained successfully")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred_test = model.predict(X_test_scaled)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = np.mean(np.abs(y_test - y_pred_test))
    
    print(f"\nTest Set Metrics:")
    print(f"  R² Score:  {r2_test:.4f}")
    print(f"  RMSE:     {rmse_test:.4f} cm")
    print(f"  MAE:      {mae_test:.4f} cm")
    
    # Get feature names
    feature_names = X_train.columns.tolist()
    
    return model, scaler, X_train, X_test, y_train, y_test, feature_names


def save_model(model, scaler, feature_names):
    """
    Save the trained model, scaler, and feature names to disk.
    
    Parameters:
        model: Trained MLPRegressor model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
    """
    print("\nSaving model and scaler...")
    
    # Save model
    joblib.dump(model, str(MODEL_PATH))
    print(f"✓ Saved model to {MODEL_PATH}")
    
    # Save scaler
    joblib.dump(scaler, str(SCALER_PATH))
    print(f"✓ Saved scaler to {SCALER_PATH}")
    
    # Save feature names
    with open(str(FEATURE_NAMES_PATH), 'w') as f:
        for feat_name in feature_names:
            f.write(f"{feat_name}\n")
    print(f"✓ Saved feature names to {FEATURE_NAMES_PATH}")


def load_dbh_growth_model_nn():
    """
    Load and return the trained neural network model, scaler, and feature names.
    
    Returns:
        model: Trained MLPRegressor model
        scaler: Fitted StandardScaler
        feature_names: List of feature names in the order used for training
    """
    global _loaded_model, _loaded_scaler, _loaded_feature_names
    
    # Return cached model if already loaded
    if _loaded_model is not None and _loaded_scaler is not None and _loaded_feature_names is not None:
        return _loaded_model, _loaded_scaler, _loaded_feature_names
    
    # Load model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please train the model first by running this script."
        )
    
    _loaded_model = joblib.load(str(MODEL_PATH))
    
    # Load scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}\n"
            "Please train the model first by running this script."
        )
    
    _loaded_scaler = joblib.load(str(SCALER_PATH))
    
    # Load feature names
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Feature names file not found: {FEATURE_NAMES_PATH}\n"
            "Please train the model first by running this script."
        )
    
    with open(str(FEATURE_NAMES_PATH), 'r') as f:
        _loaded_feature_names = [line.strip() for line in f.readlines()]
    
    print(f"✓ Loaded model from {MODEL_PATH}")
    print(f"✓ Loaded scaler from {SCALER_PATH}")
    print(f"✓ Loaded {len(_loaded_feature_names)} feature names")
    
    return _loaded_model, _loaded_scaler, _loaded_feature_names


def predict_dbh_next_year_nn(prev_dbh_cm, species=None, plot=None, gap_years=1.0, **kwargs):
    """
    Predict next year's DBH for a single tree using the neural network model.
    
    This function mirrors the XGBoost model's prediction API.
    
    IMPORTANT INTERPRETATION:
    - During training, `PrevDBH_cm` was last year's DBH and `DBH_cm` was this year's DBH.
    - In the app, we will pass the user's *current* DBH as `prev_dbh_cm`.
    - Therefore, the output of this function will be interpreted as NEXT YEAR'S DBH in the app.
    
    Parameters:
        prev_dbh_cm: float, current DBH (treated as PrevDBH_cm in the model).
        species: str, species identifier (e.g., "red oak", "sugar maple").
                  If None, will use the most common species in training data.
        plot: str, plot identifier ("Upper", "Middle", or "Lower").
               If None, will use "Lower" (reference category).
        gap_years: float, years between measurements (default: 1.0 for annual prediction).
        **kwargs: optional additional features if needed (e.g., group, growthtype).
    
    Returns:
        predicted_dbh_next_year_cm: float, predicted DBH for next year (in cm).
    """
    # Load model, scaler, and feature names
    model, scaler, feature_names = load_dbh_growth_model_nn()
    
    # Create a single-row feature vector
    feature_dict = {}
    
    # Required feature: PrevDBH_cm
    feature_dict['PrevDBH_cm'] = prev_dbh_cm
    
    # Initialize all features to 0 (numeric, not boolean)
    for feat_name in feature_names:
        if feat_name not in feature_dict:
            feature_dict[feat_name] = 0.0
    
    # Set species feature
    if species is not None:
        species_col = f'Species_{species.lower()}'
        if species_col in feature_names:
            feature_dict[species_col] = 1.0  # Numeric 1, not boolean True
    
    # Set plot feature
    # Note: "Lower" is the reference category (dropped during one-hot encoding)
    # So if plot is "Lower" or None, all Plot_* columns remain 0
    if plot is not None and plot.lower() != 'lower':
        # Normalize plot name: capitalize first letter to match feature names (e.g., "middle" -> "Middle")
        plot_normalized = plot.capitalize()
        plot_col = f'Plot_{plot_normalized}'
        if plot_col in feature_names:
            feature_dict[plot_col] = 1.0  # Numeric 1, not boolean True
    
    # Set GapYears
    if 'GapYears' in feature_names:
        feature_dict['GapYears'] = gap_years
    
    # Set any additional kwargs
    for key, value in kwargs.items():
        if key in feature_names:
            feature_dict[key] = value
    
    # Build feature vector as a DataFrame with proper column names
    feature_df = pd.DataFrame([feature_dict], columns=feature_names)
    
    # Ensure we only use selected features
    feature_df = feature_df[feature_names]
    
    # Scale features
    feature_scaled = scaler.transform(feature_df)
    
    # Predict
    predicted_dbh = model.predict(feature_scaled)[0]
    
    # Ensure non-negative prediction (DBH cannot be negative)
    predicted_dbh = max(0.0, predicted_dbh)
    
    return predicted_dbh


def compare_trajectories():
    """
    Diagnostic function to compare XGBoost and Neural Network trajectories.
    
    This function simulates 5 example trees forward 10 years using both models
    and prints side-by-side trajectories for comparison.
    """
    print("\n" + "="*70)
    print("TRAJECTORY COMPARISON: XGBoost vs Neural Network")
    print("="*70)
    
    # Import XGBoost prediction function
    from models.dbh_growth_model import predict_dbh_next_year
    
    # Import neural network prediction function (already in this module)
    
    # Load base forest
    from models.forest_snapshots import load_base_forest_df
    base_forest = load_base_forest_df()
    
    # Select 5 random example trees
    np.random.seed(42)
    sample_indices = np.random.choice(len(base_forest), size=min(5, len(base_forest)), replace=False)
    
    print(f"\nComparing trajectories for {len(sample_indices)} example trees:")
    print("-" * 70)
    
    for idx in sample_indices:
        row = base_forest.iloc[idx]
        treeid = row['TreeID']
        species = row['Species']
        plot = row['Plot']
        initial_dbh = row['DBH_cm']
        
        print(f"\nTreeID {treeid} ({species}, {plot}):")
        print(f"  Initial DBH: {initial_dbh:.4f} cm")
        print()
        print(f"{'Year':<6} | {'DBH_XGB':<12} | {'DBH_NN':<12} | {'Delta':<12}")
        print("-" * 50)
        
        # Simulate forward 10 years with both models
        dbh_xgb = initial_dbh
        dbh_nn = initial_dbh
        
        print(f"{0:<6} | {dbh_xgb:<12.6f} | {dbh_nn:<12.6f} | {0.0:<12.6f}")
        
        for year in range(1, 11):
            # XGBoost prediction
            dbh_xgb = predict_dbh_next_year(
                prev_dbh_cm=dbh_xgb,
                species=species,
                plot=plot,
                gap_years=1.0,
                silent=True
            )
            
            # Neural network prediction
            dbh_nn = predict_dbh_next_year_nn(
                prev_dbh_cm=dbh_nn,
                species=species,
                plot=plot,
                gap_years=1.0
            )
            
            delta = dbh_nn - dbh_xgb
            print(f"{year:<6} | {dbh_xgb:<12.6f} | {dbh_nn:<12.6f} | {delta:<12.6f}")
    
    print("\n" + "="*70)


# ==============================================================================
# Main Training Script
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("NEURAL NETWORK DBH GROWTH MODEL TRAINING")
    print("="*60)
    
    # 1. Load and prepare data
    df = load_and_prepare_data()
    
    # 2. Select features
    X, y, feature_names = select_features(df)
    
    # 3. Train model
    model, scaler, X_train, X_test, y_train, y_test, feature_names = train_model(X, y)
    
    # 4. Save model
    save_model(model, scaler, feature_names)
    
    # 5. Compare trajectories (if XGBoost model exists)
    print("\n" + "="*60)
    print("TRAJECTORY COMPARISON")
    print("="*60)
    
    try:
        compare_trajectories()
    except Exception as e:
        print(f"\n⚠ Could not run trajectory comparison: {e}")
        print("  (This is expected if the XGBoost model hasn't been trained yet)")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  - Compare model performance with XGBoost")
    print("  - Evaluate trajectory smoothness in multi-year simulations")
    print("  - Consider ensemble approaches if both models have strengths")

