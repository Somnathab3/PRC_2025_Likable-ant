import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
import tqdm
import pickle
import glob

# Suppress warnings
warnings.filterwarnings("ignore")

# Features that must never be negative (extend as needed)
NON_NEGATIVE_EXACT = {
    # Fuel totals / consumed
    'flt_total_fuel_acropole_kg',
    'flt_total_fuel_openap_kg',
    'int_fuel_consumed_acropole_kg',
    'int_fuel_consumed_openap_kg',
    'int_cum_fuel_acropole_kg',
    'int_cum_fuel_openap_kg',

    # Distances
    'int_ground_distance_nm',
    'int_alongtrack_distance_nm',
    'int_crosstrack_distance_nm',

    # Mass
    'int_mass_start_kg',
    'int_mass_end_kg',
    'int_mass_mean',
    
    # You can add more here (e.g. powers, energies) if you know
    # they are strictly non-negative in your feature design.
}

def is_non_negative_feature(col: str) -> bool:
    """Return True if this feature should be clipped to >= 0."""
    name = col.lower()

    # Explicit exact-list first
    if col in NON_NEGATIVE_EXACT:
        return True

    # Pattern-based rules (avoid diffs / devs / ratios)
    if 'fuel_consumed' in name and 'diff' not in name:
        return True
    if 'cum_fuel' in name:
        return True
    if name.endswith('fuel_kg') and 'diff' not in name:
        return True

    if 'distance_nm' in name and 'diff' not in name and 'dev' not in name:
        return True

    if 'mass_' in name and 'ratio' not in name and 'diff' not in name:
        return True

    # Speeds
    if 'speed' in name or 'groundspeed' in name or 'tas' in name or 'cas' in name or 'mach' in name:
        return True

    # Standard deviations, ranges, IQR, CV
    if 'std' in name or 'range' in name or 'iqr' in name or 'cv' in name:
        return True

    # Thrust, drag, power
    if 'thrust' in name or 'drag' in name or 'power' in name:
        return True

    # Altitudes (typically >=0)
    if 'altitude' in name:
        return True

    # Fuel flows and rates
    if 'fuel_flow' in name or 'fuel_rate' in name:
        return True

    # Densities, humidities, visibilities, etc.
    if 'density' in name or 'humidity' in name or 'vsby' in name or 'alti' in name or 'p01i' in name:
        return True

    # Multipliers and ratios (but avoid diffs)
    if 'multiplier' in name and 'diff' not in name:
        return True

    return False

# Paths
base_path = Path("F:/PRC_2025/Likable-ant_v1/data/processed/Stage_8_Consolidated_Features")
tow_base_path = Path("F:/PRC_2025/Likable-ant_v1/data/processed/Stage_4_tow_predictions/tow_features")
logs_path = Path("F:/PRC_2025/Likable-ant_v1/logs/feature_importances")
logs_path.mkdir(parents=True, exist_ok=True)
checkpoint_path = base_path / "tow_merge_imputation_lightgbm_checkpoint.parquet"
models_path = logs_path / "tow_merge_lightgbm_models"
models_path.mkdir(parents=True, exist_ok=True)

# Predictors
predictors = [
    'interval_rel_position_midpoint', 'time_since_takeoff_min', 'origin_dest_gcdist_nm', 'interval_rel_span', 'interval_duration_sec',
    'int_time_since_takeoff_sec', 'int_time_before_landing_sec', 'int_relative_position_in_flight',
    'tow_pred_kg', 'total_flight_duration_sec', 'aircraft_type_encoded'
]

def load_and_merge_with_tow(split):
    """Load consolidated and merge with tow features for a split."""
    # Load consolidated
    consolidated = pd.read_parquet(base_path / f"consolidated_features_{split}_imputed_lightgbm.parquet")
    
    # Load all tow files for this split
    tow_files = glob.glob(str(tow_base_path / split / "*.parquet"))
    print(f"Loading {len(tow_files)} tow files for {split}...")
    tow_dfs = [pd.read_parquet(f) for f in tow_files]
    tow_df = pd.concat(tow_dfs, ignore_index=True)
    
    # Filter tow columns: not in consolidated (no NaN filter, include all)
    valid_tow_cols = [c for c in tow_df.columns if c not in consolidated.columns]
    
    print(f"For {split}: {len(valid_tow_cols)} tow features to add")
    
    # Merge on flight_id
    consolidated = consolidated.merge(tow_df[['flight_id'] + valid_tow_cols], on='flight_id', how='left')
    
    return consolidated, valid_tow_cols

def load_and_merge_data():
    """Load and merge all datasets with tow into one."""
    splits = ['train', 'rank', 'final']
    dfs = []
    all_valid_tow_cols = set()
    for split in splits:
        df, valid_tow_cols = load_and_merge_with_tow(split)
        df['dataset_split'] = split
        dfs.append(df)
        all_valid_tow_cols.update(valid_tow_cols)
    df = pd.concat(dfs, ignore_index=True)

    # Encode 'aircraft_type' if present
    if 'aircraft_type' in df.columns:
        le = LabelEncoder()
        df['aircraft_type_encoded'] = le.fit_transform(df['aircraft_type'].astype(str))
        df = df.drop(columns=['aircraft_type'])

    return df, list(all_valid_tow_cols)

def clean_data(df):
    """Clean the data as specified."""

    # Replace inf/-inf with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop segment-based data columns, except ac_ data, predictors, and those with NaNs (to impute them)
    #cols_to_drop = [col for col in df.columns if col.startswith('seg') and not col.startswith('ac_') and col not in predictors and df[col].isnull().sum() == 0]
    #df = df.drop(columns=cols_to_drop, errors='ignore')

    # Drop specific columns
    df = df.drop(columns=['fuel_kg_actual'], errors='ignore')

    return df

def clean_duplicates_and_sparse(df, valid_tow_cols):
    """Clean duplicates (_x _y), sparse features, etc."""
    # Drop columns with _y (duplicates from merge)
    cols_to_drop = [col for col in df.columns if col.endswith('_y')]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Rename _x to original
    rename_dict = {col: col[:-2] for col in df.columns if col.endswith('_x')}
    df = df.rename(columns=rename_dict)
    
    # Update valid_tow_cols to existing columns
    valid_tow_cols = [c for c in valid_tow_cols if c in df.columns]
    
    # Drop sparse features: NaN > 80% in tow features
    nan_pct = df[valid_tow_cols].isnull().mean()
    sparse_cols = [c for c in valid_tow_cols if nan_pct[c] > 0.8]
    df = df.drop(columns=sparse_cols, errors='ignore')
    
    print(f"Dropped {len(sparse_cols)} sparse tow features")
    
    return df

def list_nans(df, valid_tow_cols):
    """List features with NaNs, prioritizing tow features."""
    nan_counts = df.isnull().sum()
    features_with_nans = nan_counts[nan_counts > 0].sort_values(ascending=False)
    # Only keep numeric columns
    features_with_nans = features_with_nans[features_with_nans.index.map(lambda col: pd.api.types.is_numeric_dtype(df[col]))]
    # Prioritize tow features
    tow_nans = [col for col in features_with_nans.index if col in valid_tow_cols]
    other_nans = [col for col in features_with_nans.index if col not in valid_tow_cols]
    return tow_nans + other_nans

def prepare_features(df, features_to_impute):
    """Prepare X and Y for imputation."""
    # X: predictors + ac_ features + flt_ features
    ac_features = [col for col in df.columns if col.startswith('ac_')]
    flt_features = [col for col in df.columns if col.startswith('flt_')]
    X_cols = predictors + ac_features + flt_features
    X = df[X_cols].copy()

    # Handle NaNs in X by filling with median
    for col in X_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # Y: features to impute
    Y = df[features_to_impute].copy()

    return X, Y

def impute_with_lightgbm(df, features_with_nans, batch_size=20):
    """Impute using LightGBM in batches with safety clipping."""
    # Prioritize fuel-related features
    fuel_features = [col for col in features_with_nans if 'fuel' in col.lower() or 'mass' in col.lower()]
    other_features = [col for col in features_with_nans if col not in fuel_features]
    prioritized_features = fuel_features + other_features

    for i in tqdm.tqdm(range(0, len(prioritized_features), batch_size), desc="Processing batches"):
        batch_features = prioritized_features[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch_features)} features")

        # Prepare X and Y
        X, Y_batch = prepare_features(df, batch_features)

        # Train only on rows where *all* targets in the batch are present
        train_mask = Y_batch.notnull().all(axis=1)
        X_train = X[train_mask]
        Y_train = Y_batch[train_mask]

        if len(Y_train) == 0:
            print(f"No complete training data for batch {i//batch_size + 1}, skipping")
            continue

        # Precompute training medians and quantile bounds for safety clipping
        medians = {}
        bounds = {}
        for col in batch_features:
            vals = Y_train[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                medians[col] = 0.0
                bounds[col] = (None, None)
                continue

            medians[col] = float(np.median(vals))
            if vals.size >= 100:
                low, high = np.quantile(vals, [0.01, 0.99])
                # Ensure valid bounds
                if np.isfinite(low) and np.isfinite(high) and low < high:
                    bounds[col] = (float(low), float(high))
                else:
                    bounds[col] = (None, None)
            else:
                bounds[col] = (None, None)

        # Train multi-output LightGBM model
        model = MultiOutputRegressor(LGBMRegressor(
            n_estimators=200,
            max_depth=12,
            num_leaves=50,
            learning_rate=0.1,
            random_state=42,
            device='gpu'
        ))
        model.fit(X_train, Y_train)

        # Predict for all rows
        Y_pred = model.predict(X)  # shape: (n_rows, len(batch_features))

        # Fill NaNs in df with clipped predictions
        for j, col in enumerate(batch_features):
            preds = np.asarray(Y_pred[:, j], dtype=float)

            # 1) Replace non-finite predictions with training median
            preds[~np.isfinite(preds)] = medians[col]

            # 2) Clip to training quantile range if available
            low, high = bounds.get(col, (None, None))
            if low is not None and high is not None:
                preds = np.clip(preds, low, high)

            # 3) Enforce non-negativity for critical features
            if is_non_negative_feature(col):
                preds = np.clip(preds, 0.0, None)

            nan_mask = df[col].isnull()
            df.loc[nan_mask, col] = preds[nan_mask]

        # Save model
        model_file = models_path / f"tow_merge_lightgbm_batch_{i//batch_size + 1}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

        # Save checkpoint
        df.to_parquet(checkpoint_path, index=False)

    return df

if __name__ == '__main__':
    # Check for checkpoint
    if checkpoint_path.exists():
        df = pd.read_parquet(checkpoint_path)
        # Assume valid_tow_cols are all columns not in original consolidated, but to simplify, impute all nans
        features_with_nans = list_nans(df, [])  # Pass empty to impute all
    else:
        # Load and merge data with tow
        df, valid_tow_cols = load_and_merge_data()

        # Drop leakage target column if any (assuming 'fuel_consumed' or similar, but adjust if needed)
        # For now, assuming no specific leakage column, but you can add: df = df.drop(columns=['leakage_col'], errors='ignore')

        # Clean data
        df = clean_data(df)
        
        # Clean duplicates and sparse
        df = clean_duplicates_and_sparse(df, valid_tow_cols)
        
        print(f"Total features after cleaning: {len(df.columns)}")
        
        # List NaNs, prioritizing tow features
        features_with_nans = list_nans(df, valid_tow_cols)

    # Impute with LightGBM
    df = impute_with_lightgbm(df, features_with_nans)

    # Calculate per-feature stats
    stats = []
    for col in features_with_nans:
        values = df[col].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            rmse_val = np.sqrt(np.mean((values - mean_val)**2))  # RMSE as std from mean
            stats.append({'feature': col, 'mean': mean_val, 'rmse': rmse_val})
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(logs_path / "tow_merge_imputation_feature_stats.csv", index=False)

    # Final save
    train_imputed = df[df['dataset_split'] == 'train'].drop(columns=['dataset_split'])
    rank_imputed = df[df['dataset_split'] == 'rank'].drop(columns=['dataset_split'])
    final_imputed = df[df['dataset_split'] == 'final'].drop(columns=['dataset_split'])

    train_imputed.to_parquet(base_path / "consolidated_features_train_with_tow_imputed_lightgbm.parquet", index=False)
    rank_imputed.to_parquet(base_path / "consolidated_features_rank_with_tow_imputed_lightgbm.parquet", index=False)
    final_imputed.to_parquet(base_path / "consolidated_features_final_with_tow_imputed_lightgbm.parquet", index=False)

    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("Tow merge and LightGBM imputation complete. Saved all datasets.")