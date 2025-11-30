import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
import tqdm
import pickle

# Suppress warnings
warnings.filterwarnings("ignore")

# Checkpoint semantics:
# - A checkpoint file exists at `imputation_lightgbm_checkpoint.parquet`.
# - We consider Stage A complete when all train rows have no NaNs for the base fuel
#   targets: 'int_fuel_consumed_acropole_kg' and 'int_fuel_consumed_openap_kg'.
# - If Stage A is done according to this rule, the script will proceed directly to Stage B
#   (global imputation across train, rank, final); otherwise it will run Stage A first.

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
logs_path = Path("F:/PRC_2025/Likable-ant_v1/logs/feature_importances")
logs_path.mkdir(parents=True, exist_ok=True)
checkpoint_path = base_path / "imputation_lightgbm_checkpoint.parquet"
models_path = logs_path / "lightgbm_models"
models_path.mkdir(parents=True, exist_ok=True)

# Predictors
BASE_PREDICTORS = [
    'interval_rel_position_midpoint', 'time_since_takeoff_min', 'origin_dest_gcdist_nm', 'interval_rel_span', 'interval_duration_sec',
    'int_time_since_takeoff_sec', 'int_time_before_landing_sec', 'int_relative_position_in_flight',
    'tow_pred_kg', 'total_flight_duration_sec', 'aircraft_type_encoded'
]

# For Stage A (train-only), allow the ground-truth per-interval fuel as an extra predictor
TRAIN_FUEL_AWARE_PREDICTORS = BASE_PREDICTORS + ['fuel_kg']

# Backwards-compatible alias used in older functions
predictors = BASE_PREDICTORS

def load_and_merge_data():
    """Load and merge the datasets."""
    train_df = pd.read_parquet(base_path / "consolidated_features_train_multiplier.parquet")
    rank_df = pd.read_parquet(base_path / "consolidated_features_rank_multiplier.parquet")
    final_df = pd.read_parquet(base_path / "consolidated_features_final_multiplier.parquet")
    train_df['dataset_split'] = 'train'
    rank_df['dataset_split'] = 'rank'
    final_df['dataset_split'] = 'final'
    df = pd.concat([train_df, rank_df, final_df], ignore_index=True)

    # Encode 'aircraft_type' if present
    if 'aircraft_type' in df.columns:
        le = LabelEncoder()
        df['aircraft_type_encoded'] = le.fit_transform(df['aircraft_type'].astype(str))
        df = df.drop(columns=['aircraft_type'])

    return df


def add_fuel_kg_to_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge interval fuel_kg from raw fuel_train.parquet into the train_df.
    """
    fuel_path = Path("F:/PRC_2025/Likable-ant_v1/data/raw/fuel_train.parquet")
    if not fuel_path.exists():
        print(f"Fuel file not found at {fuel_path}, returning train_df unchanged")
        return train_df

    fuel_df = pd.read_parquet(fuel_path)
    # Use common subset of columns
    cols = [c for c in ['flight_id', 'idx', 'start', 'end', 'fuel_kg'] if c in fuel_df.columns]
    fuel_subset = fuel_df[cols].copy()

    # Try merging on (flight_id, idx) when possible
    if 'idx' in train_df.columns and 'idx' in fuel_subset.columns:
        merged = train_df.merge(fuel_subset[['flight_id', 'idx', 'fuel_kg']], on=['flight_id', 'idx'], how='left')
    else:
        # Fallback: merge on timestamps if they exist
        if 'interval_start_ts' in train_df.columns and 'start' in fuel_subset.columns:
            try:
                train_df['interval_start_ts'] = pd.to_datetime(train_df['interval_start_ts'])
                fuel_subset['start'] = pd.to_datetime(fuel_subset['start'])
                merged = train_df.merge(fuel_subset[['flight_id', 'start', 'fuel_kg']], left_on=['flight_id', 'interval_start_ts'], right_on=['flight_id', 'start'], how='left')
                merged = merged.drop(columns=['start'], errors='ignore')
            except Exception:
                merged = train_df.merge(fuel_subset[['flight_id', 'fuel_kg']], on='flight_id', how='left')
        else:
            # Last resort: merge on flight_id
            merged = train_df.merge(fuel_subset[['flight_id', 'fuel_kg']], on='flight_id', how='left')

    # Prefer the merged fuel_kg; if train had existing fuel_kg_actual, use it to fill missing merged
    if 'fuel_kg_actual' in merged.columns:
        merged['fuel_kg'] = merged['fuel_kg'].fillna(merged['fuel_kg_actual'])

    if 'fuel_kg' in merged.columns:
        merged['fuel_kg'] = pd.to_numeric(merged['fuel_kg'], errors='coerce')
        merged.loc[merged['fuel_kg'] < 0, 'fuel_kg'] = np.nan

    if 'dataset_split' in merged.columns:
        merged.loc[merged['dataset_split'] != 'train', 'fuel_kg'] = np.nan

    return merged

def clean_data(df):
    """Clean the data as specified."""

    # Replace inf/-inf with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop segment-based data columns, except ac_ data, base predictors, and those with NaNs (to impute them)
    keep_cols = set(BASE_PREDICTORS) | set(TRAIN_FUEL_AWARE_PREDICTORS)
    cols_to_drop = [col for col in df.columns if col.startswith('seg') and not col.startswith('ac_') and col not in keep_cols and df[col].isnull().sum() == 0]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Drop specific columns
    df = df.drop(columns=['fuel_kg_actual'], errors='ignore')

    # Set negative values for strictly non-negative features to NaN so they get re-imputed
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if is_non_negative_feature(col):
            try:
                neg_mask = df[col] < 0
                if neg_mask.any():
                    df.loc[neg_mask, col] = np.nan
            except Exception:
                pass

    # For rows with no trajectory data, set trajectory-dependent interval/segment features to NaN
    if 'has_trajectory_data' in df.columns:
        traj_false_mask = df['has_trajectory_data'] == False
        if traj_false_mask.any():
            for col in df.columns:
                # key columns to preserve
                if col in BASE_PREDICTORS or col in TRAIN_FUEL_AWARE_PREDICTORS:
                    continue
                if col.startswith('ac_') or col.startswith('flt_'):
                    continue
                if col in ('idx', 'flight_id', 'dataset_split', 'interval_start_ts', 'interval_end_ts'):
                    continue
                if col.startswith('seg') or col.startswith('int_'):
                    df.loc[traj_false_mask, col] = np.nan

    return df

def list_nans(df):
    """List features with NaNs."""
    nan_counts = df.isnull().sum()
    features_with_nans = nan_counts[nan_counts > 0].sort_values(ascending=False)
    # Only keep numeric columns
    features_with_nans = features_with_nans[features_with_nans.index.map(lambda col: pd.api.types.is_numeric_dtype(df[col]))]
    return features_with_nans.index.tolist()

def prepare_features(df: pd.DataFrame, targets: List[str], predictor_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build X and Y for imputation:
    X: predictor_cols + all ac_ features
    Y: targets (subset of df)
    """
    ac_features = [col for col in df.columns if col.startswith('ac_')]
    X_cols = list(dict.fromkeys(predictor_cols + ac_features))
    X = df[X_cols].copy()

    for col in X_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    Y = df[targets].copy()
    return X, Y


def impute_targets_with_lightgbm(
    df: pd.DataFrame,
    targets: List[str],
    predictor_cols: List[str],
    train_mask: pd.Series,
    batch_size: int = 20
) -> pd.DataFrame:
    """
    Multi-output LGBM imputation for the given targets with predictors.
    """
    # Prioritize fuel/mass targets
    fuel_targets = [t for t in targets if 'fuel' in t.lower() or 'mass' in t.lower()]
    other_targets = [t for t in targets if t not in fuel_targets]
    ordered_targets = fuel_targets + other_targets

    for i in tqdm.tqdm(range(0, len(ordered_targets), batch_size), desc="Processing batches"):
        batch_targets = ordered_targets[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch_targets)} targets")

        X_all, Y_all = prepare_features(df, batch_targets, predictor_cols)
        train_rows = train_mask & Y_all.notnull().all(axis=1)
        X_train = X_all[train_rows]
        Y_train = Y_all[train_rows]

        if len(Y_train) == 0:
            print(f"No complete training data for batch {i//batch_size + 1}, skipping")
            continue

        medians = {}
        bounds = {}
        for col in batch_targets:
            vals = Y_train[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                medians[col] = 0.0
                bounds[col] = (None, None)
                continue
            medians[col] = float(np.median(vals))
            if vals.size >= 100:
                low, high = np.quantile(vals, [0.01, 0.99])
                if np.isfinite(low) and np.isfinite(high) and low < high:
                    bounds[col] = (float(low), float(high))
                else:
                    bounds[col] = (None, None)
            else:
                bounds[col] = (None, None)

        model = MultiOutputRegressor(LGBMRegressor(
            n_estimators=200,
            max_depth=12,
            num_leaves=50,
            learning_rate=0.1,
            random_state=42,
            device='gpu'
        ))
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_all)
        for j, col in enumerate(batch_targets):
            preds = np.asarray(Y_pred[:, j], dtype=float)
            preds[~np.isfinite(preds)] = medians[col]
            low, high = bounds.get(col, (None, None))
            if low is not None and high is not None:
                preds = np.clip(preds, low, high)
            if is_non_negative_feature(col):
                preds = np.clip(preds, 0.0, None)
            nan_mask = df[col].isnull()
            df.loc[nan_mask, col] = preds[nan_mask]

        model_file = models_path / f"lightgbm_batch_{i//batch_size + 1}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

        df.to_parquet(checkpoint_path, index=False)

    return df

def impute_with_lightgbm(df, features_with_nans, batch_size=20, predictor_cols=None, train_mask=None):
    """Compatibility wrapper for older code; uses the new impute_targets_with_lightgbm under the hood.
    predictor_cols defaults to BASE_PREDICTORS (no fuel_kg) and train_mask defaults to dataset_split == 'train'.
    """
    if predictor_cols is None:
        predictor_cols = BASE_PREDICTORS
    if train_mask is None and 'dataset_split' in df.columns:
        train_mask = df['dataset_split'] == 'train'
    elif train_mask is None:
        train_mask = pd.Series(True, index=df.index)

    return impute_targets_with_lightgbm(df, features_with_nans, predictor_cols, train_mask, batch_size=batch_size)


def recompute_fuel_derived_features(df: pd.DataFrame, changed_mask: pd.Series) -> pd.DataFrame:
    """
    Recompute a defined set of fuel-derived features for rows indicated by changed_mask.
    This assumes base fuel columns are now available.
    """
    if changed_mask.sum() == 0:
        return df

    # Primary columns
    a_col = 'int_fuel_consumed_acropole_kg'
    o_col = 'int_fuel_consumed_openap_kg'
    dur_col = 'interval_duration_sec'
    tow_col = 'tow_pred_kg'
    mass_col = 'int_mass_mean'
    frac_climb = 'int_frac_climb'
    frac_cruise = 'int_frac_cruise'
    frac_desc = 'int_frac_descent'

    # Ensure we have needed columns
    for c in [a_col, o_col, dur_col, tow_col, mass_col, frac_climb, frac_cruise, frac_desc]:
        if c not in df.columns:
            df[c] = np.nan

    safe_dur_hours = df[dur_col] / 3600.0
    safe_dur_hours.replace(0, np.nan, inplace=True)

    # Rates
    df.loc[changed_mask, 'int_acropole_fuel_rate'] = df.loc[changed_mask, a_col] / safe_dur_hours.loc[changed_mask]
    df.loc[changed_mask, 'int_openap_fuel_rate'] = df.loc[changed_mask, o_col] / safe_dur_hours.loc[changed_mask]

    # Normalized features
    df.loc[changed_mask, 'int_acropole_fuel_per_tow'] = df.loc[changed_mask, a_col] / df.loc[changed_mask, tow_col]
    df.loc[changed_mask, 'int_acropole_fuel_per_mass'] = df.loc[changed_mask, a_col] / df.loc[changed_mask, mass_col]

    # Phase rates
    df.loc[changed_mask, 'int_fuel_rate_climb_acropole_kgh'] = (df.loc[changed_mask, a_col] * df.loc[changed_mask, frac_climb]) / safe_dur_hours.loc[changed_mask]
    df.loc[changed_mask, 'int_fuel_rate_cruise_acropole_kgh'] = (df.loc[changed_mask, a_col] * df.loc[changed_mask, frac_cruise]) / safe_dur_hours.loc[changed_mask]
    df.loc[changed_mask, 'int_fuel_rate_descent_acropole_kgh'] = (df.loc[changed_mask, a_col] * df.loc[changed_mask, frac_desc]) / safe_dur_hours.loc[changed_mask]

    # Flight-level sums
    for col_name, base_col in [('flt_acropole_fuel_total_kg', a_col), ('flt_openap_fuel_total_kg', o_col)]:
        if base_col in df.columns:
            # remove existing column to avoid duplicate merges
            if col_name in df.columns:
                df = df.drop(columns=[col_name])
            totals = df.groupby('flight_id')[base_col].sum().reset_index().rename(columns={base_col: col_name})
            df = df.merge(totals, on='flight_id', how='left')

    # Cumulative per flight: int_cum_fuel_acropole_kg
    if a_col in df.columns:
        # compute within flight order
        if 'idx' in df.columns:
            df = df.sort_values(['flight_id', 'idx'])
        elif 'interval_start_ts' in df.columns:
            df['interval_start_ts'] = pd.to_datetime(df['interval_start_ts'])
            df = df.sort_values(['flight_id', 'interval_start_ts'])
        df['int_cum_fuel_acropole_kg'] = df.groupby('flight_id')[a_col].cumsum()

    # Comparisons and stats
    diff_col = 'int_fuel_diff_acropole_openap_kg'
    df.loc[changed_mask, diff_col] = df.loc[changed_mask, a_col] - df.loc[changed_mask, o_col]
    df.loc[changed_mask, 'int_fuel_sq_diff_acropole_openap'] = (df.loc[changed_mask, diff_col]) ** 2
    openap = df.loc[changed_mask, o_col]
    with np.errstate(divide='ignore', invalid='ignore'):
        df.loc[changed_mask, 'int_fuel_ratio_acropole_openap'] = df.loc[changed_mask, a_col] / openap
        df.loc[changed_mask, 'int_fuel_pct_error_acropole_openap'] = (df.loc[changed_mask, a_col] - df.loc[changed_mask, o_col]) / openap * 100
    df.loc[changed_mask, 'int_acropole_openap_agreement'] = np.exp(-np.abs(df.loc[changed_mask, diff_col]) / 100.0)

    # Flight-level comparisons
    if 'flt_acropole_fuel_total_kg' in df.columns and 'flt_openap_fuel_total_kg' in df.columns:
        # remove pre-existing comparison columns
        for comp in ['flt_fuel_diff_acropole_openap_kg', 'flt_fuel_ratio_acropole_openap', 'flt_fuel_pct_error_acropole_openap']:
            if comp in df.columns:
                df = df.drop(columns=[comp])
        totals = df.groupby('flight_id').agg({
            'flt_acropole_fuel_total_kg': 'first',
            'flt_openap_fuel_total_kg': 'first'
        }).reset_index()
        totals['flt_fuel_diff_acropole_openap_kg'] = totals['flt_acropole_fuel_total_kg'] - totals['flt_openap_fuel_total_kg']
        with np.errstate(divide='ignore', invalid='ignore'):
            totals['flt_fuel_ratio_acropole_openap'] = totals['flt_acropole_fuel_total_kg'] / totals['flt_openap_fuel_total_kg']
            totals['flt_fuel_pct_error_acropole_openap'] = (totals['flt_acropole_fuel_total_kg'] - totals['flt_openap_fuel_total_kg']) / totals['flt_openap_fuel_total_kg'] * 100
        df = df.merge(totals[['flight_id', 'flt_fuel_diff_acropole_openap_kg', 'flt_fuel_ratio_acropole_openap', 'flt_fuel_pct_error_acropole_openap']], on='flight_id', how='left')

    return df

if __name__ == '__main__':
    # Two-stage imputation flow
    fuel_base_targets = [
        'int_fuel_consumed_acropole_kg',
        'int_fuel_consumed_openap_kg'
    ]

    # If there's a checkpoint, load it and decide how to resume
    if checkpoint_path.exists():
        print(f"Loading checkpoint {checkpoint_path}")
        df = pd.read_parquet(checkpoint_path)
        # Check Stage A completion: are fuel base targets in train fully imputed?
        train_rows = df['dataset_split'] == 'train'
        stage_a_done = not df.loc[train_rows, fuel_base_targets].isnull().any().any()
    else:
        # Load and merge fresh data
        df_full = load_and_merge_data()
        # Separate splits
        train_df = df_full[df_full['dataset_split'] == 'train'].copy()
        rank_df = df_full[df_full['dataset_split'] == 'rank'].copy()
        final_df = df_full[df_full['dataset_split'] == 'final'].copy()

        # Merge fuel_kg into train only
        train_df = add_fuel_kg_to_train(train_df)

        # Clean each subset
        train_df = clean_data(train_df)
        rank_df = clean_data(rank_df)
        final_df = clean_data(final_df)

        # Stage A start
        stage_a_done = False

    # If we haven't computed stage_a yet, run Stage A on train only
    if not stage_a_done:
        # If we loaded checkpoint earlier, df variable exists; else we have train_df
        if 'train_df' not in locals():
            # Extract training subset
            if 'df' in locals():
                train_df = df[df['dataset_split'] == 'train'].copy()
            else:
                df_tmp = load_and_merge_data()
                train_df = df_tmp[df_tmp['dataset_split'] == 'train'].copy()
                train_df = add_fuel_kg_to_train(train_df)
                train_df = clean_data(train_df)

        print("Stage A: Train-only, fuel-aware imputation")

        # Snapshot before-stage A base fuel NaN pattern
        before_ac_na = train_df['int_fuel_consumed_acropole_kg'].isnull()
        before_op_na = train_df['int_fuel_consumed_openap_kg'].isnull()

        # A1: Impute base fuels on train using TRAIN_FUEL_AWARE_PREDICTORS
        train_mask = train_df['dataset_split'] == 'train'
        train_df = impute_targets_with_lightgbm(train_df, fuel_base_targets, TRAIN_FUEL_AWARE_PREDICTORS, train_mask)

        # Recompute derived fuel features for rows that changed from NaN to non-NaN
        changed_mask = (~train_df['int_fuel_consumed_acropole_kg'].isnull() & before_ac_na) | (~train_df['int_fuel_consumed_openap_kg'].isnull() & before_op_na)
        if 'flight_id' not in train_df.columns:
            train_df['flight_id'] = np.nan
        train_df = recompute_fuel_derived_features(train_df, changed_mask)

        # A3: Impute all remaining train NaNs using fuel-aware predictor set
        features_with_nans_train = list_nans(train_df)
        # remove identifiers or non-imputeable columns if present
        # don't impute 'flight_id', 'idx', 'dataset_split', 'interval_start_ts' etc.
        features_with_nans_train = [f for f in features_with_nans_train if f not in ('flight_id', 'idx', 'dataset_split', 'interval_start_ts', 'interval_end_ts')]

        # Key derived fuel features we want available to imputers
        key_fuel_derived = [
            'int_acropole_fuel_rate', 'int_openap_fuel_rate', 'int_cum_fuel_acropole_kg',
            'flt_acropole_fuel_total_kg', 'flt_openap_fuel_total_kg'
        ]
        predictors_stage_a_all = list(dict.fromkeys(TRAIN_FUEL_AWARE_PREDICTORS + fuel_base_targets + key_fuel_derived))
        train_df = impute_targets_with_lightgbm(train_df, features_with_nans_train, predictors_stage_a_all, train_mask)

        # Save stage A train output and checkpoint
        stage_a_path = base_path / "consolidated_features_train_imputed_lightgbm_stageA.parquet"
        train_df.to_parquet(stage_a_path, index=False)
        # Reconstruct df for checkpoint to include rank & final if present
        if 'df_full' in locals():
            df = pd.concat([train_df, rank_df, final_df], ignore_index=True)
        else:
            # If rank and final not loaded previously, read them
            rank_df = pd.read_parquet(base_path / "consolidated_features_rank_multiplier.parquet")
            rank_df['dataset_split'] = 'rank'
            final_df = pd.read_parquet(base_path / "consolidated_features_final_multiplier.parquet")
            final_df['dataset_split'] = 'final'
            rank_df = clean_data(rank_df)
            final_df = clean_data(final_df)
            df = pd.concat([train_df, rank_df, final_df], ignore_index=True)

        df.to_parquet(checkpoint_path, index=False)
        stage_a_done = True

    # Stage B: global imputation across train+rank+final **without** fuel_kg predictor
    print("Stage B: Global imputation across train, rank, final (no fuel_kg)")
    # reload rank and final fresh if not in memory
    if 'df' not in locals():
        df = load_and_merge_data()
        # Re-replace train subset with stage A result if available
        stage_a_path = base_path / "consolidated_features_train_imputed_lightgbm_stageA.parquet"
        if stage_a_path.exists():
            train_stage_a = pd.read_parquet(stage_a_path)
            train_stage_a['dataset_split'] = 'train'
            # Prepend other datasets
            rank_df = pd.read_parquet(base_path / "consolidated_features_rank_multiplier.parquet")
            rank_df['dataset_split'] = 'rank'
            final_df = pd.read_parquet(base_path / "consolidated_features_final_multiplier.parquet")
            final_df['dataset_split'] = 'final'
            rank_df = clean_data(rank_df)
            final_df = clean_data(final_df)
            df = pd.concat([train_stage_a, rank_df, final_df], ignore_index=True)
        else:
            # fallback
            df = load_and_merge_data()
            df = clean_data(df)
    else:
        # ensure stage A train subset is present in df (it should be)
        pass

    # Step B1: Impute base fuel across all splits using BASE_PREDICTORS (no fuel_kg)
    train_mask_all = pd.Series(True, index=df.index)
    # snapshot before imputation
    before_ac_all_na = df['int_fuel_consumed_acropole_kg'].isnull()
    before_op_all_na = df['int_fuel_consumed_openap_kg'].isnull()
    df = impute_targets_with_lightgbm(df, fuel_base_targets, BASE_PREDICTORS, train_mask_all)
    changed_mask_all = (~df['int_fuel_consumed_acropole_kg'].isnull() & before_ac_all_na) | (~df['int_fuel_consumed_openap_kg'].isnull() & before_op_all_na)
    df = recompute_fuel_derived_features(df, changed_mask_all)

    # Step B2: Impute remaining missing features across all splits
    features_with_nans_all = list_nans(df)
    # Exclude fuel_kg (train-only) from imputation targets
    features_with_nans_all = [f for f in features_with_nans_all if f != 'fuel_kg']

    key_fuel_derived = [
        'int_acropole_fuel_rate', 'int_openap_fuel_rate', 'int_cum_fuel_acropole_kg',
        'flt_acropole_fuel_total_kg', 'flt_openap_fuel_total_kg'
    ]
    predictors_stage_b_all = list(dict.fromkeys(BASE_PREDICTORS + fuel_base_targets + key_fuel_derived))
    df = impute_targets_with_lightgbm(df, features_with_nans_all, predictors_stage_b_all, train_mask_all)

    # Calculate per-feature stats for Stage B targets
    stats = []
    for col in features_with_nans_all:
        values = df[col].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            rmse_val = np.sqrt(np.mean((values - mean_val)**2))
            stats.append({'feature': col, 'mean': mean_val, 'rmse': rmse_val})
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(logs_path / "imputation_feature_stats.csv", index=False)

    # Final save (split back by dataset)
    train_imputed = df[df['dataset_split'] == 'train'].drop(columns=['dataset_split'])
    rank_imputed = df[df['dataset_split'] == 'rank'].drop(columns=['dataset_split'])
    final_imputed = df[df['dataset_split'] == 'final'].drop(columns=['dataset_split'])

    train_imputed.to_parquet(base_path / "consolidated_features_train_imputed_lightgbm.parquet", index=False)
    rank_imputed.to_parquet(base_path / "consolidated_features_rank_imputed_lightgbm.parquet", index=False)
    final_imputed.to_parquet(base_path / "consolidated_features_final_imputed_lightgbm.parquet", index=False)

    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("Two-stage LightGBM imputation complete. Saved train/rank/final datasets.")