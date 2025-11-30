"""
Script to train CatBoost model with proper CV and various target/scaling options.
Implements suggestions for improved RMSE:
- GroupKFold CV on flight_id
- Residual target vs physics baseline
- Feature pruning based on importance
- Mass+duration scaling
- Updated hyperparameters
- NaN assertions
- Optional monotone constraints
"""

import pandas as pd
import numpy as np
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from pathlib import Path
import json
import re
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

def get_model(algo, params, cat_cols=None, args=None):
    """Get model based on algorithm."""
    if algo == 'catboost':
        return cb.CatBoostRegressor(**params)
    elif algo == 'xgboost':
        xgb_params = {
            'max_depth': params.get('depth', 6),
            'learning_rate': params.get('learning_rate', 0.05),
            'n_estimators': params.get('iterations', 1000),
            'subsample': params.get('bagging_temperature', 1.0) / 2 + 0.5,  # approximate
            'colsample_bytree': 0.8,
            'reg_lambda': params.get('l2_leaf_reg', 5.0),
            'reg_alpha': 0.0,
            'enable_categorical': True,
            'tree_method': 'hist',
            'device': 'cuda',  # Use GPU
            'verbosity': 0,
            'booster': 'gblinear',  # Use linear booster for feature selection
            'feature_selector': args.xgboost_feature_selector if args else 'cyclic'
        }
        return xgb.XGBRegressor(**xgb_params)
    elif algo == 'lightgbm':
        lgb_params = {
            'max_depth': params.get('depth', 6),
            'learning_rate': params.get('learning_rate', 0.05),
            'n_estimators': params.get('iterations', 5000), #3000
            'subsample': params.get('bagging_temperature', 1.0) / 2 + 0.5,
            'colsample_bytree': 0.8,
            'reg_lambda': params.get('l2_leaf_reg', 5.0),
            'reg_alpha': 0.0,
            'categorical_feature': cat_cols if cat_cols else 'auto',
            'device': 'cpu',  # Use CPU
            'verbosity': -1
        }
        return lgb.LGBMRegressor(**lgb_params)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

def get_next_version(output_dir):
    """Find the next version number by checking existing files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(output_dir.glob("likable-ant_v*.parquet"))
    if not existing_files:
        return 1

    versions = []
    for f in existing_files:
        match = re.search(r'v(\d+)', f.stem)
        if match:
            versions.append(int(match.group(1)))

    return max(versions) + 1 if versions else 1

class TimeBasedGroupKFold:
    """Time-based K-fold cross-validation using flight timestamps."""
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    
    def split(self, X, y, groups, timestamps):
        """Generate time-based train/val splits.
        
        Args:
            X: Features
            y: Target
            groups: Flight IDs
            timestamps: Timestamps for each sample
        """
        # Get unique flights with their earliest timestamp
        unique_flights = pd.DataFrame({
            'flight_id': groups,
            'timestamp': timestamps
        })
        flight_times = unique_flights.groupby('flight_id')['timestamp'].min().sort_values()
        
        # Split flights into time-ordered folds
        n_flights = len(flight_times)
        fold_size = n_flights // self.n_splits
        
        flight_to_idx = {fid: i for i, fid in enumerate(groups)}
        
        for fold in range(self.n_splits):
            # Val fold: flights in this time period
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_flights
            val_flights = set(flight_times.iloc[val_start:val_end].index)
            
            # Train fold: all flights before this time period
            train_flights = set(flight_times.iloc[:val_start].index)
            
            # Skip if train is empty
            if len(train_flights) == 0:
                continue
            
            # Convert to indices
            train_idx = [i for i, fid in enumerate(groups) if fid in train_flights]
            val_idx = [i for i, fid in enumerate(groups) if fid in val_flights]
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield np.array(train_idx), np.array(val_idx)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def get_baseline_column(df):
    """Get the appropriate baseline fuel consumption column name."""
    if 'int_fuel_consumed_acropole_kg' in df.columns:
        return 'int_fuel_consumed_acropole_kg'
    elif 'int_fuel_consumed_kg' in df.columns:
        return 'int_fuel_consumed_kg'
    else:
        raise ValueError("No baseline fuel consumption column found (neither int_fuel_consumed_acropole_kg nor int_fuel_consumed_kg)")

def get_openap_column(df):
    """Get the appropriate OpenAP fuel consumption column name."""
    if 'int_fuel_consumed_openap_kg' in df.columns:
        return 'int_fuel_consumed_openap_kg'
    elif 'int_fuel_consumed_openap' in df.columns:
        return 'int_fuel_consumed_openap'
    else:
        return None  # OpenAP might not be available

def get_aircraft_column(df):
    """Get the appropriate aircraft type column name."""
    if 'aircraft_type_encoded' in df.columns:
        return 'aircraft_type_encoded'
    elif 'aircraft_type' in df.columns:
        return 'aircraft_type'
    else:
        return None

def create_physics_features(df):
    """Create physics-based interaction features."""
    print("   Creating physics-based interaction features...")

    # Mass above OEW (useful for many interactions)
    if 'ac_max_takeoff_weight_kg' in df.columns and 'ac_operating_empty_weight_kg' in df.columns:
        df['mass_above_oew'] = df['ac_max_takeoff_weight_kg'] - df['ac_operating_empty_weight_kg']
        df['mass_above_oew'] = df['mass_above_oew'].clip(lower=1.0)

    # Get baseline column (acropole or fallback)
    baseline_col = get_baseline_column(df)
    
    # Fuel per residual mass
    if baseline_col in df.columns and 'mass_above_oew' in df.columns:
        df['int_acropole_fuel_per_residual_mass'] = df[baseline_col] / (df['mass_above_oew'] + 1e-6)
        # Also add user-requested scaled and specific variants
        # scaled: fuel / mass_above_oew
        df['int_scaled_acropole_fuel_consumption'] = df[baseline_col] / (df['mass_above_oew'] + 1e-6)
        # specific: fuel / mass_above_oew * duration_hours
        if 'interval_duration_sec' in df.columns:
            duration_hr = df['interval_duration_sec'] / 3600.0
            df['int_specific_acropole_fuel_consumption'] = df[baseline_col] / (df['mass_above_oew'] + 1e-6) * duration_hr
    
    # Get OpenAP column if available
    openap_col = get_openap_column(df)
    if openap_col and 'mass_above_oew' in df.columns:
        df['int_openap_fuel_per_residual_mass'] = df[openap_col] / (df['mass_above_oew'] + 1e-6)
        # scaled: fuel / mass_above_oew
        df['int_scaled_openap_fuel_consumption'] = df[openap_col] / (df['mass_above_oew'] + 1e-6)
        # specific: fuel / mass_above_oew * duration_hours
        if 'interval_duration_sec' in df.columns:
            duration_hr = df['interval_duration_sec'] / 3600.0
            df['int_specific_openap_fuel_consumption'] = df[openap_col] / (df['mass_above_oew'] + 1e-6) * duration_hr

    # 1. Efficiency-weighted distance/time
    if 'int_ground_distance_nm' in df.columns and 'mass_above_oew' in df.columns:
        df['distance_per_mass'] = df['int_ground_distance_nm'] / df['mass_above_oew']

    if 'interval_duration_sec' in df.columns and 'mass_above_oew' in df.columns:
        df['time_per_mass'] = df['interval_duration_sec'] / df['mass_above_oew']

    # 2. Physics model error × operating conditions
    if 'flt_fuel_pct_error_acropole_openap' in df.columns and 'mass_above_oew' in df.columns:
        df['acropole_openap_error_x_mass'] = df['flt_fuel_pct_error_acropole_openap'] * df['mass_above_oew']

    if 'flt_fuel_pct_error_acropole_openap' in df.columns and 'flt_ground_distance_nm' in df.columns:
        df['acropole_openap_error_x_distance'] = df['flt_fuel_pct_error_acropole_openap'] * df['flt_ground_distance_nm']

    # 3. Aircraft "shape" × cruise state (if available)
    if 'ac_mach_altitude_product' in df.columns and 'ac_efficiency_index' in df.columns:
        df['mach_altitude_eff'] = df['ac_mach_altitude_product'] * df['ac_efficiency_index']

    if 'cruise_median_tas' in df.columns and 'ac_aspect_ratio' in df.columns:
        df['cruise_tas_x_aspect_ratio'] = df['cruise_median_tas'] * df['ac_aspect_ratio']

    # 4. Phase imbalance features
    if 'level_distance_nm' in df.columns and 'flt_ground_distance_nm' in df.columns:
        df['level_distance_fraction'] = df['level_distance_nm'] / (df['flt_ground_distance_nm'] + 1e-6)

    if 'climb_time_sec' in df.columns and 'flt_duration_sec' in df.columns:
        df['climb_time_fraction'] = df['climb_time_sec'] / (df['flt_duration_sec'] + 1e-6)

    if 'descent_time_sec' in df.columns and 'flt_duration_sec' in df.columns:
        df['descent_time_fraction'] = df['descent_time_sec'] / (df['flt_duration_sec'] + 1e-6)

    # 5. Fuel efficiency indicators
    if 'int_fuel_per_tow' in df.columns and 'int_ground_distance_nm' in df.columns:
        df['fuel_efficiency_per_nm'] = df['int_fuel_per_tow'] / (df['int_ground_distance_nm'] + 1e-6)

    return df

def get_monotone_constraints(feature_names):
    """
    Define monotone constraints for physically meaningful features.
    Returns a list of constraints: +1 (monotone increasing), -1 (decreasing), 0 (no constraint)
    """
    constraints = []

    #Define features that should monotonically increase fuel consumption
    monotone_increasing = [
        'int_ground_distance_nm',
        'interval_duration_sec',
        'ac_max_takeoff_weight_kg',
        'mass_above_oew',
    ]

    for feature in feature_names:
        if any(mono_feat in feature for mono_feat in monotone_increasing):
            constraints.append(1)
        else:
            constraints.append(0)

    return constraints

def prepare_features(df, drop_cols):
    """Prepare features by dropping specified columns and handling categoricals."""
    # Drop specified columns
    X = df.drop(columns=drop_cols, errors='ignore')

    # Identify categorical columns (object and category types)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Convert all categorical columns to string then category for consistency
    for col in cat_cols:
        X[col] = X[col].astype(str).astype('category')

    # Assert no NaNs - push imputation to Stage 8
    nan_counts = X.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"   ERROR: Found {total_nans} NaN values across {(nan_counts > 0).sum()} columns")
        print("   Top NaN columns:")
        print(nan_counts[nan_counts > 0].sort_values(ascending=False).head(10))
        raise AssertionError("Stage 9: unexpected NaNs – fix in Stage 8 imputation!")

    return X, cat_cols

def main():
    parser = argparse.ArgumentParser(description='Stage 9 CV - Train CatBoost fuel model with proper CV and target options')
    parser.add_argument('--target-type', choices=['scaled', 'residual', 'specific'], default='scaled',
                       help='Target type: scaled (fuel/mass), residual (fuel - acropole), specific (fuel/(mass*duration))')
    parser.add_argument('--monotone', action='store_true', default=False, dest='use_monotone', help='Enable monotone constraints (disabled by default)')
    parser.add_argument('--feature-pruning', type=int, default=0,
                       help='Prune to top N features by importance (0 = no pruning)')
    parser.add_argument('--cv-folds', type=int, default=9, help='Number of CV folds')
    parser.add_argument('--cv-type', choices=['time', 'group'], default='group',
                       help='CV type: time (time-based) or group (random GroupKFold)')
    parser.add_argument('--dataset-type', choices=['lightgbm', 'original', 'lightgbm_with_tow'], default='lightgbm',
                       help='Type of imputation: lightgbm or original or lightgbm_with_tow')
    parser.add_argument('--dataset', choices=['dataset_1', 'dataset_2', 'dataset_3'], default='dataset_1',
                       help='Dataset to use: dataset_1 (current), dataset_2 (old_consolidated), dataset_3 (old_imputed)')
    parser.add_argument('--algo', choices=['catboost', 'xgboost', 'lightgbm', 'ensemble'], default='ensemble',
                       help='Algorithm to use: catboost, xgboost, lightgbm, or ensemble')
    parser.add_argument('--xgboost-feature-selector', choices=['cyclic', 'shuffle', 'random', 'greedy', 'thrifty'], default='cyclic',
                       help='XGBoost feature selector for linear booster')
    parser.add_argument('--train-features', help='Path to train features (auto-set based on dataset-type)')
    parser.add_argument('--rank-features', help='Path to rank features (auto-set based on dataset-type)')
    parser.add_argument('--final-features', help='Path to final features (auto-set based on dataset-type)')
    parser.add_argument('--raw-fuel', default=r'F:/PRC_2025/Likable-ant_v1/data/raw/fuel_train.parquet',
                       help='Path to raw fuel train data')
    parser.add_argument('--submission-template', default=r'F:/PRC_2025/Likable-ant_v1/data/raw/fuel_rank_submission.parquet',
                       help='Submission template')
    parser.add_argument('--skip-cv', action='store_true', help='Skip cross-validation and train final model directly')
    parser.add_argument('--tail-blending', action='store_true', default=False, help='Enable tail blending with physics baseline')
    parser.add_argument('--tail-threshold', type=float, default=800.0, help='Fuel threshold (kg) for tail blending')
    parser.add_argument('--bias-correction', action='store_true', default=False, help='Enable per-fuel-bin bias correction (disabled by default)')
    args = parser.parse_args()

    # Set paths based on dataset
    if args.dataset == 'dataset_1':
        base_path = r'F:/PRC_2025/Likable-ant_v1/data/processed/Stage_8_Consolidated_Features'
        train_features_path = Path(f"{base_path}/Train_281125.parquet")
        rank_features_path = Path(f"{base_path}/Rank_281125.parquet")
        final_features_path = Path(f"{base_path}/Final_281125.parquet")
    elif args.dataset == 'dataset_2':
        train_features_path = Path(r"F:\PRC_2025\Likable-ant_v1\data\processed\OLD\consolidated_features\consolidated_features_train.parquet")
        rank_features_path = Path(r"F:\PRC_2025\Likable-ant_v1\data\processed\OLD\consolidated_features\consolidated_features_rank.parquet")
        final_features_path = Path(r"F:\PRC_2025\Likable-ant_v1\data\processed\OLD\consolidated_features\consolidated_features_final.parquet")
    elif args.dataset == 'dataset_3':
        train_features_path = Path(r"F:\PRC_2025\Likable-ant_v1\data\processed\OLD\consolidated_features_imputed\consolidated_features_train_multiplier_imputed.parquet")
        rank_features_path = Path(r"F:\PRC_2025\Likable-ant_v1\data\processed\OLD\consolidated_features_imputed\consolidated_features_rank_multiplier_imputed.parquet")
        final_features_path = Path(r"F:\PRC_2025\Likable-ant_v1\data\processed\OLD\consolidated_features_imputed\consolidated_features_final_multiplier_imputed.parquet")

    # Override if explicitly provided
    if args.train_features:
        train_features_path = Path(args.train_features)
    if args.rank_features:
        rank_features_path = Path(args.rank_features)
    if args.final_features:
        final_features_path = Path(args.final_features)

    raw_fuel_path = Path(args.raw_fuel)
    submission_template_path = Path(args.submission_template)

    # Output paths
    submissions_dir = Path("F:/PRC_2025/submissions")
    submissions_dir.mkdir(parents=True, exist_ok=True)

    version = get_next_version(submissions_dir)

    output_path = submissions_dir / f"likable-ant_v{version}.parquet"
    feature_importance_path = submissions_dir / f"likable-ant_v{version}_feature_importance.csv"
    metadata_path = submissions_dir / f"likable-ant_v{version}_metadata.json"

    print("=" * 80)
    print(f"CATBOOST CV TRAINING - {args.target_type.upper()} TARGET ({args.dataset.upper()} DATASET) ({args.algo.upper()} ALGO) (v{version})")
    print("=" * 80)

    # 1. Load Data
    print("\n1. Loading datasets...")
    train_df = pd.read_parquet(train_features_path)
    rank_df = pd.read_parquet(rank_features_path)
    final_df = None
    if final_features_path.exists():
        final_df = pd.read_parquet(final_features_path)
    fuel_df = pd.read_parquet(raw_fuel_path)

    print(f"   Train features shape: {train_df.shape}")
    print(f"   Rank features shape: {rank_df.shape}")
    if final_df is not None:
        print(f"   Final features shape: {final_df.shape}")

    # Load flight dates for time-based CV
    if args.cv_type == 'time':
        print("   Loading flight dates for time-based CV...")
        try:
            flightlist_train = pd.read_parquet('data/raw/flightlist_train.parquet')[['flight_id', 'flight_date']]
            flightlist_rank = pd.read_parquet('data/raw/flightlist_rank.parquet')[['flight_id', 'flight_date']]
            
            # Merge flight_date into train_df and rank_df
            train_df = train_df.merge(flightlist_train, on='flight_id', how='left')
            rank_df = rank_df.merge(flightlist_rank, on='flight_id', how='left')
            
            if final_df is not None:
                flightlist_final = pd.read_parquet('data/raw/flightlist_final.parquet')[['flight_id', 'flight_date']]
                final_df = final_df.merge(flightlist_final, on='flight_id', how='left')
            
            print(f"   ✓ Added flight_date column for time-based CV")
        except Exception as e:
            print(f"   WARNING: Could not load flight dates: {e}")
            print(f"   Time-based CV will not be available")

    # 2. Create physics-based features
    print("\n2. Engineering physics-based features...")
    train_df = create_physics_features(train_df)
    rank_df = create_physics_features(rank_df)
    if final_df is not None:
        final_df = create_physics_features(final_df)

    # Filter training data to only include aircraft types in rank/final
    print("\n2.1. Filtering aircraft types...")
    rank_aircraft_col = get_aircraft_column(rank_df)
    final_aircraft_col = get_aircraft_column(final_df) if final_df is not None else None
    
    rank_aircraft_types = set(rank_df[rank_aircraft_col].unique()) if rank_aircraft_col else set()
    final_aircraft_types = set(final_df[final_aircraft_col].unique()) if final_df is not None and final_aircraft_col else set()
    valid_aircraft_types = rank_aircraft_types.union(final_aircraft_types)
    
    print(f"   Rank aircraft types: {len(rank_aircraft_types)}")
    print(f"   Final aircraft types: {len(final_aircraft_types)}")
    print(f"   Combined valid aircraft types: {len(valid_aircraft_types)}")
    
    train_aircraft_col = get_aircraft_column(train_df)
    if train_aircraft_col and len(valid_aircraft_types) > 0:
        initial_train_len = len(train_df)
        train_aircraft_types = set(train_df[train_aircraft_col].unique())
        excluded_types = train_aircraft_types - valid_aircraft_types
        
        if len(excluded_types) > 0:
            print(f"   Excluding {len(excluded_types)} aircraft types from training: {sorted(excluded_types)}")
            train_df = train_df[train_df[train_aircraft_col].isin(valid_aircraft_types)].copy()
            print(f"   Filtered training data: {initial_train_len} -> {len(train_df)} rows ({initial_train_len - len(train_df)} removed)")
        else:
            print(f"   All training aircraft types are present in rank/final data")
    else:
        print(f"   Skipping aircraft type filtering (column not found or no valid types)")

    # Sample mode for debugging (only sample training data, keep full rank for prediction)
    sample_size = getattr(args, 'sample', 0)
    if sample_size > 0:
        sample_n = int(sample_size)
        train_df = train_df.sample(n=min(sample_n, len(train_df)), random_state=42)
        fuel_df = fuel_df[fuel_df['idx'].isin(train_df['idx'])]
        print(f"   Running in sample mode: {sample_n} rows from train data (rank data kept full for prediction)")

    # 3. Merge Target & Clean
    print("\n3. Preparing training data...")
    # Avoid collision if train_df has a fuel_kg column (imputed / feature). Keep it as fuel_kg_train
    if 'fuel_kg' in train_df.columns:
        print("   Note: 'fuel_kg' found in train features; renaming to 'fuel_kg_train' to avoid collision")
        train_df = train_df.rename(columns={'fuel_kg': 'fuel_kg_train'})

    merged_df = train_df.merge(fuel_df[['idx', 'fuel_kg']], on='idx', how='left')

    # If by any chance the merge creates fuel_kg_x / fuel_kg_y columns (older pandas behavior),
    # normalize to single 'fuel_kg' target using the right-hand (raw fuel) column.
    if 'fuel_kg' not in merged_df.columns:
        # prefer *_y coming from the fuel_df join; if not present, try *_x
        if 'fuel_kg_y' in merged_df.columns:
            merged_df['fuel_kg'] = merged_df['fuel_kg_y']
            merged_df = merged_df.drop(columns=['fuel_kg_x', 'fuel_kg_y'], errors='ignore')
        elif 'fuel_kg_x' in merged_df.columns:
            merged_df['fuel_kg'] = merged_df['fuel_kg_x']
            merged_df = merged_df.drop(columns=['fuel_kg_x'], errors='ignore')


    if 'has_trajectory_data' in merged_df.columns:
        initial_len = len(merged_df)
        merged_df = merged_df[merged_df['has_trajectory_data'] == True]
        print(f"   Kept {len(merged_df)} rows where has_trajectory_data is True (removed {initial_len - len(merged_df)} others)")
    else:
         print("   Warning: has_trajectory_data column not found, proceeding without filtering")

    #Remove outliers based on interval relative positions (-0.5 to 1.5 range)
    if 'interval_rel_position_start' in merged_df.columns and 'interval_rel_position_end' in merged_df.columns:
        before_outlier_removal = len(merged_df)
        outlier_mask = (
            (merged_df['interval_rel_position_start'] >= -1) &
            (merged_df['interval_rel_position_start'] <= 1.5) &
            (merged_df['interval_rel_position_end'] >= -0.5) &
            (merged_df['interval_rel_position_end'] <= 1.5)
        )
        merged_df = merged_df[outlier_mask]
        outliers_removed = before_outlier_removal - len(merged_df)
        print(f"   Removed {outliers_removed} rows with interval_rel_position outliers (outside -0.5 to 1.5 range)")


    # Calculate mass_above_oew if not present
    if 'mass_above_oew' not in merged_df.columns:
        mass_above_oew = merged_df['ac_max_takeoff_weight_kg'] - merged_df['ac_operating_empty_weight_kg']
        mass_above_oew = mass_above_oew.clip(lower=1.0)
        merged_df['mass_above_oew'] = mass_above_oew
    else:
        mass_above_oew = merged_df['mass_above_oew']

    # Create target based on type
    if args.target_type == 'residual':
        # Residual: actual fuel - acropole baseline
        baseline_col = get_baseline_column(merged_df)
        baseline = merged_df[baseline_col]
        merged_df['fuel_residual_kg'] = merged_df['fuel_kg'] - baseline
        y = merged_df['fuel_residual_kg']
        y_actual = merged_df['fuel_kg']
        print(f"   Residual target created: fuel_kg - {baseline_col}")
        print(f"   Target stats - Mean: {y.mean():.2f}, Std: {y.std():.2f}")

    elif args.target_type == 'specific':
        # Specific fuel: fuel / (mass * duration)
        duration_hr = merged_df['interval_duration_sec'] / 3600.0
        merged_df['fuel_specific'] = merged_df['fuel_kg'] / (mass_above_oew * duration_hr)
        y = merged_df['fuel_specific']
        y_actual = merged_df['fuel_kg']
        print(f"   Specific fuel target: fuel_kg / (mass_above_oew * duration_hr)")
        print(f"   Target stats - Mean: {y.mean():.6f}, Std: {y.std():.6f}")

    else:  # scaled
        # Scaled: fuel / mass
        merged_df['fuel_kg_scaled'] = merged_df['fuel_kg'] / mass_above_oew
        y = merged_df['fuel_kg_scaled']
        y_actual = merged_df['fuel_kg']
        print(f"   Scaled target: fuel_kg / mass_above_oew")
        print(f"   Target stats - Mean: {y.mean():.6f}, Std: {y.std():.6f}")

    # 4. Prepare Features
    print("\n4. Preparing features...")
    drop_cols = ['fuel_kg', 'fuel_kg_actual', 'fuel_kg_scaled', 'fuel_residual_kg', 'fuel_specific', 'idx', #'flight_id',
                 'fuel_kg_train', 'fuel_kg_x', 'fuel_kg_y',
                 # Trial: dropping specific features from training
                 'int_acropole_fuel_per_residual_mass',
                 'int_scaled_acropole_fuel_consumption',
                 'int_fuel_per_tow',
                 'int_fuel_per_mass',
                 'int_openap_fuel_per_residual_mass',
                 'int_scaled_openap_fuel_consumption',

                ]

    # Drop datetime columns
    datetime_cols = merged_df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    datetime_cols += [c for c in merged_df.columns if c.endswith('_ts')]
    datetime_cols = list(set(datetime_cols))
    
    # Keep timestamp columns for time-based CV
    if args.cv_type == 'time':
        timestamp_cols_to_keep = ['timestamp', 'flight_departure_ts', 'flight_date']
        datetime_cols = [col for col in datetime_cols if col not in timestamp_cols_to_keep]
        print(f"   Dropping {len(datetime_cols)} datetime columns (keeping timestamp columns for time-based CV)")
    else:
        print(f"   Dropping {len(datetime_cols)} datetime columns")
    
    drop_cols += datetime_cols

    X, cat_cols = prepare_features(merged_df, drop_cols)
    groups = merged_df['flight_id'].values

    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Categorical features: {len(cat_cols)}")
    print(f"   Numeric features: {X.shape[1] - len(cat_cols)}")

    # Feature pruning if requested
    if args.feature_pruning > 0:
        print(f"\n   Feature pruning: keeping top {args.feature_pruning} features...")
        # For first run, we need to train briefly to get importance
        temp_params = {
            'depth': 6, 'learning_rate': 0.1, 'iterations': 100,
            'verbose': False, 'thread_count': -1
        }
        temp_model = cb.CatBoostRegressor(**temp_params)
        temp_model.fit(X, y, cat_features=cat_cols)

        temp_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)

        top_features = temp_importance.head(args.feature_pruning)['feature'].tolist()
        X = X[top_features].copy()
        print(f"   Pruned to {len(top_features)} features")

        # Filter categorical columns to only include those still in the feature set
        cat_cols = [col for col in cat_cols if col in X.columns]
        print(f"   Remaining categorical features: {len(cat_cols)}")

    # Get monotone constraints if requested
    monotone_constraints = None
    if args.use_monotone:
        monotone_constraints = get_monotone_constraints(X.columns)
        n_monotone = sum(1 for c in monotone_constraints if c != 0)
        print(f"   Monotone constraints applied to {n_monotone} features")
    else:
        n_monotone = 0

    # 5. CV Training with Ensemble
    # Stronger regularization for better generalization
    best_params = {
        'depth': 5,  # Reduced from 6
        'learning_rate': 0.03,  # Reduced from 0.05
        'iterations': 10000,  # Increased from 5000
        'l2_leaf_reg': 10.0,  # Increased from 5.0
        'border_count': 254,
        'bagging_temperature': 1.0,
        'random_strength': 1.5,
        'min_data_in_leaf': 60,  # Increased from 30
        'loss_function': 'RMSE',
        'task_type': 'CPU' if monotone_constraints is not None else 'GPU',  # Use CPU for monotone constraints
        'od_type': 'Iter',
        'od_wait': 300,
        'thread_count': -1,
    }

    if args.skip_cv:
        print(f"\n5. Skipping CV as requested...")
        cv_rmse = {'catboost': [], 'lightgbm': [], 'ensemble': []}
        cv_mean = 0.0
        cv_std = 0.0
        fold_models = {'catboost': [], 'lightgbm': []}
        fold_importances = [np.zeros(X.shape[1])]
        validation_results = []
        bias_corrections = None
    else:
        cv_type_name = 'Time-based' if args.cv_type == 'time' else 'Random GroupKFold'
        print(f"\n5. Performing {args.cv_folds}-fold {cv_type_name} CV with Ensemble (CatBoost + LightGBM)...")

        # Choose CV strategy
        if args.cv_type == 'time':
            # Check for timestamp column
            if 'timestamp' not in merged_df.columns and 'flight_departure_ts' not in merged_df.columns and 'flight_date' not in merged_df.columns:
                print("   WARNING: No timestamp column found for time-based CV, falling back to GroupKFold")
                gkf = GroupKFold(n_splits=args.cv_folds)
            else:
                timestamp_col = 'timestamp' if 'timestamp' in merged_df.columns else ('flight_departure_ts' if 'flight_departure_ts' in merged_df.columns else 'flight_date')
                timestamps = pd.to_datetime(merged_df[timestamp_col])
                print(f"   Using time-based CV with column: {timestamp_col}")
                print(f"   Date range: {timestamps.min()} to {timestamps.max()}")
                gkf = TimeBasedGroupKFold(n_splits=args.cv_folds)
        else:
            gkf = GroupKFold(n_splits=args.cv_folds)
        
        cv_rmse = {'catboost': [], 'lightgbm': [], 'ensemble': []}
        fold_models = {'catboost': [], 'lightgbm': []}
        fold_importances = []
        validation_results = []

        # Pass timestamps if using time-based CV
        if args.cv_type == 'time' and isinstance(gkf, TimeBasedGroupKFold):
            cv_splits = list(gkf.split(X, y, groups, timestamps))
        else:
            cv_splits = list(gkf.split(X, y, groups))
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n   Fold {fold+1}/{len(cv_splits)}:")
            if args.cv_type == 'time' and ('timestamp' in merged_df.columns or 'flight_departure_ts' in merged_df.columns or 'flight_date' in merged_df.columns):
                timestamp_col = 'timestamp' if 'timestamp' in merged_df.columns else ('flight_departure_ts' if 'flight_departure_ts' in merged_df.columns else 'flight_date')
                train_dates = pd.to_datetime(merged_df.iloc[train_idx][timestamp_col])
                val_dates = pd.to_datetime(merged_df.iloc[val_idx][timestamp_col])
                print(f"     Train: {train_dates.min()} to {train_dates.max()}")
                print(f"     Val:   {val_dates.min()} to {val_dates.max()}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            mass_val = mass_above_oew.iloc[val_idx]

            # Train CatBoost
            print(f"     Training CatBoost...")
            cat_model = cb.CatBoostRegressor(**best_params, verbose=False)
            if monotone_constraints is not None:
                cat_model.set_params(monotone_constraints=monotone_constraints)
            cat_model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_val, y_val), verbose=False)
            fold_models['catboost'].append(cat_model)

            # Train LightGBM
            print(f"     Training LightGBM...")
            lgb_model = get_model('lightgbm', best_params, cat_cols=cat_cols, args=args)
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
            fold_models['lightgbm'].append(lgb_model)

            # Store feature importances from CatBoost
            if hasattr(cat_model, 'feature_importances_'):
                fold_importances.append(cat_model.feature_importances_)
            else:
                fold_importances.append(np.zeros(X.shape[1]))

            # Make predictions with each model
            cat_pred = cat_model.predict(X_val)
            lgb_pred = lgb_model.predict(X_val)
            ensemble_pred = (cat_pred + lgb_pred) / 2

            # Convert predictions to kg scale for each model
            def convert_to_kg(pred, val_idx, mass_val):
                if args.target_type == 'residual':
                    baseline_col = get_baseline_column(merged_df)
                    baseline_val = merged_df.iloc[val_idx][baseline_col]
                    return baseline_val + pred
                elif args.target_type == 'specific':
                    duration_hr_val = merged_df.iloc[val_idx]['interval_duration_sec'] / 3600.0
                    return pred * mass_val * duration_hr_val
                else:  # scaled
                    return pred * mass_val

            cat_pred_kg = convert_to_kg(cat_pred, val_idx, mass_val)
            lgb_pred_kg = convert_to_kg(lgb_pred, val_idx, mass_val)
            ensemble_pred_kg = convert_to_kg(ensemble_pred, val_idx, mass_val)

            # Calculate RMSE for each model
            y_val_kg = y_actual.iloc[val_idx]
            cat_rmse = np.sqrt(mean_squared_error(y_val_kg, cat_pred_kg))
            lgb_rmse = np.sqrt(mean_squared_error(y_val_kg, lgb_pred_kg))
            ensemble_rmse = np.sqrt(mean_squared_error(y_val_kg, ensemble_pred_kg))

            cv_rmse['catboost'].append(cat_rmse)
            cv_rmse['lightgbm'].append(lgb_rmse)
            cv_rmse['ensemble'].append(ensemble_rmse)

            print(f"     CatBoost RMSE_kg = {cat_rmse:.2f}")
            print(f"     LightGBM RMSE_kg = {lgb_rmse:.2f}")
            print(f"     Ensemble RMSE_kg = {ensemble_rmse:.2f}")

            # Store validation results for error analysis
            val_results = pd.DataFrame({
                'fold': fold + 1,
                'idx': merged_df.iloc[val_idx]['idx'].values,
                'flight_id': merged_df.iloc[val_idx]['flight_id'].values,
                'aircraft_type': merged_df.iloc[val_idx][get_aircraft_column(merged_df)].values if get_aircraft_column(merged_df) else 'unknown',
                'actual_fuel_kg': y_val_kg.values,
                'catboost_pred_kg': cat_pred_kg,
                'lightgbm_pred_kg': lgb_pred_kg,
                'ensemble_pred_kg': ensemble_pred_kg,
                'catboost_error': cat_pred_kg - y_val_kg.values,
                'lightgbm_error': lgb_pred_kg - y_val_kg.values,
                'ensemble_error': ensemble_pred_kg - y_val_kg.values,
                'mass_above_oew': mass_val.values,
                'duration_sec': merged_df.iloc[val_idx]['interval_duration_sec'].values if 'interval_duration_sec' in merged_df.columns else np.nan
            })
            validation_results.append(val_results)

        # Combine validation results
        validation_df = pd.concat(validation_results, ignore_index=True)

        # Compute bias corrections per fuel bin if enabled
        bias_corrections = None
        if args.bias_correction:
            print(f"\n   Computing per-fuel-bin bias corrections...")
            validation_df['fuel_bin'] = pd.qcut(validation_df['actual_fuel_kg'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            bias_corrections = validation_df.groupby('fuel_bin', observed=True)['ensemble_error'].mean().to_dict()
            print(f"   Bias corrections: {bias_corrections}")

        cv_mean = np.mean(cv_rmse['ensemble'])
        cv_std = np.std(cv_rmse['ensemble'])
        print(f"\n   CV Results:")
        print(f"     CatBoost: Mean RMSE = {np.mean(cv_rmse['catboost']):.2f}, Std = {np.std(cv_rmse['catboost']):.2f}")
        print(f"     LightGBM: Mean RMSE = {np.mean(cv_rmse['lightgbm']):.2f}, Std = {np.std(cv_rmse['lightgbm']):.2f}")
        print(f"     Ensemble: Mean RMSE = {cv_mean:.2f}, Std = {cv_std:.2f}")

    # 6. Train Final Model
    print("\n6. Training final model on full dataset...")

    # Create a small validation holdout using GroupShuffleSplit so we can show
    # validation RMSE during training and access best scores.
    use_val = False
    if len(set(groups)) > 1 and len(X) > 10:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx_full, val_idx_full = next(gss.split(X, y, groups))
        X_train_full, X_val_full = X.iloc[train_idx_full], X.iloc[val_idx_full]
        y_train_full, y_val_full = y.iloc[train_idx_full], y.iloc[val_idx_full]
        use_val = True
        print(f"   Final training using {len(train_idx_full)} rows, validation {len(val_idx_full)} rows (group-aware split)")
    else:
        X_train_full, y_train_full = X, y
        X_val_full, y_val_full = None, None
        print("   Final training without an explicit validation holdout (all data used for training)")

    if args.algo == 'ensemble':
        print("   Training ensemble of CatBoost and LightGBM...")
        cat_model = cb.CatBoostRegressor(**best_params, verbose=True)
        if monotone_constraints is not None:
            cat_model.set_params(monotone_constraints=monotone_constraints)
        if use_val:
            cat_model.fit(X_train_full, y_train_full, cat_features=cat_cols, eval_set=(X_val_full, y_val_full), verbose=True)
        else:
            cat_model.fit(X_train_full, y_train_full, cat_features=cat_cols, verbose=True)

        lgb_model = get_model('lightgbm', best_params, cat_cols=cat_cols, args=args)
        if use_val:
            lgb_model.fit(X_train_full, y_train_full, eval_set=[(X_val_full, y_val_full)], callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(1)])
        else:
            lgb_model.fit(X_train_full, y_train_full, verbose=False)

        final_model = {'catboost': cat_model, 'lightgbm': lgb_model}
    else:
        final_model = get_model(args.algo, best_params, cat_cols, args)
        if args.algo == 'catboost' and monotone_constraints is not None:
            final_model.set_params(monotone_constraints=monotone_constraints)
        # Drop categorical columns for XGBoost
        if args.algo == 'xgboost':
            if use_val:
                X_train_full = X_train_full.drop(columns=cat_cols, errors='ignore')
                X_val_full = X_val_full.drop(columns=cat_cols, errors='ignore')
            else:
                X_train_full = X_train_full.drop(columns=cat_cols, errors='ignore')
        if use_val:
            if args.algo == 'xgboost':
                final_model.fit(X_train_full, y_train_full, eval_set=[(X_val_full, y_val_full)], verbose=True)
            elif args.algo == 'lightgbm':
                final_model.fit(X_train_full, y_train_full, eval_set=[(X_val_full, y_val_full)], callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(1)])
            else:  # catboost
                final_model.fit(X_train_full, y_train_full, cat_features=cat_cols, eval_set=(X_val_full, y_val_full), verbose=True)
        else:
            if args.algo == 'catboost':
                final_model.fit(X_train_full, y_train_full, cat_features=cat_cols, verbose=True)
            else:
                final_model.fit(X_train_full, y_train_full)

    # After training, extract best metrics if available
    try:
        best_score = final_model.get_best_score()
        learn_rmse = None
        val_rmse = None
        if isinstance(best_score, dict):
            if 'learn' in best_score and 'RMSE' in best_score['learn']:
                learn_rmse = best_score['learn']['RMSE']
            # find first validation-like key
            for k in best_score:
                if k != 'learn':
                    val_rmse = best_score[k].get('RMSE', None)
                    break

        print("   Final model best scores:")
        if learn_rmse is not None:
            print(f"     best learn RMSE: {learn_rmse:.6f}")
        if val_rmse is not None:
            print(f"     best validation RMSE: {val_rmse:.6f}")
    except Exception:
        # Older/other CatBoost versions may not provide get_best_score; ignore gracefully
        pass

    # After training on the full dataset, compute in-sample predictions and RMSE (un-scaled to kg)
    print("\n   Computing final model in-sample RMSE on the full training set...")
    if isinstance(final_model, dict):
        # Ensemble
        cat_pred = final_model['catboost'].predict(X)
        lgb_pred = final_model['lightgbm'].predict(X)
        train_pred = (cat_pred + lgb_pred) / 2
    else:
        if args.algo == 'xgboost':
            X_for_pred = X.drop(columns=cat_cols, errors='ignore')
        else:
            X_for_pred = X
        train_pred = final_model.predict(X_for_pred)
    if args.target_type == 'residual':
        baseline_col = get_baseline_column(merged_df)
        baseline_train = merged_df[baseline_col]
        train_pred_kg = baseline_train + train_pred
    elif args.target_type == 'specific':
        duration_hr_train = merged_df['interval_duration_sec'] / 3600.0
        train_pred_kg = train_pred * mass_above_oew * duration_hr_train
    else:  # scaled
        train_pred_kg = train_pred * mass_above_oew

    # After training, collect validation results from final training holdout (if available)
    # This enables error analysis even when skipping CV
    if use_val and X_val_full is not None and y_val_full is not None:
        print(f"\n   Collecting validation results from final training holdout...")
        
        # Make predictions on validation holdout
        if isinstance(final_model, dict):
            # Ensemble
            cat_val_pred = final_model['catboost'].predict(X_val_full)
            lgb_val_pred = final_model['lightgbm'].predict(X_val_full)
            ensemble_val_pred = (cat_val_pred + lgb_val_pred) / 2
        else:
            if args.algo == 'xgboost':
                X_val_for_pred = X_val_full.drop(columns=cat_cols, errors='ignore')
            else:
                X_val_for_pred = X_val_full
            ensemble_val_pred = final_model.predict(X_val_for_pred)
            cat_val_pred = ensemble_val_pred  # For single models, use same prediction
            lgb_val_pred = ensemble_val_pred

        # Convert predictions to kg scale
        mass_val_full = mass_above_oew.iloc[val_idx_full]
        
        def convert_to_kg_val(pred, mass_val):
            if args.target_type == 'residual':
                baseline_col = get_baseline_column(merged_df)
                baseline_val = merged_df.iloc[val_idx_full][baseline_col]
                return baseline_val + pred
            elif args.target_type == 'specific':
                duration_hr_val = merged_df.iloc[val_idx_full]['interval_duration_sec'] / 3600.0
                return pred * mass_val * duration_hr_val
            else:  # scaled
                return pred * mass_val

        cat_val_pred_kg = convert_to_kg_val(cat_val_pred, mass_val_full)
        lgb_val_pred_kg = convert_to_kg_val(lgb_val_pred, mass_val_full)
        ensemble_val_pred_kg = convert_to_kg_val(ensemble_val_pred, mass_val_full)
        
        # Create validation results DataFrame
        val_results = pd.DataFrame({
            'fold': 1,  # Single fold for final training validation
            'idx': merged_df.iloc[val_idx_full]['idx'].values,
            'flight_id': merged_df.iloc[val_idx_full]['flight_id'].values,
            'aircraft_type': merged_df.iloc[val_idx_full][get_aircraft_column(merged_df)].values if get_aircraft_column(merged_df) else 'unknown',
            'actual_fuel_kg': y_actual.iloc[val_idx_full].values,
            'catboost_pred_kg': cat_val_pred_kg,
            'lightgbm_pred_kg': lgb_val_pred_kg,
            'ensemble_pred_kg': ensemble_val_pred_kg,
            'catboost_error': cat_val_pred_kg - y_actual.iloc[val_idx_full].values,
            'lightgbm_error': lgb_val_pred_kg - y_actual.iloc[val_idx_full].values,
            'ensemble_error': ensemble_val_pred_kg - y_actual.iloc[val_idx_full].values,
            'mass_above_oew': mass_val_full.values,
            'duration_sec': merged_df.iloc[val_idx_full]['interval_duration_sec'].values if 'interval_duration_sec' in merged_df.columns else np.nan
        })
        validation_results = [val_results]  # Put in list to match CV structure
        validation_df = pd.concat(validation_results, ignore_index=True)
        print(f"   ✓ Collected validation results from {len(validation_df)} samples")
    else:
        validation_results = []
        validation_df = None

    # Show summary of CV results as well (best/min, mean, std) for quick reference
    cv_min = np.min(cv_rmse['ensemble']) if cv_rmse['ensemble'] else float('nan')
    cv_max = np.max(cv_rmse['ensemble']) if cv_rmse['ensemble'] else float('nan')
    print(f"   CV Results Summary: Mean RMSE = {cv_mean:.2f}, Std = {cv_std:.2f}, Min = {cv_min:.2f}, Max = {cv_max:.2f}")

    # 7. Feature Importance
    print("\n7. Analyzing feature importance...")
    if args.algo == 'ensemble':
        # For ensemble, get feature importance from the final CatBoost model
        if hasattr(final_model['catboost'], 'feature_importances_'):
            cat_importance = final_model['catboost'].feature_importances_
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': cat_importance,
                'importance_std': np.zeros(len(X.columns))  # No std for single model
            }).sort_values('importance_mean', ascending=False)
            print("   Using CatBoost feature importance from final model")
        else:
            print("   Skipping feature importance for ensemble (no importance available)")
            importance_df = pd.DataFrame({'feature': X.columns, 'importance_mean': np.zeros(len(X.columns)), 'importance_std': np.zeros(len(X.columns))})
    else:
        # For single models, use CV fold importances if available, otherwise from final model
        if len(fold_importances) > 1 and not np.allclose(fold_importances[0], 0):
            # Use CV fold importances
            avg_importance = np.mean(fold_importances, axis=0)
            std_importance = np.std(fold_importances, axis=0)
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': avg_importance,
                'importance_std': std_importance,
            }).sort_values('importance_mean', ascending=False)
        else:
            # Use final model importance
            if hasattr(final_model, 'feature_importances_'):
                model_importance = final_model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance_mean': model_importance,
                    'importance_std': np.zeros(len(X.columns))  # No std for single model
                }).sort_values('importance_mean', ascending=False)
                print(f"   Using {args.algo} feature importance from final model")
            else:
                print(f"   Skipping feature importance for {args.algo} (no importance available)")
                importance_df = pd.DataFrame({'feature': X.columns, 'importance_mean': np.zeros(len(X.columns)), 'importance_std': np.zeros(len(X.columns))})

    importance_df.to_csv(feature_importance_path, index=False)
    print(f"   ✓ Saved feature importance to {feature_importance_path}")

    if args.algo != 'ensemble' or (args.algo == 'ensemble' and importance_df['importance_mean'].sum() > 0):
        print("\n   Top 20 Features by Importance:")
        print(importance_df.head(20)[['feature', 'importance_mean']].to_string(index=False))

        n_zero_importance = (importance_df['importance_mean'] == 0).sum()
        print(f"\n   Features with zero importance: {n_zero_importance} / {len(importance_df)}")
    else:
        n_zero_importance = 0

    # 8. Error Analysis on Validation Data
    if len(validation_results) > 0:
        print("\n8. Performing error analysis on validation data...")
        
        # Overall statistics
        print("\n   Overall Error Statistics:")
        for model_name in ['catboost', 'lightgbm', 'ensemble']:
            error_col = f'{model_name}_error'
            mean_error = validation_df[error_col].mean()
            median_error = validation_df[error_col].median()
            std_error = validation_df[error_col].std()
            mae = validation_df[error_col].abs().mean()
            
            over_pred = (validation_df[error_col] > 0).sum()
            under_pred = (validation_df[error_col] < 0).sum()
            over_pred_pct = over_pred / len(validation_df) * 100
            under_pred_pct = under_pred / len(validation_df) * 100
            
            print(f"\n   {model_name.upper()}:")
            print(f"     Mean Error: {mean_error:.2f} kg (bias)")
            print(f"     Median Error: {median_error:.2f} kg")
            print(f"     Std Error: {std_error:.2f} kg")
            print(f"     MAE: {mae:.2f} kg")
            print(f"     Over-prediction: {over_pred} ({over_pred_pct:.1f}%)")
            print(f"     Under-prediction: {under_pred} ({under_pred_pct:.1f}%)")
        
        # Error by aircraft type
        print("\n   Error by Aircraft Type (Ensemble):")
        aircraft_col = get_aircraft_column(validation_df)
        if aircraft_col:
            aircraft_stats = validation_df.groupby(aircraft_col).agg({
                'ensemble_error': ['mean', 'std', 'count'],
                'actual_fuel_kg': 'mean'
            }).round(2)
            aircraft_stats.columns = ['_'.join(col).strip() for col in aircraft_stats.columns]
            aircraft_stats = aircraft_stats.sort_values('ensemble_error_mean', key=abs, ascending=False)
            print(aircraft_stats.head(10).to_string())
        
        # Error by fuel consumption bins
        print("\n   Error by Fuel Consumption Bins (Ensemble):")
        validation_df['fuel_bin'] = pd.qcut(validation_df['actual_fuel_kg'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
        bin_stats = validation_df.groupby('fuel_bin', observed=True).agg({
            'ensemble_error': ['mean', 'std', 'count'],
            'actual_fuel_kg': ['mean', 'min', 'max']
        }).round(2)
        print(bin_stats.to_string())
        
        # Large errors analysis
        print("\n   Large Errors Analysis (Ensemble):")
        validation_df['ensemble_pct_error'] = (validation_df['ensemble_error'] / validation_df['actual_fuel_kg']) * 100
        large_errors = validation_df[validation_df['ensemble_pct_error'].abs() > 20].copy()
        print(f"     Predictions with >20% error: {len(large_errors)} ({len(large_errors)/len(validation_df)*100:.1f}%)")
        
        if len(large_errors) > 0:
            print(f"     Mean actual fuel for large errors: {large_errors['actual_fuel_kg'].mean():.2f} kg")
            print(f"     Mean predicted fuel for large errors: {large_errors['ensemble_pred_kg'].mean():.2f} kg")
            if aircraft_col in large_errors.columns:
                print(f"     Top aircraft types with large errors:")
                print(large_errors[aircraft_col].value_counts().head(5).to_string())
        
        # Save validation results and error analysis
        error_analysis_path = submissions_dir / f"likable-ant_v{version}_error_analysis.csv"
        validation_df.to_csv(error_analysis_path, index=False)
        print(f"\n   ✓ Saved error analysis to {error_analysis_path}")
        
        # Suggestions for improvement
        print("\n   Suggestions for Improvement:")
        mean_ensemble_error = validation_df['ensemble_error'].mean()
        if abs(mean_ensemble_error) > 10:
            if mean_ensemble_error > 0:
                print(f"     - Model is systematically over-predicting by {mean_ensemble_error:.2f} kg on average")
                print(f"     - Consider: Adding a bias correction term or adjusting target scaling")
            else:
                print(f"     - Model is systematically under-predicting by {abs(mean_ensemble_error):.2f} kg on average")
                print(f"     - Consider: Adding a bias correction term or adjusting target scaling")
        
        if aircraft_col in validation_df.columns:
            worst_aircraft = aircraft_stats.head(3).index.tolist()
            print(f"     - Aircraft types with highest errors: {worst_aircraft}")
            print(f"     - Consider: Aircraft-specific features or separate models per aircraft type")
        
        high_error_pct = (validation_df['ensemble_pct_error'].abs() > 20).sum() / len(validation_df) * 100
        if high_error_pct > 10:
            print(f"     - {high_error_pct:.1f}% of predictions have >20% error")
            print(f"     - Consider: Additional features for extreme cases, outlier handling")
    
    # 9. Predict on Rank Data
    print("\n9. Generating predictions for rank dataset...")

    X_rank, _ = prepare_features(rank_df, drop_cols)

    if args.feature_pruning > 0:
        X_rank = X_rank[top_features].copy()

    # Align columns
    missing_cols = set(X.columns) - set(X_rank.columns)
    for col in missing_cols:
        X_rank[col] = 0

    X_rank = X_rank[X.columns]

    # Ensure categorical columns are consistent
    for col in cat_cols:
        if col in X_rank.columns:
            train_cats = X[col].cat.categories
            rank_cats = X_rank[col].cat.categories
            new_cats = rank_cats.difference(train_cats)
            if len(new_cats) > 0:
                X[col] = X[col].cat.add_categories(new_cats)
                X_rank[col] = X_rank[col].cat.set_categories(train_cats.union(new_cats))

    print(f"   Rank feature matrix shape: {X_rank.shape}")

    # Predict with ensemble
    if isinstance(final_model, dict):
        # Ensemble case
        cat_rank_pred_raw = final_model['catboost'].predict(X_rank)
        lgb_rank_pred_raw = final_model['lightgbm'].predict(X_rank)
        rank_pred = (cat_rank_pred_raw + lgb_rank_pred_raw) / 2
    else:
        # Single model case
        if args.algo == 'xgboost':
            X_rank_for_pred = X_rank.drop(columns=cat_cols, errors='ignore')
        else:
            X_rank_for_pred = X_rank
        rank_pred = final_model.predict(X_rank_for_pred)
        cat_rank_pred_raw = rank_pred  # For single models, use same prediction
        lgb_rank_pred_raw = rank_pred

    # Convert back to actual fuel for each model
    mass_rank = (rank_df['ac_max_takeoff_weight_kg'] - rank_df['ac_operating_empty_weight_kg']).clip(lower=1.0)
    
    if args.target_type == 'residual':
        baseline_col = get_baseline_column(rank_df)
        baseline_rank = rank_df[baseline_col]
        cat_rank_pred = baseline_rank + cat_rank_pred_raw
        lgb_rank_pred = baseline_rank + lgb_rank_pred_raw
        rank_preds = baseline_rank + rank_pred
    elif args.target_type == 'specific':
        duration_hr_rank = rank_df['interval_duration_sec'] / 3600.0
        cat_rank_pred = cat_rank_pred_raw * mass_rank * duration_hr_rank
        lgb_rank_pred = lgb_rank_pred_raw * mass_rank * duration_hr_rank
        rank_preds = rank_pred * mass_rank * duration_hr_rank
    else:  # scaled
        cat_rank_pred = cat_rank_pred_raw * mass_rank
        lgb_rank_pred = lgb_rank_pred_raw * mass_rank
        rank_preds = rank_pred * mass_rank

    # Apply tail blending with physics baseline
    if args.tail_blending:
        baseline_col = get_baseline_column(rank_df)
        if baseline_col in rank_df.columns:
            print(f"\n   Applying tail blending (threshold={args.tail_threshold} kg)...")
            baseline_acropole = rank_df[baseline_col].values
            
            # Define blending weights
            tail_mask_high = baseline_acropole > args.tail_threshold
            tail_mask_mid = (baseline_acropole > args.tail_threshold / 2) & (baseline_acropole <= args.tail_threshold)
            
            n_high = tail_mask_high.sum()
            n_mid = tail_mask_mid.sum()
            
            # Blend: high fuel gets 80% physics, mid fuel gets 50% physics
            rank_preds_blended = rank_preds.copy()
            rank_preds_blended[tail_mask_high] = 0.8 * baseline_acropole[tail_mask_high] + 0.2 * rank_preds[tail_mask_high]
            rank_preds_blended[tail_mask_mid] = 0.5 * baseline_acropole[tail_mask_mid] + 0.5 * rank_preds[tail_mask_mid]
            
            print(f"   High-fuel intervals (>{args.tail_threshold} kg): {n_high} ({n_high/len(rank_preds)*100:.1f}%)")
            print(f"   Mid-fuel intervals ({args.tail_threshold/2:.0f}-{args.tail_threshold} kg): {n_mid} ({n_mid/len(rank_preds)*100:.1f}%)")
            print(f"   Mean prediction shift: {(rank_preds_blended - rank_preds).mean():.2f} kg")
            
            rank_preds = rank_preds_blended

    # Apply bias correction per fuel bin
    if args.bias_correction and bias_corrections is not None:
        print(f"\n   Applying per-fuel-bin bias correction...")
        rank_fuel_bins = pd.qcut(rank_preds, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
        for bin_name, bias in bias_corrections.items():
            mask = rank_fuel_bins == bin_name
            n_corrected = mask.sum()
            if n_corrected > 0:
                rank_preds[mask] -= bias
                print(f"   {bin_name}: corrected {n_corrected} intervals by {-bias:.2f} kg")

    # Save individual model predictions
    rank_predictions_df = pd.DataFrame({
        'idx': rank_df['idx'],
        'catboost_pred': cat_rank_pred,
        'lightgbm_pred': lgb_rank_pred,
        'ensemble_pred': rank_preds
    })
    rank_preds_path = submissions_dir / f"likable-ant_v{version}_rank_all_predictions.csv"
    rank_predictions_df.to_csv(rank_preds_path, index=False)
    print(f"   ✓ Saved all rank predictions to {rank_preds_path}")

    # Clip negatives
    rank_preds = np.maximum(rank_preds, 0.1)

    print(f"   Ensemble Predictions - Mean: {rank_preds.mean():.2f}, Std: {rank_preds.std():.2f}")
    print(f"   CatBoost Predictions - Mean: {cat_rank_pred.mean():.2f}, Std: {cat_rank_pred.std():.2f}")
    print(f"   LightGBM Predictions - Mean: {lgb_rank_pred.mean():.2f}, Std: {lgb_rank_pred.std():.2f}")

    # 10. Predict on Final Data (if available)
    if final_df is not None:
        print("\n10. Generating predictions for final dataset...")
        
        X_final, _ = prepare_features(final_df, drop_cols)
        
        if args.feature_pruning > 0:
            X_final = X_final[top_features].copy()
        
        # Align columns
        missing_cols = set(X.columns) - set(X_final.columns)
        for col in missing_cols:
            X_final[col] = 0
        
        X_final = X_final[X.columns]
        
        # Ensure categorical columns are consistent
        for col in cat_cols:
            if col in X_final.columns:
                train_cats = X[col].cat.categories
                final_cats = X_final[col].cat.categories
                new_cats = final_cats.difference(train_cats)
                if len(new_cats) > 0:
                    X[col] = X[col].cat.add_categories(new_cats)
                    X_final[col] = X_final[col].cat.set_categories(train_cats.union(new_cats))
        
        print(f"   Final feature matrix shape: {X_final.shape}")
        
        # Predict with ensemble
        if isinstance(final_model, dict):
            # Ensemble case
            cat_final_pred_raw = final_model['catboost'].predict(X_final)
            lgb_final_pred_raw = final_model['lightgbm'].predict(X_final)
            final_pred = (cat_final_pred_raw + lgb_final_pred_raw) / 2
        else:
            # Single model case
            if args.algo == 'xgboost':
                X_final_for_pred = X_final.drop(columns=cat_cols, errors='ignore')
            else:
                X_final_for_pred = X_final
            final_pred = final_model.predict(X_final_for_pred)
            cat_final_pred_raw = final_pred  # For single models, use same prediction
            lgb_final_pred_raw = final_pred
        
        # Convert back to actual fuel
        mass_final = (final_df['ac_max_takeoff_weight_kg'] - final_df['ac_operating_empty_weight_kg']).clip(lower=1.0)
        
        if args.target_type == 'residual':
            baseline_col = get_baseline_column(final_df)
            baseline_final = final_df[baseline_col]
            cat_final_pred = baseline_final + cat_final_pred_raw
            lgb_final_pred = baseline_final + lgb_final_pred_raw
            final_preds = baseline_final + final_pred
        elif args.target_type == 'specific':
            duration_hr_final = final_df['interval_duration_sec'] / 3600.0
            cat_final_pred = cat_final_pred_raw * mass_final * duration_hr_final
            lgb_final_pred = lgb_final_pred_raw * mass_final * duration_hr_final
            final_preds = final_pred * mass_final * duration_hr_final
        else:  # scaled
            cat_final_pred = cat_final_pred_raw * mass_final
            lgb_final_pred = lgb_final_pred_raw * mass_final
            final_preds = final_pred * mass_final
        
        # Apply tail blending with physics baseline
        if args.tail_blending:
            baseline_col = get_baseline_column(final_df)
            if baseline_col in final_df.columns:
                print(f"\n   Applying tail blending to final (threshold={args.tail_threshold} kg)...")
                baseline_acropole_final = final_df[baseline_col].values
                
                # Define blending weights
                tail_mask_high = baseline_acropole_final > args.tail_threshold
                tail_mask_mid = (baseline_acropole_final > args.tail_threshold / 2) & (baseline_acropole_final <= args.tail_threshold)
                
                n_high = tail_mask_high.sum()
                n_mid = tail_mask_mid.sum()
                
                # Blend: high fuel gets 80% physics, mid fuel gets 50% physics
                final_preds_blended = final_preds.copy()
                final_preds_blended[tail_mask_high] = 0.8 * baseline_acropole_final[tail_mask_high] + 0.2 * final_preds[tail_mask_high]
                final_preds_blended[tail_mask_mid] = 0.5 * baseline_acropole_final[tail_mask_mid] + 0.5 * final_preds[tail_mask_mid]
                
                print(f"   High-fuel intervals (>{args.tail_threshold} kg): {n_high} ({n_high/len(final_preds)*100:.1f}%)")
                print(f"   Mid-fuel intervals ({args.tail_threshold/2:.0f}-{args.tail_threshold} kg): {n_mid} ({n_mid/len(final_preds)*100:.1f}%)")
                print(f"   Mean prediction shift: {(final_preds_blended - final_preds).mean():.2f} kg")
                
                final_preds = final_preds_blended

        # Apply bias correction per fuel bin
        if args.bias_correction and bias_corrections is not None:
            print(f"\n   Applying per-fuel-bin bias correction to final...")
            final_fuel_bins = pd.qcut(final_preds, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            for bin_name, bias in bias_corrections.items():
                mask = final_fuel_bins == bin_name
                n_corrected = mask.sum()
                if n_corrected > 0:
                    final_preds[mask] -= bias
                    print(f"   {bin_name}: corrected {n_corrected} intervals by {-bias:.2f} kg")
        
        # Clip negatives
        final_preds = np.maximum(final_preds, 0.1)
        cat_final_pred = np.maximum(cat_final_pred, 0.1)
        lgb_final_pred = np.maximum(lgb_final_pred, 0.1)
        
        # Save final predictions
        final_predictions_df = pd.DataFrame({
            'idx': final_df['idx'],
            'catboost_pred': cat_final_pred,
            'lightgbm_pred': lgb_final_pred,
            'ensemble_pred': final_preds
        })
        final_preds_path = submissions_dir / f"likable-ant_v{version}_final_all_predictions.csv"
        final_predictions_df.to_csv(final_preds_path, index=False)
        print(f"   ✓ Saved all final predictions to {final_preds_path}")
        
        print(f"   Ensemble Predictions - Mean: {final_preds.mean():.2f}, Std: {final_preds.std():.2f}")
        print(f"   CatBoost Predictions - Mean: {cat_final_pred.mean():.2f}, Std: {cat_final_pred.std():.2f}")
        print(f"   LightGBM Predictions - Mean: {lgb_final_pred.mean():.2f}, Std: {lgb_final_pred.std():.2f}")

    # 11. Save Submission
    print("\n11. Saving submission...")

    submission_df = pd.read_parquet(submission_template_path)
    pred_series = pd.Series(rank_preds, index=rank_df['idx'])
    submission_df['fuel_kg'] = submission_df['idx'].map(pred_series)

    # Check for missing predictions and diagnose
    missing_preds = submission_df['fuel_kg'].isna().sum()
    if missing_preds > 0:
        print(f"\n   ERROR: {missing_preds} rows missing predictions!")
        missing_idx = submission_df[submission_df['fuel_kg'].isna()]['idx'].values
        print(f"   Missing idx values: {missing_idx[:20]}...")  # Show first 20
        
        # Check which rows are in submission but not in rank features
        rank_idx_set = set(rank_df['idx'].values)
        submission_idx_set = set(submission_df['idx'].values)
        missing_in_rank = submission_idx_set - rank_idx_set
        
        if len(missing_in_rank) > 0:
            print(f"   {len(missing_in_rank)} idx in submission but NOT in rank features:")
            print(f"   This means Stage 8 failed to generate features for these intervals!")
            print(f"   Missing idx: {sorted(list(missing_in_rank))[:20]}...")
        
        median_val = np.median(rank_preds)
        print(f"   Filling with median: {median_val:.2f}")
        print(f"   WARNING: This will hurt your score! Fix Stage 8 to generate complete features.")
        submission_df['fuel_kg'] = submission_df['fuel_kg'].fillna(median_val)

    assert submission_df['fuel_kg'].notna().all(), "ERROR: Submission contains NaN!"
    assert (submission_df['fuel_kg'] >= 0).all(), "ERROR: Submission contains negative values!"

    submission_df.to_parquet(output_path)
    print(f"   ✓ Saved submission to {output_path}")

    # 12. Save Metadata
    print("\n12. Saving metadata...")

    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'target_type': args.target_type,
        'dataset': args.dataset,
        'dataset_type': args.dataset_type,
        'algo': 'ensemble',
        'cv_folds': args.cv_folds,
        'skip_cv': args.skip_cv,
        'cv_rmse_catboost_mean': float(np.mean(cv_rmse['catboost'])) if cv_rmse['catboost'] else 0.0,
        'cv_rmse_lightgbm_mean': float(np.mean(cv_rmse['lightgbm'])) if cv_rmse['lightgbm'] else 0.0,
        'cv_rmse_ensemble_mean': float(cv_mean),
        'cv_rmse_ensemble_std': float(cv_std),
        'feature_pruning': args.feature_pruning,
        'use_monotone': args.use_monotone,
        'monotone_constraints_count': n_monotone,
        'cv_type': args.cv_type,
        'tail_blending': args.tail_blending,
        'tail_threshold': args.tail_threshold,
        'bias_correction': args.bias_correction,
        'bias_corrections': {str(k): float(v) for k, v in bias_corrections.items()} if bias_corrections else None,
        'n_features': X.shape[1],
        'n_categorical': len(cat_cols),
        'model_params': best_params,
        'features_zero_importance': int(n_zero_importance),
        'rank_predictions_mean': float(rank_preds.mean()),
        'rank_predictions_std': float(rank_preds.std()),
        'aircraft_type_filtering': 'enabled',
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved metadata to {metadata_path}")

    # Save separate submissions for individual models
    if isinstance(final_model, dict):
        print("\n13. Saving separate submissions for individual models...")

        # LightGBM submission
        lgb_pred_series = pd.Series(lgb_rank_pred, index=rank_df['idx'])
        lgb_submission_df = pd.read_parquet(submission_template_path)
        lgb_submission_df['fuel_kg'] = lgb_submission_df['idx'].map(lgb_pred_series)

        # Handle missing predictions
        missing_lgb = lgb_submission_df['fuel_kg'].isna().sum()
        if missing_lgb > 0:
            median_lgb = np.median(lgb_rank_pred)
            lgb_submission_df['fuel_kg'] = lgb_submission_df['fuel_kg'].fillna(median_lgb)

        lgb_submission_df['fuel_kg'] = np.maximum(lgb_submission_df['fuel_kg'], 0.1)

        lgb_version = get_next_version(submissions_dir)
        lgb_output_path = submissions_dir / f"likable-ant_v{lgb_version}.parquet"
        lgb_submission_df.to_parquet(lgb_output_path)
        print(f"   ✓ Saved LightGBM submission to {lgb_output_path}")

        # LightGBM metadata
        lgb_metadata = metadata.copy()
        lgb_metadata['version'] = lgb_version
        lgb_metadata['algo'] = 'lightgbm'
        lgb_metadata['timestamp'] = datetime.now().isoformat()
        lgb_metadata['rank_predictions_mean'] = float(lgb_rank_pred.mean())
        lgb_metadata['rank_predictions_std'] = float(lgb_rank_pred.std())
        lgb_metadata_path = submissions_dir / f"likable-ant_v{lgb_version}_metadata.json"
        with open(lgb_metadata_path, 'w') as f:
            json.dump(lgb_metadata, f, indent=2)
        print(f"   ✓ Saved LightGBM metadata to {lgb_metadata_path}")

        # CatBoost submission
        cat_pred_series = pd.Series(cat_rank_pred, index=rank_df['idx'])
        cat_submission_df = pd.read_parquet(submission_template_path)
        cat_submission_df['fuel_kg'] = cat_submission_df['idx'].map(cat_pred_series)

        # Handle missing predictions
        missing_cat = cat_submission_df['fuel_kg'].isna().sum()
        if missing_cat > 0:
            median_cat = np.median(cat_rank_pred)
            cat_submission_df['fuel_kg'] = cat_submission_df['fuel_kg'].fillna(median_cat)

        cat_submission_df['fuel_kg'] = np.maximum(cat_submission_df['fuel_kg'], 0.1)

        cat_version = get_next_version(submissions_dir)
        cat_output_path = submissions_dir / f"likable-ant_v{cat_version}.parquet"
        cat_submission_df.to_parquet(cat_output_path)
        print(f"   ✓ Saved CatBoost submission to {cat_output_path}")

        # CatBoost metadata
        cat_metadata = metadata.copy()
        cat_metadata['version'] = cat_version
        cat_metadata['algo'] = 'catboost'
        cat_metadata['timestamp'] = datetime.now().isoformat()
        cat_metadata['rank_predictions_mean'] = float(cat_rank_pred.mean())
        cat_metadata['rank_predictions_std'] = float(cat_rank_pred.std())
        cat_metadata_path = submissions_dir / f"likable-ant_v{cat_version}_metadata.json"
        with open(cat_metadata_path, 'w') as f:
            json.dump(cat_metadata, f, indent=2)
        print(f"   ✓ Saved CatBoost metadata to {cat_metadata_path}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"Version: {version}")
    print(f"\nCV RMSE Results:")
    if cv_rmse['catboost']:
        print(f"  CatBoost:  {np.mean(cv_rmse['catboost']):.2f} ± {np.std(cv_rmse['catboost']):.2f}")
        print(f"  LightGBM:  {np.mean(cv_rmse['lightgbm']):.2f} ± {np.std(cv_rmse['lightgbm']):.2f}")
    print(f"  Ensemble:  {cv_mean:.2f} ± {cv_std:.2f}")
    print(f"\nOutput Files:")
    print(f"  Submission: {output_path}")
    print(f"  Rank predictions: {rank_preds_path}")
    if final_df is not None:
        print(f"  Final predictions: {final_preds_path}")
    if not args.skip_cv:
        print(f"  Error analysis: {error_analysis_path}")
    if isinstance(final_model, dict):
        print(f"  LightGBM Submission: {lgb_output_path}")
        print(f"  CatBoost Submission: {cat_output_path}")

if __name__ == "__main__":
    main()