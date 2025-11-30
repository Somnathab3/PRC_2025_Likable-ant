"""
Script to train a LightGBM model with scaled fuel_kg target.
Scaling: fuel_kg / (MTOW_kg - ac_operating_empty_weight_kg). The script supports
multiple MTOW column names (ac_max_takeoff_weight_kg, ac_MTOW_kg, ac_MTOW_lb).
Uses Stage_8 consolidated features by default and accepts CLI args. Includes
cross-validation grouped by flight_id and feature importance analysis.
"""

import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from pathlib import Path
import json
import re
from datetime import datetime
from sklearn.covariance import EmpiricalCovariance

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

def create_physics_features(df):
    """Create physics-based interaction features."""
    print("   Creating physics-based interaction features...")

    # Mass above OEW (useful for many interactions), handle several MTOW variants
    mtow_col = None
    if 'ac_max_takeoff_weight_kg' in df.columns:
        mtow_col = 'ac_max_takeoff_weight_kg'
    elif 'ac_MTOW_kg' in df.columns:
        mtow_col = 'ac_MTOW_kg'
    elif 'ac_MTOW_lb' in df.columns:
        # convert lbs to kg
        mtow_col = 'ac_MTOW_lb'

    if mtow_col and 'ac_operating_empty_weight_kg' in df.columns:
        if mtow_col == 'ac_MTOW_lb':
            df['ac_MTOW_kg'] = df['ac_MTOW_lb'] / 2.2046226218
            mtow_used = 'ac_MTOW_kg'
        else:
            mtow_used = mtow_col
        df['mass_above_oew'] = df[mtow_used] - df['ac_operating_empty_weight_kg']
        df['mass_above_oew'] = df['mass_above_oew'].clip(lower=1.0)
        print(f"      Mass above OEW created using {mtow_used} and ac_operating_empty_weight_kg")

    # 1. Efficiency-weighted distance/time
    if 'int_ground_distance_nm' in df.columns and 'mass_above_oew' in df.columns:
        df['distance_per_mass'] = df['int_ground_distance_nm'] / df['mass_above_oew']

    if 'interval_duration_sec' in df.columns and 'mass_above_oew' in df.columns:
        df['time_per_mass'] = df['interval_duration_sec'] / df['mass_above_oew']

    # 2. Physics model error × operating conditions
    if 'flt_fuel_pct_error_acropole_openap' in df.columns and 'mass_above_oew' in df.columns:
        df['acropole_openap_error_x_mass'] = df['flt_fuel_pct_error_acropole_openap'] * df['mass_above_oew']

    # Use origin_dest_gcdist_nm instead of flt_ground_distance_nm
    if 'flt_fuel_pct_error_acropole_openap' in df.columns and 'origin_dest_gcdist_nm' in df.columns:
        df['acropole_openap_error_x_distance'] = df['flt_fuel_pct_error_acropole_openap'] * df['origin_dest_gcdist_nm']

    # 3. Aircraft "shape" × cruise state (if available)
    if 'ac_mach_altitude_product' in df.columns and 'ac_efficiency_index' in df.columns:
        df['mach_altitude_eff'] = df['ac_mach_altitude_product'] * df['ac_efficiency_index']

    if 'cruise_median_tas' in df.columns and 'ac_aspect_ratio' in df.columns:
        df['cruise_tas_x_aspect_ratio'] = df['cruise_median_tas'] * df['ac_aspect_ratio']

    # 4. Phase imbalance features - use available columns
    # Use cruise_distance_nm as level distance approximation (fallbacks supported)
    # 1) cruise_distance_nm if available
    # 2) origin_dest_gcdist_nm * flt_phase_cruise_fraction if cruise fraction exists
    # 3) int_ground_distance_nm * flt_phase_cruise_fraction as last resort
    if 'origin_dest_gcdist_nm' in df.columns:
        if 'cruise_distance_nm' in df.columns:
            df['level_distance_fraction'] = df['cruise_distance_nm'] / (df['origin_dest_gcdist_nm'] + 1e-6)
        elif 'flt_phase_cruise_fraction' in df.columns:
            df['level_distance_fraction'] = (df['origin_dest_gcdist_nm'] * df['flt_phase_cruise_fraction']) / (df['origin_dest_gcdist_nm'] + 1e-6)
        elif 'int_ground_distance_nm' in df.columns and 'flt_phase_cruise_fraction' in df.columns:
            df['level_distance_fraction'] = (df['int_ground_distance_nm'] * df['flt_phase_cruise_fraction']) / (df['origin_dest_gcdist_nm'] + 1e-6)

    # Use total_flight_duration_sec instead of flt_duration_sec
    if 'climb_time_sec' in df.columns and 'total_flight_duration_sec' in df.columns:
        df['climb_time_fraction'] = df['climb_time_sec'] / (df['total_flight_duration_sec'] + 1e-6)

    if 'descent_time_sec' in df.columns and 'total_flight_duration_sec' in df.columns:
        df['descent_time_fraction'] = df['descent_time_sec'] / (df['total_flight_duration_sec'] + 1e-6)

    # 5. Fuel efficiency indicators - use available fuel per distance or per mass columns
    if 'int_acropole_fuel_per_tow' in df.columns and 'int_ground_distance_nm' in df.columns:
        df['fuel_efficiency_per_nm'] = df['int_acropole_fuel_per_tow'] / (df['int_ground_distance_nm'] + 1e-6)
        print('      Using int_acropole_fuel_per_tow for fuel_efficiency_per_nm')
    elif 'int_acropole_fuel_per_mass' in df.columns and 'int_ground_distance_nm' in df.columns and 'mass_above_oew' in df.columns:
        # fuel per mass can be converted to kgs per nm via mass and distance
        df['fuel_efficiency_per_nm'] = (df['int_acropole_fuel_per_mass'] * df['mass_above_oew']) / (df['int_ground_distance_nm'] + 1e-6)
        print('      Using int_acropole_fuel_per_mass and mass_above_oew for fuel_efficiency_per_nm')

    return df

def get_monotone_constraints(feature_names):
    """
    Define monotone constraints for physically meaningful features.
    Returns a list of constraints: +1 (monotone increasing), -1 (decreasing), 0 (no constraint)
    """
    constraints = []

    # Define features that should monotonically increase fuel consumption
    monotone_increasing = [
        'int_ground_distance_nm',
        'interval_duration_sec',
        'origin_dest_gcdist_nm',  # Use available distance column
        'total_flight_duration_sec',  # Use available duration column
        'ac_max_takeoff_weight_kg',
        'ac_MTOW_kg',
        'ac_MTOW_lb',
        'mass_above_oew',
        'int_time_since_takeoff_sec',  # Use available time column
        'int_time_before_landing_sec',  # Use available time column
        'climb_time_sec',
        'descent_time_sec',
        'cruise_distance_nm',  # Use available level distance
        'climb_distance_nm',
        'descent_distance_nm',
    ]

    for feature in feature_names:
        if any(mono_feat in feature for mono_feat in monotone_increasing):
            constraints.append(1)
        else:
            constraints.append(0)

    return constraints

def main():
    parser = argparse.ArgumentParser(description='Stage 9 - Train scaled fuel model')
    parser.add_argument('--train-features', default=r'F:/PRC_2025/Likable-ant_v1/data/processed/Stage_8_Consolidated_Features/consolidated_features_train_imputed_lightgbm.parquet', help='Path to train features')
    parser.add_argument('--rank-features', default=r'F:/PRC_2025/Likable-ant_v1/data/processed/Stage_8_Consolidated_Features/consolidated_features_rank_imputed_lightgbm.parquet', help='Path to rank features')
    parser.add_argument('--final-features', default=r'F:/PRC_2025/Likable-ant_v1/data/processed/Stage_8_Consolidated_Features/consolidated_features_final_imputed_lightgbm.parquet', help='Path to final features (optional)')
    parser.add_argument('--raw-fuel', default=r'F:/PRC_2025/Likable-ant_v1/data/raw/fuel_train.parquet', help='Path to raw fuel train data')
    parser.add_argument('--submission-template', default=r'F:/PRC_2025/Likable-ant_v1/data/raw/fuel_rank_submission.parquet', help='Submission template')
    parser.add_argument('--sample', type=int, default=0, help='If > 0, only use this many rows from train and rank for a faster debug run')
    args = parser.parse_args()

    # Define paths - UPDATED TO USE NEW IMPUTED DATASETS
    train_features_path = Path(args.train_features)
    rank_features_path = Path(args.rank_features)
    final_features_path = Path(args.final_features)
    raw_fuel_path = Path(args.raw_fuel)
    submission_template_path = Path(args.submission_template)
    
    # Output paths
    submissions_dir = Path("F:/PRC_2025/submissions")
    version = get_next_version(submissions_dir)
    
    output_path = submissions_dir / f"likable-ant_v{version}.parquet"
    feature_importance_path = submissions_dir / f"likable-ant_v{version}_feature_importance.csv"
    metadata_path = submissions_dir / f"likable-ant_v{version}_metadata.json"
    
    print("=" * 80)
    print(f"SCALED FUEL MODEL - CV WITH FLIGHT_ID GROUPING (v{version})")
    print("=" * 80)
    
    # 1. Load Data
    print("\n1. Loading datasets...")
    train_df = pd.read_parquet(train_features_path)
    rank_df = pd.read_parquet(rank_features_path)
    # optional final features file (only read if exists)
    final_df = None
    if final_features_path.exists():
        final_df = pd.read_parquet(final_features_path)
    
    fuel_df = pd.read_parquet(raw_fuel_path)
    
    # Drop columns with any NaN
    print("   Dropping columns with any NaN...")
    initial_train_cols = train_df.shape[1]
    train_df = train_df.dropna(axis=1, how='any')
    dropped_train = initial_train_cols - train_df.shape[1]
    print(f"   Dropped {dropped_train} columns from train_df")
    
    initial_rank_cols = rank_df.shape[1]
    rank_df = rank_df.dropna(axis=1, how='any')
    dropped_rank = initial_rank_cols - rank_df.shape[1]
    print(f"   Dropped {dropped_rank} columns from rank_df")
    
    if final_df is not None:
        initial_final_cols = final_df.shape[1]
        final_df = final_df.dropna(axis=1, how='any')
        dropped_final = initial_final_cols - final_df.shape[1]
        print(f"   Dropped {dropped_final} columns from final_df")
    
    print(f"   Train features shape: {train_df.shape}")
    print(f"   Rank features shape: {rank_df.shape}")
    if final_df is not None:
        print(f"   Final features shape: {final_df.shape}")
    
    # Create additional features for rank data
    print("\n   Checking feature availability for rank data...")
    rank_missing_features = []

    # # Check for airport columns - use origin_icao and destination_icao instead of adep/ades
    # if 'origin_icao' in rank_df.columns and 'destination_icao' in rank_df.columns:
    #     rank_df['same_airport'] = (rank_df['origin_icao'] == rank_df['destination_icao']).astype(str)
    #     rank_df['origin_destination_icao'] = rank_df['origin_icao'].astype(str) + '_' + rank_df['destination_icao'].astype(str)
    #     print("   ✅ Created same_airport and origin_destination_icao features")
    # else:
    #     print("   ⚠️  Missing origin_icao or destination_icao columns")
    #     rank_df['same_airport'] = 'False'
    #     rank_df['origin_destination_icao'] = 'UNKNOWN_UNKNOWN'
    #     rank_missing_features.extend(['origin_icao', 'destination_icao'])
    
    # Create physics-based features
    print("\n1.5. Engineering physics-based features...")
    train_df = create_physics_features(train_df)
    rank_df = create_physics_features(rank_df)

    # If sample mode requested, reduce sizes for quick debugging (apply after feature engineering)
    if args.sample and args.sample > 0:
        sample_n = int(args.sample)
        train_df = train_df.sample(n=min(sample_n, len(train_df)), random_state=42)
        rank_df = rank_df.sample(n=min(sample_n, len(rank_df)), random_state=42)
        fuel_df = fuel_df[fuel_df['idx'].isin(train_df['idx'])]
        print(f"   Running in sample mode: {sample_n} rows (or less) per dataset")

    # Debug: Check which physics features were created
    physics_features_created = [
        'mass_above_oew', 'distance_per_mass', 'time_per_mass',
        'acropole_openap_error_x_mass', 'acropole_openap_error_x_distance',
        'mach_altitude_eff', 'cruise_tas_x_aspect_ratio',
        'level_distance_fraction', 'climb_time_fraction', 'descent_time_fraction',
        'fuel_efficiency_per_nm'
    ]

    print("   Physics features created:")
    created_count = 0
    for feat in physics_features_created:
        if feat in train_df.columns:
            nan_pct = train_df[feat].isna().mean() * 100
            print(f"   ✅ {feat:<30} ({nan_pct:.1f}% NaN)")
            created_count += 1
        else:
            print(f"   ❌ {feat:<30} (missing)")
    print(f"   Successfully created {created_count}/{len(physics_features_created)} physics features")
    
    # DIAGNOSTICS: Check correlations and covariance between features and fuel_kg (pre-cleaning)
    def diagnose_leakage_pre(merged_sample_df):
        """Compute cov/corr between numerical features and fuel_kg to detect potential leakage."""
        print("\nDIAGNOSTICS: Leakage and covariance checks (pre-cleaning)")
        numeric_cols = merged_sample_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove the target itself from numeric columns
        numeric_cols = [c for c in numeric_cols if c not in ['fuel_kg', 'fuel_kg_scaled', 'fuel_kg_actual']]
        if not numeric_cols:
            print("   No numeric columns available for leakage check.")
            return

        # Compute Pearson correlations with fuel_kg
        corrs = merged_sample_df[numeric_cols].corrwith(merged_sample_df['fuel_kg']).abs().sort_values(ascending=False)
        print("   Top correlations with fuel_kg:")
        print(corrs.head(20).to_string())

        # Compute covariance (numpy) between each numeric feature and fuel_kg
        covs = {}
        y = merged_sample_df['fuel_kg'].values
        for col in numeric_cols:
            x = merged_sample_df[col].values
            if np.all(np.isnan(x)):
                covs[col] = np.nan
            else:
                covs[col] = np.cov(x, y, rowvar=False)[0, 1]
        covs_series = pd.Series(covs).abs().sort_values(ascending=False)
        print('\n   Top absolute covariances with fuel_kg:')
        print(covs_series.head(20).to_string())

        # Find columns equal to the target (perfect leakage)
        equal_cols = []
        for col in numeric_cols:
            if merged_sample_df[col].equals(merged_sample_df['fuel_kg']):
                equal_cols.append(col)
            else:
                try:
                    if np.allclose(merged_sample_df[col].fillna(0).values, merged_sample_df['fuel_kg'].fillna(0).values):
                        equal_cols.append(col)
                except Exception:
                    pass
        if equal_cols:
            print('\n   WARNING: The following columns are effectively equal to fuel_kg (perfect leakage):')
            print('   - ' + '\n   - '.join(equal_cols))
        else:
            print('\n   No columns are exactly equal to fuel_kg (no perfect equality detected).')

        # High correlation (potential leakage)
        high_corrs = corrs[corrs > 0.95].index.tolist()
        if high_corrs:
            print('\n   High-correlation candidates (abs corr > 0.95):')
            for col in high_corrs:
                print(f"   - {col}: corr={corrs[col]:.6f}, cov={covs_series.get(col, np.nan):.6f}")
        else:
            print('\n   No very-high-correlation features found (abs corr > 0.95).')

        # Check for columns that are obviously derived from fuel, e.g. contain 'fuel' in name
        fuel_like_cols = [c for c in merged_sample_df.columns if 'fuel' in c and c not in ['fuel_kg', 'fuel_kg_scaled', 'fuel_kg_actual']]
        if fuel_like_cols:
            print('\n   Columns with "fuel" in the name (possible leakage):')
            for c in fuel_like_cols:
                print(f"   - {c}")
        else:
            print('\n   No columns with "fuel" in the name detected (besides target).')

    # Diagnostic run on a subset (if dataset large, reduce rows) - premerging with fuel happens later, but we can check train_df and fuel separately.
    try:
        debug_sample = train_df.merge(fuel_df[['idx', 'fuel_kg']], on='idx', how='left').head(2000)
        diagnose_leakage_pre(debug_sample)
    except Exception as ex:
        print('   Diagnostic pre-cleaning check failed:', str(ex))
    
    # 2. Merge Target & Clean
    print("\n2. Preparing training data...")
    merged_df = train_df.merge(fuel_df[['idx', 'fuel_kg']], on='idx', how='left')
    
    # Remove rows with missing target
    initial_len = len(merged_df)
    merged_df = merged_df.dropna(subset=['fuel_kg'])
    print(f"   Removed {initial_len - len(merged_df)} rows with missing fuel_kg")

    # Keep rows where has_trajectory_data is True
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

    
    print(f"   Final training data: {merged_df.shape}")
    print(f"   Target (fuel_kg) - Mean: {merged_df['fuel_kg'].mean():.2f}, Std: {merged_df['fuel_kg'].std():.2f}")
    
    # Create additional features
    print("\n3.5. Creating additional features...")
    if 'origin_icao' in merged_df.columns and 'destination_icao' in merged_df.columns:
        merged_df['same_airport'] = (merged_df['origin_icao'] == merged_df['destination_icao']).astype(str)
        # merged_df['origin_destination_icao'] = merged_df['origin_icao'].astype(str) + '_' + merged_df['destination_icao'].astype(str)
        print("   ✅ Created same_airport feature")
    else:
        print("   ⚠️  Missing origin_icao or destination_icao columns")
        merged_df['same_airport'] = 'False'
        # merged_df['origin_destination_icao'] = 'UNKNOWN_UNKNOWN'
    
    # Keep rows where has_trajectory_data is True
    #if 'has_trajectory_data' in merged_df.columns:
    #    initial_len = len(merged_df)
    #    merged_df = merged_df[merged_df['has_trajectory_data'] == True]
    #    print(f"   Kept {len(merged_df)} rows where has_trajectory_data is True (removed {initial_len - len(merged_df)} others)")
    
    # 3. Create Scaled Target
    print("\n3. Creating scaled target...")
    # Check for required columns and robustly determine MTOW column
    mtow_candidates = ['ac_max_takeoff_weight_kg', 'ac_MTOW_kg', 'ac_MTOW_lb']
    mtow_col = None
    for c in mtow_candidates:
        if c in merged_df.columns:
            mtow_col = c
            break

    if 'ac_operating_empty_weight_kg' not in merged_df.columns or mtow_col is None:
        print("   ERROR: Required aircraft weight columns not found!")
        print(f"   Available columns: {sorted(merged_df.columns.tolist())}")
        return

    # Calculate scaling factor (mass above OEW) using available column
    if mtow_col == 'ac_MTOW_lb':
        merged_df['ac_MTOW_kg'] = merged_df['ac_MTOW_lb'] / 2.2046226218
        mtow_used = 'ac_MTOW_kg'
    else:
        mtow_used = mtow_col

    # If mass_above_oew wasn't created previously, create it
    if 'mass_above_oew' not in merged_df.columns:
        merged_df['mass_above_oew'] = merged_df[mtow_used] - merged_df['ac_operating_empty_weight_kg']
        merged_df['mass_above_oew'] = merged_df['mass_above_oew'].clip(lower=1.0)

    mass_above_oew = merged_df['mass_above_oew']
    
    # Create scaled target (multiply by 10000 for better numerical precision)
    scaling_factor = 10000.0
    merged_df['fuel_kg_scaled'] = (merged_df['fuel_kg'] / mass_above_oew) * scaling_factor
    
    print(f"   Scaled target statistics (multiplied by {scaling_factor}):")
    print(f"   Mean: {merged_df['fuel_kg_scaled'].mean():.6f}")
    print(f"   Std: {merged_df['fuel_kg_scaled'].std():.6f}")
    print(f"   Min: {merged_df['fuel_kg_scaled'].min():.6f}")
    print(f"   Max: {merged_df['fuel_kg_scaled'].max():.6f}")
    
    # 4. Prepare Features
    print("\n4. Preparing features...")

    # Base columns to drop (targets, IDs, timestamps)
    drop_cols = ['fuel_kg', 'fuel_kg_scaled', 'fuel_kg_actual', 'idx', 'flight_id']

    # Drop datetime columns
    datetime_cols = merged_df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    datetime_cols += [c for c in merged_df.columns if c.endswith('_ts')]
    datetime_cols = list(set(datetime_cols))
    drop_cols += datetime_cols
    print(f"   Dropping {len(datetime_cols)} datetime columns")

    # Drop name columns (airport names, etc.)
    name_cols = [c for c in merged_df.columns if c.startswith('name_')]
    drop_cols += name_cols
    print(f"   Dropping {len(name_cols)} name columns")

    # Drop segment has_data flags (already handled in imputation)
    seg_has_data_cols = [c for c in merged_df.columns if c.startswith('seg') and c.endswith('_has_data')]
    drop_cols += seg_has_data_cols
    print(f"   Dropping {len(seg_has_data_cols)} segment has_data columns")

    # Drop encoded categorical columns (keep only raw categoricals)
    encoded_cols = [c for c in merged_df.columns if 'encoded' in c.lower()]
    drop_cols += encoded_cols
    print(f"   Dropping {len(encoded_cols)} encoded categorical columns")

    # Drop raw METAR text columns
    metar_text_cols = [c for c in merged_df.columns if 'raw_text' in c.lower()]
    drop_cols += metar_text_cols
    print(f"   Dropping {len(metar_text_cols)} raw METAR text columns")

    # Drop fuel leakage columns
    fuel_leakage_cols = [c for c in merged_df.columns if 'fuel_kg_actual' in c or c == 'fuel_kg_actual']
    drop_cols += fuel_leakage_cols
    if fuel_leakage_cols:
        print(f"   Dropping {len(fuel_leakage_cols)} fuel_kg_actual columns to prevent data leakage")

    # Drop mass change columns (potential leakage)
    mass_change_cols = [c for c in merged_df.columns if 'mass_change' in c.lower()]
    drop_cols += mass_change_cols
    print(f"   Dropping {len(mass_change_cols)} mass change columns")

    # Remove duplicates
    drop_cols = list(set(drop_cols))

    print(f"   Total columns to drop: {len(drop_cols)}")
    print(f"   Columns being dropped: {sorted(drop_cols)[:10]}..." if len(drop_cols) > 10 else f"   Columns being dropped: {sorted(drop_cols)}")
    
    # Save flight_id for grouping before dropping
    groups = merged_df['flight_id'].values
    
    X = merged_df.drop(columns=drop_cols, errors='ignore')
    y = merged_df['fuel_kg_scaled']  # Use scaled target
    y_actual = merged_df['fuel_kg']  # Keep actual for validation
    
    # Identify all object/categorical columns
    all_cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Only keep aircraft_type and origin_destination_icao categorical features
    cat_cols = [c for c in all_cat_cols if c in ['aircraft_type', ]]#'origin_destination_icao'
    
    # Drop all other categorical columns
    drop_cat_cols = [c for c in all_cat_cols if c not in cat_cols]
    if drop_cat_cols:
        print(f"   Dropping {len(drop_cat_cols)} non-aircraft_type and non-origin_destination_icao categorical columns")
        X = X.drop(columns=drop_cat_cols)
    
    # Convert aircraft_type column to categorical
    cat_mappings = {}
    for col in cat_cols:
        X[col] = X[col].astype(str).astype('category')
        cat_mappings[col] = X[col].cat.categories.tolist()
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Categorical features (aircraft_type and origin_destination_icao): {len(cat_cols)}")
    print(f"   Numeric features: {X.shape[1] - len(cat_cols)}")
    
    # Fill NaN values with 0 (but handle categorical separately)
    print("\n   Filling NaN values...")
    nan_counts = X.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"   Found {total_nans:,} NaN values across {(nan_counts > 0).sum()} columns")

        # Show which columns have NaNs
        nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
        print("   Columns with NaN values:")
        for col, count in nan_cols.items():
            pct = (count / len(X)) * 100
            print(f"   - {col:<50} {count:>8,} NaNs ({pct:>5.1f}%)")

        # Fill numeric columns with 0
        numeric_cols = X.select_dtypes(include=['number']).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)

        # Fill categorical columns with 'UNKNOWN' and add to categories
        for col in cat_cols:
            if X[col].isna().any():
                X[col] = X[col].cat.add_categories(['UNKNOWN']).fillna('UNKNOWN')
                cat_mappings[col] = X[col].cat.categories.tolist()
    else:
        print("   No NaN values found")
    
    # DIAGNOSTICS: Post-cleaning check on feature matrix X vs actual target fuel_kg
    print("\n   Starting post-cleaning diagnostics...")
    def diagnose_leakage_post(X_df, merged_df):
        print('\nDIAGNOSTICS: Leakage and covariance checks (post-cleaning)')
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print('   No numeric columns available in X for leakage check.')
            return

        # Check for zero-variance features (constant)
        variances = X_df[numeric_cols].var()
        zero_var_cols = variances[variances == 0].index.tolist()
        if zero_var_cols:
            print(f'   Found {len(zero_var_cols)} zero-variance features (constant across rows):')
            print('   - ' + '\n   - '.join(zero_var_cols[:20]))
        else:
            print('   No zero-variance features found in X.')

        # Pearson correlations
        print('   Computing correlations...')
        corr_series = X_df[numeric_cols].corrwith(merged_df['fuel_kg']).abs().sort_values(ascending=False)
        print('\n   Top correlations (post-cleaning) with fuel_kg:')
        print(corr_series.head(20).to_string())

        # Covariance - limit to top 100 correlated for speed
        print('   Computing covariances...')
        top_corr_cols = corr_series.head(100).index.tolist()
        covs = {}
        y = merged_df['fuel_kg'].values
        for col in top_corr_cols:
            x = X_df[col].values
            try:
                covs[col] = np.cov(x, y, rowvar=False)[0, 1]
            except Exception:
                covs[col] = np.nan
        covs_series = pd.Series(covs).abs().sort_values(ascending=False)
        print('\n   Top absolute covariances (post-cleaning) with fuel_kg:')
        print(covs_series.head(20).to_string())

        # Perfect equality - skip for speed
        print('\n   Skipping perfect equality check for speed.')

        # High correlation candidates
        high_corrs = corr_series[corr_series > 0.95].index.tolist()
        if high_corrs:
            print('\n   High-correlation candidates (abs corr > 0.95) post-cleaning:')
            for col in high_corrs:
                print(f"   - {col}: corr={corr_series[col]:.6f}, cov={covs_series.get(col, np.nan):.6f}")
        else:
            print('\n   No very-high-correlation features found post-cleaning.')

        # Check for duplicates or repeated columns - skip for speed
        print('   Skipping duplicate column check for speed.')

    try:
        diagnose_leakage_post(X, merged_df)
        print("   Post-cleaning diagnostics completed.")
    except Exception as ex:
        print('   Diagnostic post-cleaning check failed:', str(ex))

    # Filter out leakage features by name patterns (conservative approach)
    leakage_patterns = [

        
    ]
    leak_cols = [c for c in X.columns if any(patt in c for patt in leakage_patterns)]
    if leak_cols:
        print(f"   Dropping {len(leak_cols)} fuel-leakage-like columns (by name patterns)")
        print('   - ' + '\n   - '.join(leak_cols[:30]))
        X = X.drop(columns=leak_cols, errors='ignore')

    # Drop zero-variance features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    variances = X[numeric_cols].var()
    zero_var_cols = variances[variances == 0].index.tolist()
    if zero_var_cols:
        print(f"   Dropping {len(zero_var_cols)} zero-variance features")
        X = X.drop(columns=zero_var_cols, errors='ignore')

    # Drop duplicate columns (identical series)
    # This can be slow for many columns, commenting out to speed up
    # numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # to_drop_dups = []
    # seen = {}
    # for col in numeric_cols:
    #     key = pd.util.hash_pandas_object(X[col].fillna(0)).sum()
    #     if key in seen:
    #         # keep the first, drop duplicates
    #         to_drop_dups.append(col)
    #     else:
    #         seen[key] = col
    # if to_drop_dups:
    #     print(f"   Dropping {len(to_drop_dups)} duplicate columns (identical values)")
    #     print('   - ' + '\n   - '.join(to_drop_dups[:30]))
    #     X = X.drop(columns=to_drop_dups, errors='ignore')

    print(f"   Feature matrix after leakage/variance/duplicate pruning: {X.shape}")
        
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 3000,  # Reduced for faster training
        'learning_rate': 0.1,
        'num_leaves': 50,
        'max_depth': 8,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': 1,  # Enable verbose output
        'n_jobs': -1,
        #'device': 'gpu',
        #'max_bin': 255,
    }
    
    # Get monotone constraints
    monotone_constraints = get_monotone_constraints(X.columns)
    n_monotone = sum(1 for c in monotone_constraints if c != 0)
    print(f"   Monotone constraints applied to {n_monotone} features")
    params['monotone_constraints'] = monotone_constraints
        
    # 5. Train Final Model and Analyze Feature Importance
    print("\n5. Training final model on full dataset...")
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y, categorical_feature=cat_cols)
    
    print("\n   Analyzing feature importance...")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_,
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(feature_importance_path, index=False)
    print(f"   ✓ Saved feature importance to {feature_importance_path}")
    
    print("\n   Top 20 Features by Importance:")
    print(importance_df.head(20)[['feature', 'importance']].to_string(index=False))
    
    n_zero_importance = (importance_df['importance'] == 0).sum()
    print(f"\n   Features with zero importance: {n_zero_importance} / {len(importance_df)}")
    
    # 6. Predict on Rank Data
    print("\n9. Generating predictions for rank dataset...")
    
    X_rank = rank_df.drop(columns=drop_cols, errors='ignore')
    
    # Apply categorical mappings - NEW: Allow all categories instead of restricting to training ones
    for col in cat_cols:
        if col in X_rank.columns:
            X_rank[col] = pd.Categorical(X_rank[col].astype(str))
            # Note: Not restricting to training categories - keeps original route names for better prediction
    
    # Align columns
    missing_cols = set(X.columns) - set(X_rank.columns)
    for col in missing_cols:
        X_rank[col] = 0
    
    X_rank = X_rank[X.columns]
    
    # Fill NaN values (handle categorical separately)
    print(f"   Filling NaN values in rank data...")
    rank_nan_count = X_rank.isna().sum().sum()
    if rank_nan_count > 0:
        print(f"   Found {rank_nan_count} NaN values")
        
        # Fill numeric columns with 0
        numeric_cols = X_rank.select_dtypes(include=['number']).columns
        X_rank[numeric_cols] = X_rank[numeric_cols].fillna(0)
        
        # Note: No longer filling categorical NaNs with 'UNKNOWN' - keeping original category names
    else:
        print("   No NaN values found")
    
    print(f"   Rank feature matrix shape: {X_rank.shape}")
    
    # Predict scaled fuel
    rank_preds_scaled = final_model.predict(X_rank)
    
    # Convert back to actual fuel
    # Compute rank_mass_above_oew if not already present
    if 'mass_above_oew' in rank_df.columns:
        rank_mass_above_oew = rank_df['mass_above_oew']
    else:
        mtow_col_rank = 'ac_max_takeoff_weight_kg' if 'ac_max_takeoff_weight_kg' in rank_df.columns else ('ac_MTOW_kg' if 'ac_MTOW_kg' in rank_df.columns else ('ac_MTOW_lb' if 'ac_MTOW_lb' in rank_df.columns else None))
        if mtow_col_rank == 'ac_MTOW_lb':
            rank_df['ac_MTOW_kg'] = rank_df['ac_MTOW_lb'] / 2.2046226218
            mtow_col_rank = 'ac_MTOW_kg'
        if mtow_col_rank is None or 'ac_operating_empty_weight_kg' not in rank_df.columns:
            raise ValueError('MTOW / OEW not found in rank_df to convert back predictions to kg')
        rank_mass_above_oew = (rank_df[mtow_col_rank] - rank_df['ac_operating_empty_weight_kg']).clip(lower=1.0)
    rank_preds = (rank_preds_scaled / scaling_factor) * rank_mass_above_oew
    rank_preds = np.maximum(rank_preds, 0.1)  # Ensure no negative
    
    print(f"   Predictions - Mean: {rank_preds.mean():.2f}, Std: {rank_preds.std():.2f}")
    print(f"   Predictions - Min: {rank_preds.min():.2f}, Max: {rank_preds.max():.2f}")
    
    # 10. Save Submission
    print("\n10. Saving submission...")
    
    if not submission_template_path.exists():
        print("   Error: Submission template not found!")
        return
    
    submission_df = pd.read_parquet(submission_template_path)
    print(f"   Submission template shape: {submission_df.shape}")
    
    # Map predictions
    pred_series = pd.Series(rank_preds, index=rank_df['idx'])
    submission_df['fuel_kg'] = submission_df['idx'].map(pred_series)
    
    # Fill missing
    missing_preds = submission_df['fuel_kg'].isna().sum()
    if missing_preds > 0:
        print(f"   Warning: {missing_preds} rows missing. Filling with median: {np.median(rank_preds):.2f}")
        submission_df['fuel_kg'] = submission_df['fuel_kg'].fillna(np.median(rank_preds))
    
    # Validation
    assert submission_df['fuel_kg'].notna().all(), "ERROR: Submission contains NaN!"
    assert (submission_df['fuel_kg'] >= 0).all(), "ERROR: Submission contains negative values!"
    
    # Save
    submission_df.to_parquet(output_path)
    print(f"   ✓ Saved submission to {output_path}")
    print(f"   Submission shape: {submission_df.shape}")
    print(f"\n   First few predictions:")
    print(submission_df.head(10))

    # ADDITIONAL DIAGNOSTICS: Check prediction variance per flight_id to detect constant predictions
    try:
        # Attach predictions back to rank_df
        rank_df['fuel_pred_kg'] = rank_df['idx'].map(pred_series)
        # Fill any NaN preds if they exist with median
        rank_df['fuel_pred_kg'] = rank_df['fuel_pred_kg'].fillna(np.median(rank_preds))

        # Compute per-flight stats
        flight_stats = rank_df.groupby('flight_id')['fuel_pred_kg'].agg(['count', 'mean', 'std'])
        flight_stats['std'] = flight_stats['std'].fillna(0.0)
        # Count flights with zero variance (std == 0)
        zero_var_flights = flight_stats[flight_stats['std'] == 0.0]
        low_var_flights = flight_stats[flight_stats['std'] / (flight_stats['mean'] + 1e-9) < 0.01]

        print('\nDIAGNOSTICS: Prediction variance per flight_id')
        print(f"   Total flights in rank data: {len(flight_stats)}")
        print(f"   Flights with zero std predictions: {len(zero_var_flights)}")
        print(f"   Flights with low std (<1% mean): {len(low_var_flights)}")

        if len(zero_var_flights) > 0:
            print('   Example flights with zero variance predictions:')
            example_zero_flights = zero_var_flights.head(5).index.tolist()
            for fid in example_zero_flights:
                mask = rank_df['flight_id'] == fid
                print(f"   - flight_id: {fid}, count: {mask.sum()}, pred: {rank_df.loc[mask, 'fuel_pred_kg'].iloc[0]:.3f}")
                # Show top features variability for that flight
                top_feats = list(importance_df['feature'].head(10).values)
                var_df = rank_df.loc[mask, top_feats].describe().T
                print(var_df[['mean', 'std']].head(10).to_string())
        else:
            print('   No zero-variance flights found.')

        # Check if any predictions are identical across most rows
        unique_counts = pd.Series(rank_preds).value_counts().head(10)
        print('\nDIAGNOSTICS: Top repeated prediction values (count)')
        print(unique_counts.to_string())

    except Exception as ex:
        print('   Prediction variance diagnostics failed:', str(ex))

    # Show how many submission rows were model-predicted vs median-filled
    try:
        submission_df['is_model_pred'] = submission_df['idx'].isin(pred_series.index)
        n_model_pred = submission_df['is_model_pred'].sum()
        n_filled = (~submission_df['is_model_pred']).sum()
        print('\nDIAGNOSTICS: Submission mapping')
        print(f"   Submission rows: {len(submission_df)}; Model-predicted rows: {n_model_pred}; Median-filled rows: {n_filled}")
        if n_filled > 0:
            print('   Example missing idxs (first 20):')
            print(submission_df[~submission_df['is_model_pred']]['idx'].head(20).tolist())
    except Exception as ex:
        print('   Submission mapping diagnostics failed:', str(ex))
    
    # 11. Save Metadata
    print("\n11. Saving metadata...")
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'train_features_path': str(train_features_path),
        'rank_features_path': str(rank_features_path),
        'train_shape': list(train_df.shape),
        'rank_shape': list(rank_df.shape),
        'final_shape': list(final_df.shape) if final_df is not None else None,
        'final_train_shape': list(merged_df.shape),
        'n_features': X.shape[1],
        'n_categorical': len(cat_cols),
        'model_params': params,
        'target_type': f'scaled fuel_kg: {scaling_factor} * fuel_kg / (ac_max_takeoff_weight_kg - ac_operating_empty_weight_kg)',
        'mtow_column_used': mtow_used if 'mtow_used' in locals() else None,
        'features_zero_importance': int(n_zero_importance),
        'predictions_mean': float(rank_preds.mean()),
        'predictions_std': float(rank_preds.std()),
        'predictions_min': float(rank_preds.min()),
        'predictions_max': float(rank_preds.max()),
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved metadata to {metadata_path}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nVersion: {version}")
    print(f"Output: {output_path}")
    print(f"Feature Importance: {feature_importance_path}")
    print(f"Metadata: {metadata_path}")

if __name__ == "__main__":
    main()
