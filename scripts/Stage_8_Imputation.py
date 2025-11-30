import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress the "Mean of empty slice" warning
warnings.filterwarnings("ignore", message="Mean of empty slice")

# Paths
base_path = Path("F:/PRC_2025/Likable-ant_v1/data/processed/Stage_8_Consolidated_Features")
logs_path = Path("F:/PRC_2025/Likable-ant_v1/logs/feature_importances")
logs_path.mkdir(parents=True, exist_ok=True)
checkpoint_path = base_path / "imputation_checkpoint.parquet"

# Function for fast weighted group imputation (primary method)
def weighted_group_impute(df, target_col, group_cols, weight_col='interval_duration_sec', N_min=5, bounds=None, use_median=True):
    """Impute using weighted median/mean per group with outlier-safe clipping.
    bounds: dict mapping column->(low, high) or None.
    use_median: True for median, False for weighted mean
    """
    def weighted_median(values, weights):
        if len(values) == 0 or weights.sum() == 0:
            return np.nan
        sorted_idx = np.argsort(values)
        values = values[sorted_idx]
        weights = weights[sorted_idx]
        cum_weights = np.cumsum(weights)
        cutoff = weights.sum() / 2
        idx = np.searchsorted(cum_weights, cutoff)
        return values[idx] if idx < len(values) else values[-1]
    
    def weighted_mean(values, weights):
        if len(values) == 0 or weights.sum() == 0:
            return np.nan
        return np.average(values, weights=weights)
    
    impute_func = weighted_median if use_median else weighted_mean
    
    imputed = df[target_col].copy()
    low, high = (None, None)
    if bounds is not None and target_col in bounds:
        low, high = bounds[target_col]
    
    # Debug counters
    filled_full_group = 0
    filled_drop_position = 0
    filled_aircraft_only = 0
    filled_global = 0
    # Fallback group patterns to widen the group progressively: first full group, then aircraft only
    def fallback_group_masks(df, group_values):
        # group_values is a tuple corresponding to group_cols
        masks = []
        # Full group
        mask_full = np.ones(len(df), dtype=bool)
        for col, val in zip(group_cols, group_values):
            if pd.isna(val):
                mask_full &= df[col].isna()
            else:
                mask_full &= (df[col] == val)
        masks.append(mask_full)

        # Aircraft only fallback
        if 'aircraft_type' in group_cols:
            mask_aircraft = (df['aircraft_type'] == group_values[group_cols.index('aircraft_type')])
            masks.append(mask_aircraft)

        return masks

    for group, group_df in df.groupby(group_cols, observed=False):
        mask = group_df[target_col].isna()
        if mask.any():
            non_na = group_df[~mask]
            # Check if we have enough rows and non-zero weights
            if len(non_na) >= N_min and non_na[weight_col].fillna(1).sum() > 0:
                vals = non_na[target_col].astype(float).values
                # Garbage-in protection: clip to global bounds if available
                if low is not None and high is not None:
                    vals = np.clip(vals, low, high)
                weights = non_na[weight_col].replace(0, 1).fillna(1)  # Sanitize weights: replace zeros/NaNs with 1
                imputed.loc[group_df.index[mask]] = impute_func(vals, weights.values)
                filled_full_group += mask.sum()
            else:
                # Try hierarchical fallbacks
                group_values = group if isinstance(group, tuple) else (group,)
                masks = fallback_group_masks(df, group_values)
                filled = False
                for i, fb_mask in enumerate(masks[1:], 1):  # skip the first one because it's the current group
                    fb_non_na = df[fb_mask & df[target_col].notna()]
                    if len(fb_non_na) >= N_min:
                        weights = fb_non_na[weight_col].fillna(1)
                        imputed.loc[group_df.index[mask]] = impute_func(fb_non_na[target_col].values, weights.values)
                        filled = True
                        if i == 1:
                            filled_aircraft_only += mask.sum()
                        break
                if not filled:
                    # Global median fallback
                    finite_values = df[target_col][np.isfinite(df[target_col])]
                    if len(finite_values) > 0:
                        global_val = finite_values.median() if use_median else np.average(finite_values, weights=df.loc[finite_values.index, weight_col].fillna(1))
                        imputed.loc[group_df.index[mask]] = global_val
                        filled_global += mask.sum()
                    else:
                        imputed.loc[group_df.index[mask]] = 0
    # Garbage-out protection: clip final column
    if low is not None and high is not None:
        imputed = imputed.clip(lower=low, upper=high)
    
    # Print debug summary for this column
    total_filled = filled_full_group + filled_aircraft_only + filled_global
    if total_filled > 0:
        print(f"  {target_col}: Filled {total_filled} NaNs - Full group: {filled_full_group}, Aircraft only: {filled_aircraft_only}, Global: {filled_global}")
    
    return imputed


def compute_outlier_bounds(series, lower_q=0.01, upper_q=0.99, min_n=100):
    """Compute robust bounds for a numeric series using quantiles.
    Returns (low, high) or (None, None) if not enough data or invalid.
    """
    s = series.dropna().values
    if s.size < min_n:
        return None, None
    low = np.quantile(s, lower_q)
    high = np.quantile(s, upper_q)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return None, None
    return float(low), float(high)


def median_group_impute(df, target_col, group_cols, bounds=None):
    """Impute using outlier-robust group medians with fallback to global median."""
    low, high = (None, None)
    if bounds is not None and target_col in bounds:
        low, high = bounds[target_col]

    global_series = df[target_col]
    finite_vals = global_series[np.isfinite(global_series)].dropna()
    global_median = finite_vals.median() if len(finite_vals) > 0 else np.nan

    def fill_group(x):
        x_clip = x
        if low is not None and high is not None:
            x_clip = x_clip.clip(lower=low, upper=high)
        grp_median = x_clip.median()
        if np.isnan(grp_median):
            grp_median = global_median
        return x.fillna(grp_median)

    out = df.groupby(group_cols, observed=False)[target_col].transform(fill_group)
    if low is not None and high is not None:
        out = out.clip(lower=low, upper=high)
    return out

# New: KNN-based imputation for non-time-dependent features using KDTree for speed
# KNN removed: prefer group-based methods only

# Flight-wise interpolation for time/position sequential features
def flightwise_interpolate(df, target_cols, flight_col='flight_id', sort_col='interval_rel_position_midpoint'):
    """Perform linear interpolation within a flight for target_cols. Only fills interior NaNs (limit_area='inside')."""
    df = df.copy()
    if flight_col not in df.columns:
        print(f"Flight column {flight_col} not found; skipping flightwise interpolation")
        return df

    for col in target_cols:
        if col not in df.columns:
            continue
        # For each flight, sort by sort_col and interpolate only internal NaNs
        def interp_group(g):
            if sort_col in g.columns:
                g = g.sort_values(sort_col)
            g[col] = g[col].interpolate(method='linear', limit_area='inside')
            return g
        df = df.groupby(flight_col, group_keys=False).apply(interp_group)
    return df

# Function for LightGBM imputation (secondary, for critical features)
# LightGBM removed: we keep only group-based imputation

# Global LightGBM imputation for multiple features at once
# Global LGBM removed. We'll only use group-based imputation methods

# Function to recalculate derived features
def recalculate_derived_features(df):
    """Recalculate features derived from base ones."""
    # Fuel ratios and diffs
    if 'int_fuel_consumed_acropole_kg' in df and 'int_fuel_consumed_openap_kg' in df:
        df['int_fuel_ratio_acropole_openap'] = df['int_fuel_consumed_acropole_kg'] / df['int_fuel_consumed_openap_kg'].replace(0, np.nan)
        df['int_fuel_diff_acropole_openap_kg'] = df['int_fuel_consumed_acropole_kg'] - df['int_fuel_consumed_openap_kg']
    
    # Mass ratios
    if 'int_mass_start_kg' in df and 'int_mass_openap_start_kg' in df:
        df['int_mass_ratio_acropole_openap'] = df['int_mass_start_kg'] / df['int_mass_openap_start_kg'].replace(0, np.nan)
    
    # Thrust-drag net force
    if 'int_thrust_openap_mean' in df and 'int_drag_openap_mean' in df:
        df['int_net_force_openap_n'] = df['int_thrust_openap_mean'] - df['int_drag_openap_mean']
    
    # Power calculations
    if 'int_thrust_power' in df and 'int_drag_power' in df:
        df['int_net_power'] = df['int_thrust_power'] - df['int_drag_power']
    
    # Energy changes
    if 'int_dPE_MJ' in df and 'int_dKE_MJ' in df:
        df['int_total_energy_change_MJ'] = df['int_dPE_MJ'] + df['int_dKE_MJ']
    
    # Higher-order transforms (squares)
    for base in ['int_tas_mean', 'int_mass_mean', 'int_fuel_flow_acropole_mean']:
        if base in df:
            df[f'{base}_sq'] = df[base] ** 2
    
    # Add more as needed
    return df


def recalculate_categorical_features(df):
    """Recalculate categorical features derived from imputed numeric features."""
    
    # Recalculate int_phase_mode from phase fractions
    phase_cols = ['int_frac_climb', 'int_frac_cruise', 'int_frac_descent', 'int_frac_ground']
    if all(col in df.columns for col in phase_cols):
        mask = df['int_phase_mode'].isna()
        if mask.any():
            print(f"Recalculating int_phase_mode for {mask.sum()} rows...")
            for idx in df[mask].index:
                fracs = {
                    'climb': df.loc[idx, 'int_frac_climb'] or 0,
                    'cruise': df.loc[idx, 'int_frac_cruise'] or 0,
                    'descent': df.loc[idx, 'int_frac_descent'] or 0,
                    'ground': df.loc[idx, 'int_frac_ground'] or 0
                }
                if sum(fracs.values()) > 0:
                    df.loc[idx, 'int_phase_mode'] = max(fracs, key=fracs.get)
                else:
                    df.loc[idx, 'int_phase_mode'] = 'unknown'
    
    # For int_contains_tod and int_contains_toc, since we can't recalculate without raw trajectory,
    # set to False if NaN (assuming no transition detected)
    for col in ['int_contains_tod', 'int_contains_toc']:
        if col in df.columns:
            mask = df[col].isna()
            if mask.any():
                print(f"Setting {col} to False for {mask.sum()} rows (no trajectory data to detect transitions)")
                df.loc[mask, col] = False
    
    return df

def enforce_monotonic_constraints(df, flight_col='flight_id', sort_col='interval_rel_position_midpoint'):
    """Enforce monotonicity and non-negativity for cumulative and consumed variables."""
    df = df.copy()
    
    # Cumulative variables that must be non-decreasing
    cumulative_cols = ['int_cum_fuel_acropole_kg', 'int_cum_fuel_openap_kg', 'int_ground_distance_nm']
    
    # Consumed variables that must be >= 0
    consumed_cols = ['int_fuel_consumed_acropole_kg', 'int_fuel_consumed_openap_kg', 'int_ground_distance_nm']
    
    for col in cumulative_cols:
        if col in df.columns:
            print(f"Enforcing non-decreasing for {col}...")
            def enforce_cumulative(g):
                if sort_col in g.columns:
                    g = g.sort_values(sort_col)
                if g[col].notna().any():
                    g[col] = np.maximum.accumulate(g[col].fillna(method='ffill').fillna(0))
                return g
            df = df.groupby(flight_col, group_keys=False).apply(enforce_cumulative)
    
    for col in consumed_cols:
        if col in df.columns:
            print(f"Enforcing non-negative for {col}...")
            df[col] = df[col].clip(lower=0)
    
    return df

if __name__ == '__main__':
    # Example column for quick debug/insight
    example_col = 'int_fuel_consumed_acropole_kg'
    # Check for checkpoint
    if checkpoint_path.exists():
        print("Checkpoint found, loading from checkpoint...")
        try:
            df = pd.read_parquet(checkpoint_path)
            print(f"Loaded checkpoint with shape: {df.shape}")
            # Assume checkpoint is after KNN, skip to LightGBM
            skip_to_lightgbm = False
        except Exception as e:
            print(f"Checkpoint file corrupted: {e}. Deleting and starting from scratch.")
            checkpoint_path.unlink()
            skip_to_lightgbm = False
        
        # Convert any infinite values to NaN immediately after loading
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        group_cols = ['aircraft_type', 'position_bin', 'phase_bin']
        
        # Define variables for LightGBM
        predictors = [
            'interval_rel_position_midpoint', 'time_since_takeoff_min', 'origin_dest_gcdist_nm',
            'tow_pred_kg', 'total_flight_duration_sec', 'aircraft_type'
        ]
        numeric_predictors = [p for p in predictors if p != 'aircraft_type']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        protected_fuel_mass_cols = [
            'int_fuel_consumed_acropole_kg',
            'int_fuel_consumed_openap_kg',
            'int_fuel_flow_acropole_mean',
            'int_fuel_flow_openap_mean',
            'int_mass_mean',
            'int_mass_openap_mean',
            # add any other core fuel/mass int_ features handled by your V2 pipeline
        ]
        features_to_impute = [
            col for col in numeric_cols
            if col not in predictors
            and col not in protected_fuel_mass_cols
            and df[col].isna().sum() > 0
            and df[col].isna().mean() < 0.5
        ]
        
        # If needed, a machine-learning targeted list of important features could be selected here, but we skip ML-based imputation.
        
        # Define duration-weighted features (cumulative/time-dependent)
        time_dependent_keywords = ['consumed', 'total', 'change', 'distance', 'fuel_per', 'MJ_per', 'cum_', 'time_since', 'duration']
        duration_weighted_features = [col for col in features_to_impute if any(kw in col for kw in time_dependent_keywords)]
        
        # Non-duration-weighted features (rest, including LightGBM ones)
        non_duration_weighted_features = [col for col in features_to_impute if col not in duration_weighted_features]
        
        if example_col in duration_weighted_features:
            print(f"{example_col} is in duration_weighted_features")
        elif example_col in non_duration_weighted_features:
            print(f"{example_col} is in non_duration_weighted_features")
        else:
            print(f"{example_col} is not in either imputation list")
    else:
        skip_to_lightgbm = False
        # Load and combine datasets (unchanged)
        print("Loading datasets...")
        train_df = pd.read_parquet(base_path / "consolidated_features_train_multiplier.parquet")
        rank_df = pd.read_parquet(base_path / "consolidated_features_rank_multiplier.parquet")
        final_df = pd.read_parquet(base_path / "consolidated_features_final_multiplier.parquet")
        train_df['dataset_split'] = 'train'
        rank_df['dataset_split'] = 'rank'
        final_df['dataset_split'] = 'final'
        df = pd.concat([train_df, rank_df, final_df], ignore_index=True)
        print(f"Combined dataset shape: {df.shape}")

        # Convert any infinite values to NaN immediately after loading
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Predictors (core features, ensure complete)
        predictors = [
            'interval_rel_position_midpoint', 'time_since_takeoff_min', 'origin_dest_gcdist_nm',
            'tow_pred_kg', 'total_flight_duration_sec', 'aircraft_type'
        ]
        numeric_predictors = [p for p in predictors if p != 'aircraft_type']
        for col in numeric_predictors:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Adjustable bin sizes (experiment with smaller values for accuracy)
        position_bins = np.arange(0, 1.1, 0.1)  # 10 bins; try 0.05 for 20 bins
        
        df['position_bin'] = pd.cut(df['interval_rel_position_midpoint'], bins=position_bins, labels=False)
        # Phase binning: use only for intervals with trajectory data
        phase_bins = [-0.01, 0.1, 0.6, 1.01]
        df['phase_bin'] = np.where(df['has_trajectory_data'] == True, 
                                   pd.cut(df['int_frac_cruise'], bins=phase_bins, labels=[0, 1, 2], include_lowest=True), 
                                   np.nan)
        group_cols = ['aircraft_type', 'position_bin', 'phase_bin']  # Use phase_bin when available

        # Encode and defragment
        le = LabelEncoder()
        df['aircraft_type_encoded'] = le.fit_transform(df['aircraft_type'])
        df = df.copy()

        protected_fuel_mass_cols = [
            'int_fuel_consumed_acropole_kg',
            'int_fuel_consumed_openap_kg',
            'int_fuel_flow_acropole_mean',
            'int_fuel_flow_openap_mean',
            'int_mass_mean',
            'int_mass_openap_mean',
            # add any other core fuel/mass int_ features handled by your V2 pipeline
        ]

        # Auto-select features for imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features_to_impute = [
            col for col in numeric_cols
            if col not in predictors
            and col not in protected_fuel_mass_cols
            and df[col].isna().sum() > 0
            and df[col].isna().mean() < 0.5
        ]
        
        # Debug for example feature
        example_col = 'int_fuel_consumed_acropole_kg'
        print(f"Combined df shape: {df.shape}")
        if example_col in df.columns:
            print(f"{example_col} NaNs: {df[example_col].isna().sum()} out of {len(df)} ({df[example_col].isna().mean():.4f})")
            if example_col in features_to_impute:
                print(f"{example_col} is in features_to_impute")
            else:
                print(f"{example_col} is not in features_to_impute")
        
        # Print excluded features
        excluded_features = [col for col in numeric_cols if col not in predictors and (df[col].isna().sum() == 0 or df[col].isna().mean() >= 0.5)]
        print(f"Excluded features (no NaNs or >=50% NaNs): {len(excluded_features)}")
        if excluded_features:
            print("Sample excluded features:", excluded_features[:10])

        # Categorize features
        time_dependent_keywords = ['consumed', 'total', 'change', 'distance', 'fuel_per', 'MJ_per', 'cum_', 'time_since', 'duration']
        duration_weighted_features = [col for col in features_to_impute if any(kw in col for kw in time_dependent_keywords)]
        
        # Flight-level statistics disguised as int_ features (organizational note for modeling)
        # These are constant per flight and should be considered for flight-level models, not interval variation
        flight_level_int_features = [
            'int_frac_climb', 'int_frac_cruise', 'int_frac_descent', 'int_frac_ground',
            'int_num_climb', 'int_num_cruise', 'int_num_descent', 'int_num_ground',
            'int_day_of_week', 'int_phase_mode', 'int_contains_tod', 'int_contains_toc'
        ]
        
        # LightGBM-based feature selection removed. We keep only group-based imputation strategies.
        
        # Define duration-weighted features (cumulative/time-dependent)
        duration_weighted_features = [col for col in features_to_impute if any(kw in col for kw in time_dependent_keywords)]
        
        # Non-duration-weighted features (rest, including LightGBM ones)
        non_duration_weighted_features = [col for col in features_to_impute if col not in duration_weighted_features]

    # Optional: flight-wise interpolation (no KNN/machine-learning tiers used in production)
    # Interpolation targets: smooth-over-time features
    interpolation_candidates = [
        'int_fuel_consumed_acropole_kg', 'int_fuel_consumed_openap_kg', 'int_ground_distance_nm',
        'int_alongtrack_distance_nm', 'int_crosstrack_distance_nm', 'int_mass_mean', 'int_mass_openap_mean',
        'int_tas_mean', 'int_mach_mean', 'int_vertical_rate_mean', 'int_altitude_mean',
        'int_thrust_openap_mean', 'int_drag_openap_mean'
    ]
    interpolation_features = [col for col in interpolation_candidates if col in df.columns and col in features_to_impute]
    if interpolation_features:
        print(f"Flightwise interpolating {len(interpolation_features)} features: {interpolation_features}")
        df = flightwise_interpolate(df, interpolation_features, flight_col='flight_id', sort_col='interval_rel_position_midpoint')

    # KNN imputation candidates and predictors
    # KNN imputation removed: rely only on group-based methods (duration-weighted and median). If needed, add KNN here later.

    # Unified imputation
    # Convert any infinite values to NaN to prevent issues and compute feature-specific bounds
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Compute robust global bounds for each feature_to_impute
    feature_bounds = {}
    for col in features_to_impute:
        low, high = compute_outlier_bounds(df[col], lower_q=0.01, upper_q=0.99, min_n=100)
        feature_bounds[col] = (low, high)

    print("Imputing features using group-based methods...")
    print(f"NaN counts before imputation in features_to_impute: {df[features_to_impute].isnull().sum().sum()}")
    group_cols = ['aircraft_type', 'position_bin', 'phase_bin']
    
    # Duration-weighted imputation (weighted mean or median for heavy-tailed)
    heavy_tailed_features = [
        'int_fuel_consumed_acropole_kg', 'int_fuel_consumed_openap_kg',
        'int_cum_fuel_acropole_kg', 'int_cum_fuel_openap_kg',
        'int_dPE_MJ', 'int_dKE_MJ', 'int_total_energy_change_MJ',
        'int_thrust_power', 'int_drag_power', 'int_net_power'
    ]
    if duration_weighted_features:
        print(f"Imputing {len(duration_weighted_features)} duration-weighted features...")
        for col in tqdm(duration_weighted_features, desc="Duration-Weighted Imputing"):
            use_median = col in heavy_tailed_features
            df[col] = weighted_group_impute(df, col, group_cols, weight_col='interval_duration_sec', bounds=feature_bounds, use_median=use_median)
    
    # Non-duration-weighted imputation (simple group median)
    if non_duration_weighted_features:
        print(f"Imputing {len(non_duration_weighted_features)} non-duration-weighted features...")
        for col in tqdm(non_duration_weighted_features, desc="Median Imputing"):
            df[col] = median_group_impute(df, col, group_cols, bounds=feature_bounds)
    
    print("Unified imputation completed.")
    print(f"NaN counts after imputation in features_to_impute: {df[features_to_impute].isnull().sum().sum()}")
    print(f"After imputation, {example_col} NaNs: {df[example_col].isna().sum()}")
    
    # Diagnostics - top offending features
    remaining_by_feature = df[features_to_impute].isna().sum().sort_values(ascending=False)
    print("Top 20 features still NaN after first pass:")
    print(remaining_by_feature.head(20))
    
    # Ensure bin columns have explicit unknown values to improve fallbacks
    if 'position_bin' in df.columns:
        df['position_bin'] = df['position_bin'].fillna(-1)
    
    # Aircraft-only weighted fallback with small N_min
    to_fix = [
        c for c in features_to_impute
        if df[c].isna().sum() > 0 and not c.startswith('int_')
    ]
    if to_fix:
        print(f"Applying aircraft-only fallback to {len(to_fix)} features")
        for col in tqdm(to_fix, desc="Aircraft-only fallback"):
            use_median = col not in duration_weighted_features
            df[col] = weighted_group_impute(
                df,
                col,
                ['aircraft_type'],
                weight_col='interval_duration_sec',
                N_min=1,
                bounds=feature_bounds,
                use_median=use_median,
            )
    
    # Final per-aircraft / per-split / global median fill
    to_fix = [
        c for c in features_to_impute
        if df[c].isna().sum() > 0 and not c.startswith('int_')
    ]
    for col in tqdm(to_fix, desc="Final median filling"):
        df[col] = df[col].fillna(df.groupby('aircraft_type')[col].transform('median'))
        df[col] = df[col].fillna(df.groupby('dataset_split')[col].transform('median'))
        df[col] = df[col].fillna(df[col].median())
    
    # Optional: impute high-missing features if user opts in (toggle)
    IMPUTE_HIGH_MISSING = False
    if IMPUTE_HIGH_MISSING:
        high_missing = [
            col
            for col in numeric_cols
            if col not in predictors and df[col].isna().sum() > 0 and df[col].isna().mean() >= 0.5
        ]
        if high_missing:
            print(f"Imputing {len(high_missing)} high-missing features with coarse per-aircraft medians")
            for col in tqdm(high_missing, desc="High-missing impute"):
                df[col] = df[col].fillna(df.groupby('aircraft_type')[col].transform('median'))
                df[col] = df[col].fillna(df[col].median())
    
    # Print diagnostics after fallback & median fills
    remaining_by_feature_after = df[features_to_impute].isna().sum().sort_values(ascending=False)
    print("Top 20 features still NaN after fallback and median fills:")
    print(remaining_by_feature_after.head(20))
    print("Total NaNs left in features_to_impute:", remaining_by_feature_after.sum())
    
    # Enforce monotonic constraints for cumulative and consumed variables
    print("Enforcing monotonic and non-negativity constraints...")
    df = enforce_monotonic_constraints(df, flight_col='flight_id', sort_col='interval_rel_position_midpoint')
    
    # Save checkpoint if needed
    df.to_parquet(checkpoint_path, index=False)



    # Recalculate derived features
    print("Recalculating derived features...")
    df = recalculate_derived_features(df)
    
    # Recalculate categorical features from imputed numeric ones
    print("Recalculating categorical features...")
    df = recalculate_categorical_features(df)
    
    df = df.copy()  # Defragment DataFrame to reduce fragmentation warnings
    
    # Final machine-learning passes (LightGBM/KNN) are disabled in production; rely on group-based imputation.

    # Validation and save (unchanged)
    remaining_nans = df.isnull().sum().sum()
    print(f"Remaining NaNs after imputation: {remaining_nans}")

    train_imputed = df[df['dataset_split'] == 'train'].drop(columns=['dataset_split', 'position_bin'])
    rank_imputed = df[df['dataset_split'] == 'rank'].drop(columns=['dataset_split', 'position_bin'])
    final_imputed = df[df['dataset_split'] == 'final'].drop(columns=['dataset_split', 'position_bin'])

    train_imputed.to_parquet(base_path / "consolidated_features_train_imputed.parquet", index=False)
    rank_imputed.to_parquet(base_path / "consolidated_features_rank_imputed.parquet", index=False)
    final_imputed.to_parquet(base_path / "consolidated_features_final_imputed.parquet", index=False)

    # Remove checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Removed checkpoint file.")

    print("Imputation complete. Saved all datasets.")