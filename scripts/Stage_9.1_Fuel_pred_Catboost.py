"""
Script to train CatBoost model with fixed hyperparameters (from HPO).
Predicts scaled fuel consumption, clips negatives to 1kg.
Includes:
- Physics-based feature engineering
- Monotone constraints for physically meaningful features
- Outlier removal based on interval relative positions
- Trajectory data filtering (keep only rows with trajectory data)
- Scaled target with flight_id grouped 5-fold CV
- Minimal column dropping: only drop fuel_kg and fuel_kg_actual, keep all numerical columns
"""

import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from pathlib import Path
import json
import re
from datetime import datetime
import argparse

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

    # Mass above OEW (useful for many interactions)
    if 'ac_max_takeoff_weight_kg' in df.columns and 'ac_operating_empty_weight_kg' in df.columns:
        df['mass_above_oew'] = df['ac_max_takeoff_weight_kg'] - df['ac_operating_empty_weight_kg']
        df['mass_above_oew'] = df['mass_above_oew'].clip(lower=1.0)

    # Fuel per residual mass (MTOW - OEW)
    if 'int_fuel_consumed_acropole_kg' in df.columns and 'mass_above_oew' in df.columns:
        df['int_acropole_fuel_per_residual_mass'] = df['int_fuel_consumed_acropole_kg'] / (df['mass_above_oew'] + 1e-6)
    if 'int_fuel_consumed_openap_kg' in df.columns and 'mass_above_oew' in df.columns:
        df['int_openap_fuel_per_residual_mass'] = df['int_fuel_consumed_openap_kg'] / (df['mass_above_oew'] + 1e-6)

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

    # 6. Interaction features among top 25 important features
    # print("   Creating interaction features among top 25 features...")
    # top_features = [
    #     'int_acropole_fuel_per_tow',
    #     'time_per_mass',
    #     'int_acropole_fuel_per_mass',
    #     'interval_duration_sec',
    #     'log_int_fuel_consumed_acropole_kg',
    #     'distance_per_mass',
    #     'int_openap_multiplier_max',
    #     'int_fuel_consumed_acropole_kg_sq',
    #     'ac_efficiency_index',
    #     'int_fuel_consumed_openap_kg',
    #     'interval_rel_span',
    #     'ac_wing_surface_area_m2',
    #     'interval_rel_position_start',
    #     'int_fuel_consumed_acropole_kg',
    #     'int_total_leveloff_time_sec',
    #     'int_crosstrack_distance_nm',
    #     'int_mass_mean_rolling_std_3',
    #     'int_fuel_consumed_acropole_kg_prev_diff',
    #     'ac_Approach_Speed_knot',
    #     'seg00_openap_multiplier_mean',
    #     'seg04_fuel_flow_openap_mean',
    #     'seg_mass_acropole_std',
    #     'int_num_points',
    #     'seg05_sfc_openap',
    #     'seg18_TW'
    # ]
    
    # # Create pairwise interactions: products and ratios
    # for i in range(len(top_features)):
    #     for j in range(i + 1, len(top_features)):
    #         feat1 = top_features[i]
    #         feat2 = top_features[j]
    #         if feat1 in df.columns and feat2 in df.columns:
    #             # Product
    #             df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
    #             # Ratio
    #             df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)

    # 6. Interaction features among top 25 important features
    # print("   Creating interaction features among top 25 features...")
    # top_features = [
    #     'int_acropole_fuel_per_tow',
    #     'time_per_mass',
    #     'int_acropole_fuel_per_mass',
    #     'interval_duration_sec',
    #     'log_int_fuel_consumed_acropole_kg',
    #     'distance_per_mass',
    #     'int_openap_multiplier_max',
    #     'int_fuel_consumed_acropole_kg_sq',
    #     'ac_efficiency_index',
    #     'int_fuel_consumed_openap_kg',
    #     'interval_rel_span',
    #     'ac_wing_surface_area_m2',
    #     'interval_rel_position_start',
    #     'int_fuel_consumed_acropole_kg',
    #     'int_total_leveloff_time_sec',
    #     'int_crosstrack_distance_nm',
    #     'int_mass_mean_rolling_std_3',
    #     'int_fuel_consumed_acropole_kg_prev_diff',
    #     'ac_Approach_Speed_knot',
    #     'seg00_openap_multiplier_mean',
    #     'seg04_fuel_flow_openap_mean',
    #     'seg_mass_acropole_std',
    #     'int_num_points',
    #     'seg05_sfc_openap',
    #     'seg18_TW'
    # ]
    
    # # Create pairwise interactions: products and ratios
    # for i in range(len(top_features)):
    #     for j in range(i + 1, len(top_features)):
    #         feat1 = top_features[i]
    #         feat2 = top_features[j]
    #         if feat1 in df.columns and feat2 in df.columns:
    #             # Product
    #             df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
    #             # Ratio
    #             df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)

    # print(f"   Added {len(top_features) * (len(top_features) - 1)} interaction features")

    # Add new categorical features
    print("   Adding new categorical features...")

    # Time/Position buckets
    if 'interval_duration_sec' in df.columns:
        df['interval_duration_bucket'] = pd.cut(df['interval_duration_sec'], bins=[0, 300, 900, float('inf')], labels=['short', 'medium', 'long'])
    if 'int_altitude_median' in df.columns:
        df['altitude_bucket'] = pd.cut(df['int_altitude_median'], bins=[0, 10000, 25000, 40000, float('inf')], labels=['low', 'medium', 'high', 'very_high'])
    if 'int_groundspeed_median' in df.columns:
        df['ground_speed_bucket'] = pd.cut(df['int_groundspeed_median'], bins=[0, 200, 400, 600, float('inf')], labels=['slow', 'medium', 'fast', 'very_fast'])
    # vertical_rate absent, skip
    # track_change absent, skip

    # Phase/Vertical/Speed profiles
    if 'phase_of_flight' in df.columns:
        df['phase_of_flight_coarse'] = df['phase_of_flight'].map({
            'CLIMB': 'climb', 'CRUISE': 'cruise', 'DESCENT': 'descent', 'TAKEOFF': 'takeoff', 'LANDING': 'landing', 'TAXI': 'taxi', 'HOLDING': 'holding'
        }).fillna('unknown')
    if 'vertical_profile' in df.columns:
        df['vertical_profile_coarse'] = df['vertical_profile'].map({
            'climbing': 'climbing', 'descending': 'descending', 'level': 'level'
        }).fillna('unknown')
    if 'speed_profile' in df.columns:
        df['speed_profile_coarse'] = df['speed_profile'].map({
            'accelerating': 'accelerating', 'decelerating': 'decelerating', 'constant': 'constant'
        }).fillna('unknown')

    # Track/Wind buckets
    if 'track_change' in df.columns:
        df['track_change_bucket'] = pd.cut(df['track_change'], bins=[0, 10, 45, 90, float('inf')], labels=['straight', 'slight_turn', 'turn', 'sharp_turn'])
    if 'wind_speed' in df.columns:
        df['wind_speed_bucket'] = pd.cut(df['wind_speed'], bins=[0, 10, 25, 50, float('inf')], labels=['calm', 'light', 'moderate', 'strong'])
    if 'wind_direction' in df.columns:
        df['wind_direction_bucket'] = pd.cut(df['wind_direction'], bins=[0, 90, 180, 270, 360], labels=['north', 'east', 'south', 'west'], right=False)
    if 'temperature' in df.columns:
        df['temperature_bucket'] = pd.cut(df['temperature'], bins=[-float('inf'), 0, 15, 30, float('inf')], labels=['cold', 'cool', 'warm', 'hot'])
    if 'humidity' in df.columns:
        df['humidity_bucket'] = pd.cut(df['humidity'], bins=[0, 30, 60, 90, 100], labels=['dry', 'moderate', 'humid', 'very_humid'], right=False)
    if 'pressure' in df.columns:
        df['pressure_bucket'] = pd.cut(df['pressure'], bins=[0, 1000, 1013, 1030, float('inf')], labels=['low', 'normal', 'high', 'very_high'])

    # Aircraft buckets
    if 'aircraft_type_encoded' in df.columns:
        df['aircraft_type_bucket'] = df['aircraft_type_encoded'].astype(str).str[:2]  # First 2 chars as bucket
    # engine_type not present, skip
    if 'ac_manufacturer_encoded' in df.columns:
        df['aircraft_manufacturer_bucket'] = df['ac_manufacturer_encoded'].astype(str).str[:3]

    # Route buckets
    if 'origin_country' in df.columns:
        df['origin_country_bucket'] = df['origin_country'].astype(str).str[:2]
    if 'destination_country' in df.columns:
        df['destination_country_bucket'] = df['destination_country'].astype(str).str[:2]
    if 'route_type' in df.columns:
        df['route_type_bucket'] = df['route_type'].astype(str)

    # Flight buckets
    if 'origin_dest_gcdist_nm' in df.columns:
        df['flight_distance_bucket'] = pd.cut(df['origin_dest_gcdist_nm'], bins=[0, 500, 1500, 3000, float('inf')], labels=['short', 'medium', 'long', 'very_long'])
    if 'total_flight_duration_sec' in df.columns:
        df['flight_duration_bucket'] = pd.cut(df['total_flight_duration_sec'], bins=[0, 3600, 7200, 14400, float('inf')], labels=['short', 'medium', 'long', 'very_long'])

    # Temporal buckets
    if 'flight_date' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['flight_date']).dt.dayofweek
        df['month_of_year'] = pd.to_datetime(df['flight_date']).dt.month
        df['season_bucket'] = pd.cut(df['month_of_year'], bins=[0, 3, 6, 9, 12], labels=['winter', 'spring', 'summer', 'fall'], right=False)
    if 'utc_time' in df.columns:
        df['local_time_of_day_bucket'] = pd.cut(pd.to_datetime(df['utc_time']).dt.hour, bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'], right=False)

    # Cruise/Climb/Descent buckets
    if 'cruise_median_altitude' in df.columns:
        df['cruise_altitude_bucket'] = pd.cut(df['cruise_median_altitude'], bins=[0, 25000, 35000, 45000, float('inf')], labels=['low', 'medium', 'high', 'very_high'])
    if 'climb_avg_vr' in df.columns:
        df['climb_rate_bucket'] = pd.cut(df['climb_avg_vr'], bins=[0, 500, 1500, 3000, float('inf')], labels=['slow', 'medium', 'fast', 'very_fast'])
    if 'descent_avg_vr' in df.columns:
        df['descent_rate_bucket'] = pd.cut(df['descent_avg_vr'], bins=[-float('inf'), -3000, -1500, -500, 0], labels=['very_fast', 'fast', 'medium', 'slow'])

    # Duration/Distance buckets
    if 'int_total_leveloff_time_sec' in df.columns:
        df['level_flight_duration_bucket'] = pd.cut(df['int_total_leveloff_time_sec'], bins=[0, 600, 1800, 3600, float('inf')], labels=['short', 'medium', 'long', 'very_long'])
    # climb_distance_nm, descent_distance_nm present
    # total_level_distance_nm skip

    # Fuel/Mass buckets
    if 'int_fuel_flow_acropole_mean' in df.columns:
        df['fuel_flow_bucket'] = pd.cut(df['int_fuel_flow_acropole_mean'], bins=[0, 1000, 3000, 6000, float('inf')], labels=['low', 'medium', 'high', 'very_high'])
    elif 'int_fuel_flow_openap_mean' in df.columns:
        df['fuel_flow_bucket'] = pd.cut(df['int_fuel_flow_openap_mean'], bins=[0, 1000, 3000, 6000, float('inf')], labels=['low', 'medium', 'high', 'very_high'])
    # int_mass_mean present
    # ac_max_takeoff_weight_kg, ac_operating_empty_weight_kg present
    # payload_kg not present, skip
    # ac_fuel_capacity_kg not present, skip

    # Aircraft design buckets
    if 'ac_wing_loading_kg_per_m2' in df.columns:
        df['wing_loading_bucket'] = pd.cut(df['ac_wing_loading_kg_per_m2'], bins=[0, 300, 600, 900, float('inf')], labels=['low', 'medium', 'high', 'very_high'])
    # ac_aspect_ratio present
    # mach, tas, ias, heading absent, skip

    # Navigation buckets
    if 'int_crosstrack_distance_nm' in df.columns:
        df['crosstrack_distance_bucket'] = pd.cut(df['int_crosstrack_distance_nm'], bins=[-float('inf'), -5, -1, 1, 5, float('inf')], labels=['far_left', 'left', 'center', 'right', 'far_right'])
    if 'int_alongtrack_distance_nm' in df.columns:
        df['alongtrack_distance_bucket'] = pd.cut(df['int_alongtrack_distance_nm'], bins=[-float('inf'), -10, -1, 1, 10, float('inf')], labels=['behind', 'slightly_behind', 'on_track', 'slightly_ahead', 'ahead'])

    # Time buckets
    if 'time_since_takeoff_min' in df.columns:
        df['time_since_takeoff_bucket'] = pd.cut(df['time_since_takeoff_min'], bins=[0, 30, 120, 300, float('inf')], labels=['early', 'mid', 'late', 'very_late'])
    if 'time_to_landing_min' in df.columns:
        df['time_to_landing_bucket'] = pd.cut(df['time_to_landing_min'], bins=[0, 30, 120, 300, float('inf')], labels=['very_late', 'late', 'mid', 'early'])
    if 'interval_rel_position_start' in df.columns:
        df['interval_rel_position_bucket'] = pd.cut(df['interval_rel_position_start'], bins=[-1, -0.5, 0, 0.5, 1], labels=['before', 'early', 'mid', 'late'])
    if 'int_num_points' in df.columns:
        df['num_points_bucket'] = pd.cut(df['int_num_points'], bins=[0, 10, 50, 100, float('inf')], labels=['few', 'some', 'many', 'lots'])

    # Interaction categories
    if 'phase_of_flight_coarse' in df.columns and 'altitude_bucket' in df.columns:
        df['phase_altitude'] = df['phase_of_flight_coarse'].astype(str) + '_' + df['altitude_bucket'].astype(str)
    if 'phase_of_flight_coarse' in df.columns and 'vertical_rate_bucket' in df.columns:
        df['phase_vertical_rate'] = df['phase_of_flight_coarse'].astype(str) + '_' + df['vertical_rate_bucket'].astype(str)
    if 'phase_of_flight_coarse' in df.columns and 'ground_speed_bucket' in df.columns:
        df['phase_ground_speed'] = df['phase_of_flight_coarse'].astype(str) + '_' + df['ground_speed_bucket'].astype(str)
    if 'vertical_profile_coarse' in df.columns and 'altitude_bucket' in df.columns:
        df['vertical_altitude'] = df['vertical_profile_coarse'].astype(str) + '_' + df['altitude_bucket'].astype(str)
    if 'speed_profile_coarse' in df.columns and 'ground_speed_bucket' in df.columns:
        df['speed_ground_speed'] = df['speed_profile_coarse'].astype(str) + '_' + df['ground_speed_bucket'].astype(str)
    if 'altitude_bucket' in df.columns and 'wind_speed_bucket' in df.columns:
        df['altitude_wind'] = df['altitude_bucket'].astype(str) + '_' + df['wind_speed_bucket'].astype(str)
    if 'phase_of_flight_coarse' in df.columns and 'wind_direction_bucket' in df.columns:
        df['phase_wind_direction'] = df['phase_of_flight_coarse'].astype(str) + '_' + df['wind_direction_bucket'].astype(str)
    if 'aircraft_type_bucket' in df.columns and 'altitude_bucket' in df.columns:
        df['aircraft_altitude'] = df['aircraft_type_bucket'].astype(str) + '_' + df['altitude_bucket'].astype(str)
    if 'aircraft_type_bucket' in df.columns and 'flight_distance_bucket' in df.columns:
        df['aircraft_distance'] = df['aircraft_type_bucket'].astype(str) + '_' + df['flight_distance_bucket'].astype(str)
    if 'route_type_bucket' in df.columns and 'flight_distance_bucket' in df.columns:
        df['route_distance'] = df['route_type_bucket'].astype(str) + '_' + df['flight_distance_bucket'].astype(str)
    if 'day_of_week' in df.columns and 'phase_of_flight_coarse' in df.columns:
        df['day_phase'] = df['day_of_week'].astype(str) + '_' + df['phase_of_flight_coarse'].astype(str)
    if 'season_bucket' in df.columns and 'altitude_bucket' in df.columns:
        df['season_altitude'] = df['season_bucket'].astype(str) + '_' + df['altitude_bucket'].astype(str)
    if 'local_time_of_day_bucket' in df.columns and 'phase_of_flight_coarse' in df.columns:
        df['time_phase'] = df['local_time_of_day_bucket'].astype(str) + '_' + df['phase_of_flight_coarse'].astype(str)
    if 'cruise_altitude_bucket' in df.columns and 'mach_bucket' in df.columns:
        df['cruise_mach'] = df['cruise_altitude_bucket'].astype(str) + '_' + df['mach_bucket'].astype(str)
    if 'climb_rate_bucket' in df.columns and 'altitude_bucket' in df.columns:
        df['climb_altitude'] = df['climb_rate_bucket'].astype(str) + '_' + df['altitude_bucket'].astype(str)
    if 'descent_rate_bucket' in df.columns and 'altitude_bucket' in df.columns:
        df['descent_altitude'] = df['descent_rate_bucket'].astype(str) + '_' + df['altitude_bucket'].astype(str)
    if 'mass_bucket' in df.columns and 'altitude_bucket' in df.columns:
        df['mass_altitude'] = df['mass_bucket'].astype(str) + '_' + df['altitude_bucket'].astype(str)
    if 'fuel_flow_bucket' in df.columns and 'altitude_bucket' in df.columns:
        df['fuel_flow_altitude'] = df['fuel_flow_bucket'].astype(str) + '_' + df['altitude_bucket'].astype(str)
    if 'wing_loading_bucket' in df.columns and 'aspect_ratio_bucket' in df.columns:
        df['wing_aspect'] = df['wing_loading_bucket'].astype(str) + '_' + df['aspect_ratio_bucket'].astype(str)
    if 'tas_bucket' in df.columns and 'altitude_bucket' in df.columns:
        df['tas_altitude'] = df['tas_bucket'].astype(str) + '_' + df['altitude_bucket'].astype(str)
    if 'heading_bucket' in df.columns and 'wind_direction_bucket' in df.columns:
        df['heading_wind'] = df['heading_bucket'].astype(str) + '_' + df['wind_direction_bucket'].astype(str)
    if 'crosstrack_distance_bucket' in df.columns and 'phase_of_flight_coarse' in df.columns:
        df['crosstrack_phase'] = df['crosstrack_distance_bucket'].astype(str) + '_' + df['phase_of_flight_coarse'].astype(str)
    if 'time_since_takeoff_bucket' in df.columns and 'time_to_landing_bucket' in df.columns:
        df['time_progress'] = df['time_since_takeoff_bucket'].astype(str) + '_' + df['time_to_landing_bucket'].astype(str)
    if 'interval_rel_position_bucket' in df.columns and 'phase_of_flight_coarse' in df.columns:
        df['position_phase'] = df['interval_rel_position_bucket'].astype(str) + '_' + df['phase_of_flight_coarse'].astype(str)

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
        'flt_ground_distance_nm',
        'flt_duration_sec',
        'ac_max_takeoff_weight_kg',
        'mass_above_oew',
        'time_since_takeoff_min',
        'time_to_landing_min',
        'climb_time_sec',
        'descent_time_sec',
        'level_distance_nm',
        'total_level_distance_nm',
        'climb_distance_nm',
        'descent_distance_nm',
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

    # Fill NaN values with specific strategies
    nan_counts = X.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"   Found {total_nans} NaN values across {(nan_counts > 0).sum()} columns")

        # Specific imputation for climb and descent features
        descent_cols = [col for col in X.columns if col.startswith('descent_')]
        for col in descent_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(0)

        climb_cols = [col for col in X.columns if col.startswith('climb_')]
        for col in climb_cols:
            if X[col].isna().any():
                if col == 'climb_avg_vr':
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(X[col].median())

        # Fill remaining numeric columns with 0
        numeric_cols = X.select_dtypes(include=['number']).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)

        # Fill categorical columns with 'UNKNOWN'
        for col in cat_cols:
            if X[col].isna().any():
                X[col] = X[col].cat.add_categories(['UNKNOWN']).fillna('UNKNOWN')
    else:
        print("   No NaN values found")

    return X, cat_cols

def main():
    parser = argparse.ArgumentParser(description='Stage 9.1 - Train CatBoost fuel model with minimal column dropping')
    parser.add_argument('--dataset-type', choices=['lightgbm', 'original'], default='lightgbm', help='Type of imputation: lightgbm or original')
    parser.add_argument('--train-features', help='Path to train features (auto-set based on dataset-type)')
    parser.add_argument('--rank-features', help='Path to rank features (auto-set based on dataset-type)')
    parser.add_argument('--final-features', help='Path to final features (auto-set based on dataset-type)')
    parser.add_argument('--raw-fuel', default=r'F:/PRC_2025/Likable-ant_v1/data/raw/fuel_train.parquet', help='Path to raw fuel train data')
    parser.add_argument('--submission-template', default=r'F:/PRC_2025/Likable-ant_v1/data/raw/fuel_rank_submission.parquet', help='Submission template')
    args = parser.parse_args()

    # Set paths based on dataset type
    base_path = r'F:/PRC_2025/Likable-ant_v1/data/processed/Stage_8_Consolidated_Features'
    train_features_path = Path(f"{base_path}/Train_281125.parquet")
    rank_features_path = Path(f"{base_path}/Rank_281125.parquet")
    final_features_path = Path(f"{base_path}/Final_281125.parquet")

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

    # Initialize variables for metadata
    negative_mask = None
    negative_csv_path = None

    print("=" * 80)
    print(f"CATBOOST FINAL TRAINING - SCALED TARGET ({args.dataset_type.upper()} IMPUTATION) (v{version})")
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
    
    print(f"   Train features shape: {train_df.shape}")
    print(f"   Rank features shape: {rank_df.shape}")
    if final_df is not None:
        print(f"   Final features shape: {final_df.shape}")

    # 2. Create physics-based features
    print("\n2. Engineering physics-based features...")
    train_df = create_physics_features(train_df)
    rank_df = create_physics_features(rank_df)

    # Create additional features for rank data
    # if 'origin_icao' in rank_df.columns and 'destination_icao' in rank_df.columns:
    #     rank_df['origin_destination_icao'] = rank_df['origin_icao'].astype(str) + '_' + rank_df['destination_icao'].astype(str)
    # else:
    #     rank_df['origin_destination_icao'] = 'UNKNOWN_UNKNOWN'
    #rank_df['origin_destination_icao'] = 'UNKNOWN_UNKNOWN'

    # 3. Merge Target & Clean
    print("\n3. Preparing training data...")
    # Drop existing fuel_kg from features to avoid conflict
    if 'fuel_kg' in train_df.columns:
        train_df = train_df.drop(columns=['fuel_kg'])
    merged_df = train_df.merge(fuel_df[['idx', 'fuel_kg']], on='idx', how='left')

    # Remove rows with missing target
    initial_len = len(merged_df)
    merged_df = merged_df.dropna(subset=['fuel_kg'])
    print(f"   Removed {initial_len - len(merged_df)} rows with missing fuel_kg")

    # # Keep rows where has_trajectory_data is True
    # if 'has_trajectory_data' in merged_df.columns:
    #     initial_len = len(merged_df)
    #     merged_df = merged_df[merged_df['has_trajectory_data'] == True]
    #     print(f"   Kept {len(merged_df)} rows where has_trajectory_data is True (removed {initial_len - len(merged_df)} others)")
    # else:
    #     print("   Warning: has_trajectory_data column not found, proceeding without filtering")

    # Remove outliers based on interval relative positions (-0.5 to 1.5 range)
    # if 'interval_rel_position_start' in merged_df.columns and 'interval_rel_position_end' in merged_df.columns:
    #     before_outlier_removal = len(merged_df)
    #     outlier_mask = (
    #         (merged_df['interval_rel_position_start'] >= -1) &
    #         (merged_df['interval_rel_position_start'] <= 1.5) &
    #         (merged_df['interval_rel_position_end'] >= -0.5) &
    #         (merged_df['interval_rel_position_end'] <= 1.5)
    #     )
    #     merged_df = merged_df[outlier_mask]
    #     outliers_removed = before_outlier_removal - len(merged_df)
    #     print(f"   Removed {outliers_removed} rows with interval_rel_position outliers (outside -0.5 to 1.5 range)")

    print(f"   Final training data: {merged_df.shape}")
    print(f"   Original fuel_kg - Mean: {merged_df['fuel_kg'].mean():.2f}, Std: {merged_df['fuel_kg'].std():.2f}")

    # Create additional features
    # if 'origin_icao' in merged_df.columns and 'destination_icao' in merged_df.columns:
    #     merged_df['origin_destination_icao'] = merged_df['origin_icao'].astype(str) + '_' + merged_df['destination_icao'].astype(str)
    # else:
    #     merged_df['origin_destination_icao'] = 'UNKNOWN_UNKNOWN'
    #merged_df['origin_destination_icao'] = 'UNKNOWN_UNKNOWN'

    # Calculate scaling factor (mass above OEW)
    if 'mass_above_oew' not in merged_df.columns:
        mass_above_oew = merged_df['ac_max_takeoff_weight_kg'] - merged_df['ac_operating_empty_weight_kg']
        mass_above_oew = mass_above_oew.clip(lower=1.0)
        merged_df['mass_above_oew'] = mass_above_oew
    else:
        mass_above_oew = merged_df['mass_above_oew']

    # Create scaled target (no squaring, clip negatives instead)
    merged_df['fuel_kg_scaled'] = merged_df['fuel_kg'] / mass_above_oew

    print(f"   Target (fuel_kg_scaled) - Mean: {merged_df['fuel_kg_scaled'].mean():.6f}, Std: {merged_df['fuel_kg_scaled'].std():.6f}")

    # 4. Prepare Features
    print("\n4. Preparing features...")
    # Minimal dropping: only drop fuel_kg and fuel_kg_actual, keep all numerical columns
    drop_cols = ['fuel_kg', 'fuel_kg_actual', 'idx', 'flight_id','fuel_kg_scaled']

    # Drop datetime columns
    datetime_cols = merged_df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    datetime_cols += [c for c in merged_df.columns if c.endswith('_ts')]
    datetime_cols = list(set(datetime_cols))
    drop_cols += datetime_cols

    print(f"   Dropping {len(datetime_cols)} datetime columns")

    X, cat_cols = prepare_features(merged_df, drop_cols)
    y = merged_df['fuel_kg_scaled']
    y_actual = merged_df['fuel_kg']

    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Categorical features: {len(cat_cols)}")
    print(f"   Numeric features: {X.shape[1] - len(cat_cols)}")

    # Get monotone constraints
    #monotone_constraints = get_monotone_constraints(X.columns)
    #n_monotone = sum(1 for c in monotone_constraints if c != 0)
    #print(f"   Monotone constraints applied to {n_monotone} features")
    n_monotone = 0  # Not using monotone constraints for consistency with HPO

    # Save flight_id for grouping
    groups = merged_df['flight_id'].values

    # 5. Set Fixed Hyperparameters
    print("\n5. Setting fixed hyperparameters...")
    
    best_params = {
        'depth': 8,
        'learning_rate': 0.1,
        'iterations': 3000,
        'l2_leaf_reg': 1.0,
        'border_count': 200,
        'bagging_temperature': 0.5,
        'random_strength': 1.0,
        'min_data_in_leaf': 10,
        'task_type': 'GPU',
        'early_stopping_rounds': 200,
        'loss_function': 'RMSE',
        'thread_count': -1,
    }
    
    best_rmse = None  # No validation RMSE since no CV

    # Create small validation set for early stopping
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # 20% for validation
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    print(f"   Training on {len(X_train)} samples, validating on {len(X_val)} samples")

    # 6. Train Final Model (No CV)
    print(f"\n6. Training final model on full dataset (no CV)...")
    final_model = cb.CatBoostRegressor(**best_params, verbose=True)
    final_model.fit(X_train, y_train, cat_features=cat_cols, eval_set=[(X_val, y_val)])
    print("   Done")

    # Compute validation RMSE
    val_preds = final_model.predict(X_val)
    val_rmse_scaled = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"   Validation RMSE on scaled: {val_rmse_scaled:.6f}")
    
    # Unscale predictions for actual fuel RMSE
    mass_above_oew_val = mass_above_oew.iloc[val_idx]
    val_preds_unscaled = val_preds * mass_above_oew_val
    y_actual_val = y_actual.iloc[val_idx]
    val_rmse_actual = np.sqrt(mean_squared_error(y_actual_val, val_preds_unscaled))
    print(f"   Validation RMSE on fuel_kg: {val_rmse_actual:.2f}")
    
    best_rmse = val_rmse_actual

    # 7. Save Feature Importance
    print("\n7. Analyzing feature importance...")
    all_feature_importances = [final_model.get_feature_importance()]
    avg_importance = all_feature_importances[0]
    std_importance = np.zeros_like(avg_importance)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': avg_importance,
        'importance_std': std_importance,
    }).sort_values('importance_mean', ascending=False)

    importance_df.to_csv(feature_importance_path, index=False)
    print(f"   ✓ Saved feature importance to {feature_importance_path}")

    print("\n   Top 20 Features by Importance:")
    print(importance_df.head(20)[['feature', 'importance_mean', 'importance_std']].to_string(index=False))

    n_zero_importance = (importance_df['importance_mean'] == 0).sum()
    print(f"\n   Features with zero importance: {n_zero_importance} / {len(importance_df)}")

    # Check if new physics features are in top 50
    new_features = ['distance_per_mass', 'time_per_mass', 'acropole_openap_error_x_mass',
                    'acropole_openap_error_x_distance', 'level_distance_fraction',
                    'climb_time_fraction', 'fuel_efficiency_per_nm']
    top_50_features = importance_df.head(50)['feature'].tolist()
    new_in_top_50 = [f for f in new_features if f in top_50_features]
    if new_in_top_50:
        print(f"\n   Physics features in top 50: {new_in_top_50}")

    # 9. Predict on Rank Data
    print("\n9. Generating predictions for rank dataset...")

    X_rank, _ = prepare_features(rank_df, drop_cols)

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

    # Predict
    rank_preds_scaled = final_model.predict(X_rank)

    # Convert back to actual fuel
    rank_mass_above_oew = (rank_df['ac_max_takeoff_weight_kg'] - rank_df['ac_operating_empty_weight_kg']).clip(lower=1.0)
    rank_preds = rank_preds_scaled * rank_mass_above_oew

    # Clip negative predictions to 1Kg
    negative_mask = rank_preds < 0
    if negative_mask.any():
        # Save negative predictions with their features before clipping
        negative_indices = rank_df[negative_mask]['idx'].values
        negative_features = rank_df[negative_mask].copy()
        negative_features['predicted_fuel_kg'] = rank_preds[negative_mask]
        negative_features['predicted_fuel_kg_scaled'] = rank_preds_scaled[negative_mask]
        
        negative_csv_path = submissions_dir / f"likable-ant_v{version}_negative_predictions.csv"
        negative_features.to_csv(negative_csv_path, index=False)
        print(f"   ✓ Saved {len(negative_features)} negative predictions with features to {negative_csv_path}")
        
        rank_preds = np.maximum(rank_preds, 1)
        print(f"   Clipped {negative_mask.sum()} negative predictions to 1 kg")

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

    # # Fill missing with median
    # missing_preds = submission_df['fuel_kg'].isna().sum()
    # if missing_preds > 0:
    #     median_val = np.median(rank_preds)
    #     print(f"   Warning: {missing_preds} rows missing. Filling with median: {median_val:.2f}")
    #     submission_df['fuel_kg'] = submission_df['fuel_kg'].fillna(median_val)

    # Validation
    assert submission_df['fuel_kg'].notna().all(), "ERROR: Submission contains NaN!"
    assert (submission_df['fuel_kg'] >= 0).all(), "ERROR: Submission contains negative values!"

    # Save
    submission_df.to_parquet(output_path)
    print(f"   ✓ Saved submission to {output_path}")
    print(f"   Submission shape: {submission_df.shape}")
    print(f"\n   First few predictions:")
    print(submission_df.head(10))

    # 11. Save Metadata
    print("\n11. Saving metadata...")

    pct_error = best_rmse / y_actual.mean() * 100 if best_rmse is not None else None

    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'dataset_type': args.dataset_type,
        'train_features_path': str(train_features_path),
        'rank_features_path': str(rank_features_path),
        'final_features_path': str(final_features_path),
        'train_shape': list(train_df.shape),
        'rank_shape': list(rank_df.shape),
        'final_shape': list(final_df.shape) if final_df is not None else None,
        'n_features': X.shape[1],
        'n_categorical': len(cat_cols),
        'model_params': best_params,
        'cv_folds': None,
        'cv_rmse_mean': best_rmse,
        'cv_rmse_std': None,
        'validation_error_pct': pct_error,
        'target_type': 'fuel_kg_scaled: fuel_kg / (ac_max_takeoff_weight_kg - ac_operating_empty_weight_kg)',
        'model_type': 'CatBoost with Fixed Params + Physics Features + Trajectory Filter + Minimal Column Dropping + Scaled Target (No CV)',
        'physics_features_added': new_features,
        'features_zero_importance': int(n_zero_importance),
        'monotone_constraints_count': n_monotone,
        'trajectory_filter_applied': True,
        'minimal_dropping': True,
        'predictions_mean': float(rank_preds.mean()),
        'predictions_std': float(rank_preds.std()),
        'predictions_min': float(rank_preds.min()),
        'predictions_max': float(rank_preds.max()),
        'negative_predictions_count': int(negative_mask.sum()) if negative_mask is not None and negative_mask.any() else 0,
        'negative_predictions_csv': str(negative_csv_path) if negative_csv_path is not None else None,
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
    if negative_csv_path is not None:
        print(f"Negative Predictions: {negative_csv_path}")
    print(f"\nNo CV performed - trained on full dataset")
    if best_rmse is not None:
        print(f"Best Validation RMSE: {best_rmse:.6f}")
    print(f"Model: CatBoost with Fixed Params + Physics Features + Trajectory Filter + Minimal Dropping + Scaled Target (No CV)")

if __name__ == "__main__":
    main()