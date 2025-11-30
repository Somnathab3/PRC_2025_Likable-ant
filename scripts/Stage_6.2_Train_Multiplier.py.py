"""
Per-Aircraft Model Training with Validation
===========================================
Train separate models for each aircraft using only that aircraft's training data
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
import sys
sys.path.append('.')
from train_point_wise_fuel_multipliers import integrate_fuel, apply_point_wise_multipliers
import lightgbm as lgb

def train_single_aircraft_model(aircraft_type, training_data_file):
    """Train models for a single aircraft type using its dedicated training data"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {aircraft_type}")
    print(f"{'='*80}")

    # Load training data for this aircraft
    print(f"Loading training data from: {training_data_file}")
    training_data = pd.read_parquet(training_data_file)

    if len(training_data) < 100:
        print(f"âš  Skipping {aircraft_type}: only {len(training_data)} samples")
        return None

    print(f"Samples: {len(training_data):,}")

    # Fill NaN values using interpolation
    for col in ['altitude', 'groundspeed', 'TAS', 'vertical_rate',
                'mass_acropole_kg', 'mass_openap_kg',
                'fuel_flow_acropole_kgh', 'fuel_flow_openap_kgh']:
        if col in training_data.columns:
            # Use linear interpolation between known values, then backward/forward fill, then 0 as fallback
            training_data[col] = training_data[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').fillna(0)

    # Sort by flight_id and timestamp for time-series features
    training_data = training_data.sort_values(['flight_id', 'timestamp'])

    # Reset index to avoid duplicate labels
    training_data = training_data.reset_index(drop=True)

    # Compute time differences and accelerations
    training_data['dt'] = training_data.groupby('flight_id')['timestamp'].diff().dt.total_seconds()
    training_data['accel_tas'] = training_data.groupby('flight_id')['TAS'].diff() / training_data['dt']
    training_data['accel_groundspeed'] = training_data.groupby('flight_id')['groundspeed'].diff() / training_data['dt']
    training_data['accel_tas'] = training_data['accel_tas'].fillna(0)
    training_data['accel_groundspeed'] = training_data['accel_groundspeed'].fillna(0)

    # Check if features are already computed
    has_features = any('_last_' in col for col in training_data.columns)
    if not has_features:
        print("Computing rolling features...")
        # Key features for rolling averages
        key_features = ['groundspeed', 'TAS', 'vertical_rate', 'altitude', 'mass_acropole_kg', 'fuel_flow_acropole_kgh']

        # Compute rolling averages (past)
        for feat in key_features:
            training_data[f'{feat}_last_10s_mean'] = training_data.groupby('flight_id').apply(
                lambda g: g.set_index('timestamp')[feat].rolling('10s').mean()
            ).values
            
            training_data[f'{feat}_last_60s_mean'] = training_data.groupby('flight_id').apply(
                lambda g: g.set_index('timestamp')[feat].rolling('60s').mean()
            ).values
            
            # Forward averages (next)
            def add_future_mean(g, feat, window):
                g = g.sort_values('timestamp', ascending=False)
                g[f'{feat}_next_{window}s_mean'] = g.set_index('timestamp')[feat].rolling(f'{window}s').mean().values
                g = g.sort_values('timestamp', ascending=True)
                return g
            
            training_data = training_data.groupby('flight_id').apply(lambda g: add_future_mean(g, feat, 10)).reset_index(drop=True)
            training_data = training_data.groupby('flight_id').apply(lambda g: add_future_mean(g, feat, 60)).reset_index(drop=True)
            
            training_data[f'{feat}_next_10s_mean'] = training_data[f'{feat}_next_10s_mean'].fillna(training_data[feat])
            training_data[f'{feat}_next_60s_mean'] = training_data[f'{feat}_next_60s_mean'].fillna(training_data[feat])
    else:
        print("Rolling features already present, skipping computation...")

    # Define features (no aircraft one-hot encoding needed since we're training per aircraft)
    feature_cols = [
        'phase_climb', 'phase_cruise', 'phase_descent', 'phase_ground',
        'altitude', 'groundspeed', 'TAS', 'vertical_rate',
        'mass_acropole_kg', 'mass_openap_kg',
        'fuel_flow_acropole_kgh', 'fuel_flow_openap_kgh',
        'accel_tas', 'accel_groundspeed'
    ] + [f'{feat}_last_10s_mean' for feat in key_features] + [f'{feat}_last_60s_mean' for feat in key_features] + [f'{feat}_next_10s_mean' for feat in key_features] + [f'{feat}_next_60s_mean' for feat in key_features]

    X = training_data[feature_cols]
    y_acro = training_data['acropole_multiplier']
    y_open = training_data['openap_multiplier']

    # Train Acropole model
    print("Training Acropole model...")
    acro_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        device='gpu',  # Enable GPU acceleration
        random_state=42,
        verbose=-1
    )
    acro_model.fit(X, y_acro)

    # Train OpenAP model
    print("Training OpenAP model...")
    open_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        device='gpu',  # Enable GPU acceleration
        random_state=42,
        verbose=-1
    )
    open_model.fit(X, y_open)

    return {
        'acropole_model': acro_model,
        'openap_model': open_model,
        'n_samples': len(training_data),
        'feature_cols': feature_cols
    }

def validate_single_aircraft(aircraft_type, models, max_flights=1000):
    """Validate a single aircraft model on interval fuel RMSE"""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {aircraft_type}")
    print(f"{'='*80}")

    # Load ground truth
    ground_truth_df = pd.read_parquet('data/raw/fuel_train.parquet')

    # Load consolidated features to get aircraft types
    consolidated_path = 'data/processed/consolidated_features/consolidated_features_train_imputed.parquet'
    consolidated_df = pd.read_parquet(consolidated_path, engine='fastparquet')

    # Merge to get aircraft types
    merged_data = ground_truth_df.merge(consolidated_df, on='idx', how='inner')

    # Handle duplicate flight_id columns after merge
    if 'flight_id_x' in merged_data.columns:
        merged_data['flight_id'] = merged_data['flight_id_x']
        merged_data = merged_data.drop(columns=['flight_id_x', 'flight_id_y'], errors='ignore')

    # Get validation flights for this aircraft type
    aircraft_flights = merged_data[merged_data['aircraft_type'] == aircraft_type]['flight_id'].unique()
    validation_flights = aircraft_flights[:max_flights]

    print(f"Validating on {len(validation_flights)} flights for {aircraft_type}...")

    # Check GPU usage for this aircraft
    if aircraft_type in models['aircraft_models']:
        acro_model = models['aircraft_models'][aircraft_type]['acropole_model']
        device = acro_model.booster_.params.get('device', 'cpu')
        print(f"    Model device: {device} (GPU enabled: {device == 'gpu'})")

    trajectory_dir = 'data/processed/trajectories_acropole_only/train'

    interval_results = []
    processed = 0
    skipped = 0

    for flight_id in tqdm(validation_flights, desc="Validating"):
        try:
            traj_file = os.path.join(trajectory_dir, f"{flight_id}.parquet")
            if not os.path.exists(traj_file):
                skipped += 1
                continue

            traj_df = pd.read_parquet(traj_file)

            # Ensure timestamps are timezone-aware
            if not pd.api.types.is_datetime64_any_dtype(traj_df['timestamp']):
                traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])
            if traj_df['timestamp'].dt.tz is None:
                traj_df['timestamp'] = traj_df['timestamp'].dt.tz_localize('UTC')

            # Apply multipliers
            traj_corrected = apply_point_wise_multipliers(traj_df, models)

            # Get intervals for this flight
            flight_intervals = merged_data[merged_data['flight_id'] == flight_id]

            for _, interval in flight_intervals.iterrows():
                try:
                    # Extract interval data
                    start_time = pd.to_datetime(interval['start'])
                    end_time = pd.to_datetime(interval['end'])

                    # Make interval times timezone-aware if needed
                    if start_time.tz is None:
                        start_time = start_time.tz_localize('UTC')
                    if end_time.tz is None:
                        end_time = end_time.tz_localize('UTC')

                    ground_truth_fuel = interval['fuel_kg']

                    # Get trajectory points in this interval
                    mask = (traj_df['timestamp'] >= start_time) & (traj_df['timestamp'] <= end_time)
                    interval_points = traj_df[mask].copy()

                    mask_corr = (traj_corrected['timestamp'] >= start_time) & (traj_corrected['timestamp'] <= end_time)
                    interval_points_corr = traj_corrected[mask_corr].copy()

                    if len(interval_points) < 2:
                        skipped += 1
                        continue

                    # Integrate fuel for this interval
                    acropole_raw = integrate_fuel(interval_points['fuel_flow_acropole_kgh'].values, interval_points['timestamp'])
                    acropole_corrected = integrate_fuel(interval_points_corr['adjusted_fuel_flow_acropole_kgh'].values, interval_points_corr['timestamp'])

                    # For OpenAP, we need to calculate it differently since apply_point_wise_multipliers
                    # only corrects Acropole. Let's calculate OpenAP raw and assume same multiplier effect
                    openap_raw = integrate_fuel(interval_points['fuel_flow_openap_kgh'].values, interval_points['timestamp'])
                    openap_corrected = integrate_fuel(interval_points_corr['adjusted_fuel_flow_openap_kgh'].values, interval_points_corr['timestamp'])

                    interval_results.append({
                        'flight_id': flight_id,
                        'interval_idx': interval['idx'],
                        'actual_fuel_kg': interval['fuel_kg'],
                        'acropole_raw_kg': acropole_raw,
                        'acropole_corrected_kg': acropole_corrected,
                        'openap_raw_kg': openap_raw,
                        'openap_corrected_kg': openap_corrected
                    })
                    processed += 1

                except Exception as e:
                    skipped += 1
                    continue

        except Exception as e:
            skipped += 1
            continue

    print(f"\nâœ“ Processed: {processed} intervals, Skipped: {skipped}")

    if len(interval_results) == 0:
        print("âš  No intervals validated!")
        return None

    results_df = pd.DataFrame(interval_results)

    # Calculate RMSE
    acro_raw_rmse = np.sqrt(mean_squared_error(results_df['actual_fuel_kg'], results_df['acropole_raw_kg']))
    acro_corr_rmse = np.sqrt(mean_squared_error(results_df['actual_fuel_kg'], results_df['acropole_corrected_kg']))
    acro_improvement = ((acro_raw_rmse - acro_corr_rmse) / acro_raw_rmse) * 100

    open_raw_rmse = np.sqrt(mean_squared_error(results_df['actual_fuel_kg'], results_df['openap_raw_kg']))
    open_corr_rmse = np.sqrt(mean_squared_error(results_df['actual_fuel_kg'], results_df['openap_corrected_kg']))
    open_improvement = ((open_raw_rmse - open_corr_rmse) / open_raw_rmse) * 100

    print(f"\nðŸ“Š RESULTS ({len(results_df)} intervals):")
    print(f"   Acropole:  {acro_raw_rmse:.2f} kg â†’ {acro_corr_rmse:.2f} kg  ({acro_improvement:+.2f}%) {'âœ“' if acro_improvement > 0 else 'âœ—'}")
    print(f"   OpenAP:    {open_raw_rmse:.2f} kg â†’ {open_corr_rmse:.2f} kg  ({open_improvement:+.2f}%) {'âœ“' if open_improvement > 0 else 'âœ—'}")

    return {
        'n_intervals': len(results_df),
        'acropole_improvement': acro_improvement,
        'openap_improvement': open_improvement,
        'acropole_raw_rmse': acro_raw_rmse,
        'acropole_corrected_rmse': acro_corr_rmse,
        'openap_raw_rmse': open_raw_rmse,
        'openap_corrected_rmse': open_corr_rmse
    }

def main():
    print("="*80)
    print("PER-AIRCRAFT MODEL TRAINING WITH VALIDATION")
    print("="*80)

    # Define feature columns (must match training)
    key_features = ['groundspeed', 'TAS', 'vertical_rate', 'altitude', 'mass_acropole_kg', 'fuel_flow_acropole_kgh']
    feature_cols = [
        'phase_climb', 'phase_cruise', 'phase_descent', 'phase_ground',
        'altitude', 'groundspeed', 'TAS', 'vertical_rate',
        'mass_acropole_kg', 'mass_openap_kg',
        'fuel_flow_acropole_kgh', 'fuel_flow_openap_kgh',
        'accel_tas', 'accel_groundspeed'
    ] + [f'{feat}_last_10s_mean' for feat in key_features] + [f'{feat}_last_60s_mean' for feat in key_features] + [f'{feat}_next_10s_mean' for feat in key_features] + [f'{feat}_next_60s_mean' for feat in key_features]

    # Load training data summary
    training_data_dir = 'training_data_per_aircraft'
    summary_file = f'{training_data_dir}/training_data_summary.csv'

    if not os.path.exists(summary_file):
        print(f"âŒ Training data summary not found: {summary_file}")
        print("Please run regenerate_training_data.py first")
        return

    print("\n1. Loading training data summary...")
    summary_df = pd.read_csv(summary_file)
    summary_df = summary_df.sort_values('samples', ascending=False)

    print(f"Found {len(summary_df)} aircraft with training data:")
    for _, row in summary_df.iterrows():
        print(f"   - {row['aircraft_type']}: {row['samples']:,} samples")

    # Create models directory
    models_dir = 'models/per_aircraft'
    os.makedirs(models_dir, exist_ok=True)

    # Track results
    all_models = {}
    validation_results = []

    # Train each aircraft
    print(f"\n2. Training models per aircraft...")
    for _, row in summary_df.iterrows():
        aircraft_type = row['aircraft_type']
        training_file = row['file_path']

        # Check if model already exists
        model_file = f'{models_dir}/{aircraft_type}_model.pkl'
        if os.path.exists(model_file):
            print(f"\nâœ“ Model already exists for {aircraft_type}, skipping...")
            with open(model_file, 'rb') as f:
                aircraft_model = pickle.load(f)
            all_models[aircraft_type] = aircraft_model
            continue

        # Train the model
        aircraft_model = train_single_aircraft_model(aircraft_type, training_file)

        if aircraft_model:
            # Save immediately
            with open(model_file, 'wb') as f:
                pickle.dump(aircraft_model, f)
            print(f"âœ“ Saved: {model_file}")

            all_models[aircraft_type] = aircraft_model

            # Validate immediately
            temp_models = {
                'aircraft_models': {aircraft_type: aircraft_model},
                'feature_cols': aircraft_model['feature_cols'],
                'aircraft_types': [aircraft_type]
            }

            val_result = validate_single_aircraft(aircraft_type, temp_models, max_flights=10000)
            if val_result:
                val_result['aircraft_type'] = aircraft_type
                validation_results.append(val_result)

    # Save combined models
    print(f"\n{'='*80}")
    print("SAVING COMBINED MODELS")
    print(f"{'='*80}")

    final_models = {
        'aircraft_models': all_models,
        'aircraft_types': list(all_models.keys()),
        'feature_cols': feature_cols  # Add feature columns
    }

    os.makedirs('models', exist_ok=True)
    with open('models/point_wise_fuel_multiplier_models.pkl', 'wb') as f:
        pickle.dump(final_models, f)

    print(f"\nâœ“ Saved {len(all_models)} aircraft models to:")
    print("   models/point_wise_fuel_multiplier_models.pkl")
    print(f"âœ“ Individual models saved in: {models_dir}/")

    # Save validation results
    if validation_results:
        results_df = pd.DataFrame(validation_results)
        results_file = 'logs/per_aircraft_validation_results.csv'
        os.makedirs('logs', exist_ok=True)
        results_df.to_csv(results_file, index=False)

        print(f"\nâœ“ Validation results saved to: {results_file}")

        # Summary
        print(f"\nðŸ“Š OVERALL VALIDATION SUMMARY:")
        avg_acro_improvement = results_df['acropole_improvement'].mean()
        avg_open_improvement = results_df['openap_improvement'].mean()
        print(f"   Average Acropole improvement: {avg_acro_improvement:+.2f}%")
        print(f"   Average OpenAP improvement: {avg_open_improvement:+.2f}%")

    print(f"\n{'='*80}")
    print("PER-AIRCRAFT TRAINING COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
                    
