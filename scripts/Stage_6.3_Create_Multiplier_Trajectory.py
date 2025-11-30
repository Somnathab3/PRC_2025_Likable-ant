"""
Apply Per-Aircraft Multipliers to All Datasets
===============================================
Apply trained multiplier models to:
1. Full train dataset (all flights)
2. Rank dataset
3. Final dataset

IMPORTANT: Features must match exactly with training data generation!
"""
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from tqdm.auto import tqdm

def get_flight_phase(vertical_rate_fpm: float, altitude_ft: float) -> str:
    """Determine flight phase based on vertical rate and altitude - MUST MATCH TRAINING!"""
    if altitude_ft < 500:
        return 'ground'
    elif altitude_ft < 10000:
        if vertical_rate_fpm > 500:
            return 'climb'
        elif vertical_rate_fpm < -500:
            return 'descent'
        else:
            return 'cruise'
    else:
        if vertical_rate_fpm > 300:
            return 'climb'
        elif vertical_rate_fpm < -300:
            return 'descent'
        else:
            return 'cruise'

def process_single_trajectory(args):
    """Process a single trajectory file - for multiprocessing"""
    traj_file, input_dir, output_dir, aircraft_models, feature_cols, all_aircraft_types = args
    
    try:
        flight_id = traj_file.stem
        
        # Check if output already exists - use absolute path
        output_file = Path(output_dir) / f"{flight_id}.parquet"
        if output_file.exists():
            return 'skipped', flight_id
        
        # Load trajectory - use absolute path
        traj_path = Path(input_dir) / f"{flight_id}.parquet"
        traj_df = pd.read_parquet(traj_path)
        
        # Get aircraft type
        aircraft_type = traj_df['aircraft_type'].iloc[0] if 'aircraft_type' in traj_df.columns else None
        
        if aircraft_type is None:
            return 'skipped', flight_id
        
        if aircraft_type not in aircraft_models:
            # No model for this aircraft - save with multiplier = 1.0
            traj_df['acropole_multiplier'] = 1.0
            traj_df['openap_multiplier'] = 1.0
            traj_df['fuel_flow_acropole_corrected_kgh'] = traj_df['fuel_flow_acropole_kgh']
            traj_df['fuel_flow_openap_corrected_kgh'] = traj_df['fuel_flow_openap_kgh']
            
            # Add phase column for consistency
            traj_df['phase'] = traj_df.apply(
                lambda row: get_flight_phase(row['vertical_rate'], row['altitude']), 
                axis=1
            )
            
            output_file = Path(output_dir) / f"{flight_id}.parquet"
            traj_df.to_parquet(output_file, index=False)
            return 'no_model', flight_id
        
        # Ensure timestamps are timezone-aware
        if not pd.api.types.is_datetime64_any_dtype(traj_df['timestamp']):
            traj_df['timestamp'] = pd.to_datetime(traj_df['timestamp'])
        if traj_df['timestamp'].dt.tz is None:
            traj_df['timestamp'] = traj_df['timestamp'].dt.tz_localize('UTC')
        
        # Add phase detection - MUST MATCH TRAINING EXACTLY
        traj_df['phase'] = traj_df.apply(
            lambda row: get_flight_phase(row['vertical_rate'], row['altitude']), 
            axis=1
        )
        traj_df['phase_climb'] = (traj_df['phase'] == 'climb').astype(int)
        traj_df['phase_cruise'] = (traj_df['phase'] == 'cruise').astype(int)
        traj_df['phase_descent'] = (traj_df['phase'] == 'descent').astype(int)
        traj_df['phase_ground'] = (traj_df['phase'] == 'ground').astype(int)
        
        # One-hot encode aircraft types
        for ac in all_aircraft_types:
            traj_df[f'aircraft_{ac}'] = (traj_df['aircraft_type'] == ac).astype(int)
        
        # Prepare features for prediction - only use available columns
        available_features = [f for f in feature_cols if f in traj_df.columns]
        X = traj_df[available_features].fillna(0)
        
        # Get models for this aircraft
        acropole_model = aircraft_models[aircraft_type]['acropole_model']
        openap_model = aircraft_models[aircraft_type]['openap_model']
        
        # Predict multipliers
        acropole_multiplier = acropole_model.predict(X)
        openap_multiplier = openap_model.predict(X)
        
        # Add multipliers to trajectory
        traj_df['acropole_multiplier'] = acropole_multiplier
        traj_df['openap_multiplier'] = openap_multiplier
        
        # Calculate corrected fuel flows
        traj_df['fuel_flow_acropole_corrected_kgh'] = traj_df['fuel_flow_acropole_kgh'] * traj_df['acropole_multiplier']
        traj_df['fuel_flow_openap_corrected_kgh'] = traj_df['fuel_flow_openap_kgh'] * traj_df['openap_multiplier']
        
        # Remove one-hot encoded columns to save space (keep phase and aircraft_type)
        columns_to_drop = [col for col in traj_df.columns if col.startswith('phase_') or col.startswith('aircraft_')]
        traj_df = traj_df.drop(columns=columns_to_drop, errors='ignore')
        
        # Save new trajectory with multipliers
        output_file = Path(output_dir) / f"{flight_id}.parquet"
        traj_df.to_parquet(output_file, index=False)
        
        return 'processed', flight_id
        
    except Exception as e:
        return 'error', f"{flight_id}: {str(e)}"

def apply_multipliers_to_dataset(dataset_name, input_dir, output_dir, aircraft_models, feature_cols, all_aircraft_types):
    """Apply multipliers to all trajectories in a dataset"""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all trajectory files
    trajectory_files = list(Path(input_dir).glob('*.parquet'))
    print(f"Found {len(trajectory_files)} trajectory files")
    
    # Check if all output files already exist
    output_path = Path(output_dir)
    all_exist = True
    for traj_file in trajectory_files:
        flight_id = traj_file.stem
        output_file = output_path / f"{flight_id}.parquet"
        if not output_file.exists():
            all_exist = False
            break
    
    if all_exist:
        print(f" All {len(trajectory_files)} trajectories already processed, skipping dataset")
        return len(trajectory_files), 0, 0, 0  # processed, no_model, skipped, errors
    
    # Prepare arguments for multiprocessing - use absolute paths
    abs_input_dir = str(Path(input_dir).resolve())
    abs_output_dir = str(Path(output_dir).resolve())
    args_list = [(traj_file, abs_input_dir, abs_output_dir, aircraft_models, feature_cols, all_aircraft_types) 
                 for traj_file in trajectory_files]
    
    # Use sequential processing to avoid multiprocessing issues with LightGBM models
    print(f"Using sequential processing...")
    
    processed = 0
    skipped = 0
    no_model = 0
    errors = 0
    
    for traj_file in tqdm(trajectory_files, desc=f"Processing {dataset_name}"):
        result = process_single_trajectory((traj_file, abs_input_dir, abs_output_dir, aircraft_models, feature_cols, all_aircraft_types))
        status, info = result
        
        if status == 'processed':
            processed += 1
        elif status == 'no_model':
            no_model += 1
        elif status == 'skipped':
            skipped += 1
        elif status == 'error':
            errors += 1
            print(f"   ERROR: {info}")
    
    print(f"\n Processed: {processed} trajectories")
    print(f" No model (used 1.0): {no_model} trajectories")
    print(f"⚠ Skipped: {skipped} trajectories")
    print(f"❌ Errors: {errors} trajectories")
    print(f" Saved to: {output_dir}")
    
    return processed, no_model, skipped, errors


def main():
    print("="*80)
    print("APPLY PER-AIRCRAFT MULTIPLIERS TO ALL DATASETS")
    print("="*80)
    
    # Load trained models
    print("\n1. Loading trained per-aircraft models...")
    with open('models/point_wise_fuel_multiplier_models.pkl', 'rb') as f:
        models_data = pickle.load(f)
    
    aircraft_models = models_data['aircraft_models']
    feature_cols = models_data.get('feature_cols', [
        'phase_climb', 'phase_cruise', 'phase_descent', 'phase_ground',
        'altitude', 'groundspeed', 'TAS', 'vertical_rate',
        'mass_acropole_kg', 'mass_openap_kg',
        'fuel_flow_acropole_kgh', 'fuel_flow_openap_kgh'
    ])
    all_aircraft_types = models_data['aircraft_types']
    
    print(f"    Loaded models for {len(aircraft_models)} aircraft types")
    
    # Check GPU usage for a sample model
    if aircraft_models:
        sample_aircraft = list(aircraft_models.keys())[0]
        acro_model = aircraft_models[sample_aircraft]['acropole_model']
        device = acro_model.booster_.params.get('device', 'cpu')
        print(f"    Model device: {device} (GPU enabled: {device == 'gpu'})")
    
    # Dataset configurations
    datasets = [
        {
            'name': 'train',
            'input': 'data/processed/trajectories_acropole_only/train',
            'output': 'data/processed/trajectories_with_multipliers/train'
        },
        {
            'name': 'rank',
            'input': 'data/processed/trajectories_acropole_only/rank',
            'output': 'data/processed/trajectories_with_multipliers/rank'
        },
        {
            'name': 'final',
            'input': 'data/processed/trajectories_acropole_only/final',
            'output': 'data/processed/trajectories_with_multipliers/final'
        }
    ]
    
    summary = []
    
    for dataset in datasets:
        if not Path(dataset['input']).exists():
            print(f"\n⚠ Skipping {dataset['name']}: input directory not found")
            continue
        
        processed, no_model, skipped, errors = apply_multipliers_to_dataset(
            dataset['name'],
            dataset['input'],
            dataset['output'],
            aircraft_models,
            feature_cols,
            all_aircraft_types
        )
        
        summary.append({
            'dataset': dataset['name'],
            'processed': processed,
            'no_model': no_model,
            'skipped': skipped,
            'errors': errors,
            'total': processed + no_model + skipped + errors
        })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Dataset':<10} {'Processed':<12} {'No Model':<12} {'Skipped':<12} {'Errors':<12} {'Total':<12}")
    print("-"*80)
    for s in summary:
        print(f"{s['dataset']:<10} {s['processed']:<12} {s['no_model']:<12} {s['skipped']:<12} {s.get('errors', 0):<12} {s['total']:<12}")
    print("-"*80)
    
    print("\n✅ All datasets processed!")
    print("\nNext steps:")
    print("1. Integrate corrected fuel flows over intervals")
    print("2. Generate submission files")
    print("="*80)


if __name__ == "__main__":
    main()
