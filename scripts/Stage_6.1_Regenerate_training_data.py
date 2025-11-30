"""
Generate Per-Aircraft Training Data with 100% Flights
======================================================

Generate separate training data files for each aircraft type using 100% of available flights,
with memory optimization and reduced column set for better performance.
Uses parallel processing to generate data for multiple aircraft simultaneously.
"""

import pandas as pd
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from train_point_wise_fuel_multipliers import load_point_wise_training_data

def generate_aircraft_training_data(aircraft_type, ground_truth_path, trajectory_dir):
    """Generate training data for a specific aircraft type"""
    print(f"\n{'='*60}")
    print(f"GENERATING DATA FOR: {aircraft_type}")
    print(f"{'='*60}")

    # Generate training data for this aircraft only
    training_data = load_point_wise_training_data(
        ground_truth_path=ground_truth_path,
        trajectory_dir=trajectory_dir,
        flights_per_aircraft=10000,  # Use up to 10,000 flights (effectively 100%)
        aircraft_filter=[aircraft_type]  # Process only this aircraft
    )

    if len(training_data) == 0:
        print(f"⚠ No training data found for {aircraft_type}")
        return None

    print(f"Initial samples: {len(training_data):,}")
    initial_cols = training_data.shape[1]

    # Keep only essential columns for training
    essential_columns = [
        'flight_id', 'interval_idx', 'aircraft_type',
        'altitude', 'TAS', 'groundspeed', 'vertical_rate',
        'phase_climb', 'phase_cruise', 'phase_descent', 'phase_ground',
        'mass_acropole_kg', 'mass_openap_kg',
        'fuel_flow_acropole_kgh', 'fuel_flow_openap_kgh',
        'acropole_multiplier', 'openap_multiplier'
    ]

    # Keep only essential columns
    training_data = training_data[essential_columns]

    # Optimize data types
    for col in training_data.select_dtypes(include=['float64']).columns:
        training_data[col] = training_data[col].astype('float32')

    for col in training_data.select_dtypes(include=['int64']).columns:
        training_data[col] = training_data[col].astype('int32')

    optimized_cols = training_data.shape[1]

    print(f"Columns reduced: {initial_cols} → {optimized_cols}")
    print(f"Memory optimization applied")

    return training_data

def process_single_aircraft(aircraft_type, ground_truth_path, trajectory_dir, training_data_dir):
    """Process a single aircraft type"""
    output_file = f'{training_data_dir}/point_wise_multiplier_training_data_{aircraft_type}.parquet'

    # Check if file already exists
    if os.path.exists(output_file):
        file_size_mb = os.path.getsize(output_file) / (1024**2)
        df = pd.read_parquet(output_file)
        print(f"   ⏭️  Skipped {aircraft_type}: file already exists ({len(df):,} samples, {file_size_mb:.1f} MB)")
        return (aircraft_type, len(df), True)  # True = skipped

    # Generate training data for this aircraft
    training_data = generate_aircraft_training_data(
        aircraft_type, ground_truth_path, trajectory_dir
    )

    if training_data is not None and len(training_data) > 0:
        # Save to separate file
        training_data.to_parquet(output_file, index=False)
        file_size_mb = os.path.getsize(output_file) / (1024**2)
        print(f"   ✓ Generated {len(training_data):,} samples for {aircraft_type}")
        print(f"     Saved to: {output_file} ({file_size_mb:.1f} MB)")
        return (aircraft_type, len(training_data), False)  # False = processed
    else:
        print(f"   ✗ Failed to generate data for {aircraft_type}")
        return (aircraft_type, 0, False)

def main():
    print("="*80)
    print("GENERATING PER-AIRCRAFT TRAINING DATA WITH 100% FLIGHTS (PARALLEL)")
    print("="*80)

    # Paths
    ground_truth_path = 'data/raw/fuel_train.parquet'
    trajectory_dir = 'data/processed/trajectories_acropole_only/train'

    # Create training data directory
    training_data_dir = 'training_data_per_aircraft'
    os.makedirs(training_data_dir, exist_ok=True)

    # Get all aircraft types from consolidated features file
    print("\n1. Discovering aircraft types from consolidated features file...")
    consolidated_path = 'data/processed/consolidated_features/consolidated_features_train_imputed.parquet'
    
    try:
        df = pd.read_parquet(consolidated_path, engine='fastparquet')
        aircraft_types = df['aircraft_type'].unique()
        aircraft_types = sorted(aircraft_types)
        print(f"Found {len(aircraft_types)} aircraft types in consolidated features:")
        for ac in aircraft_types:
            print(f"   - {ac}")
    except Exception as e:
        print(f"Error reading consolidated features file: {str(e)}")
        return

    # Sort by typical frequency (put common ones first)
    # Based on what we saw: A20N, B788, B789, A21N, A332, A359
    aircraft_order = ['A20N', 'B788', 'B789', 'A21N', 'A332', 'A359']
    aircraft_order = [ac for ac in aircraft_order if ac in aircraft_types]  # Keep only found types

    # Add any remaining aircraft types
    for ac in aircraft_types:
        if ac not in aircraft_order:
            aircraft_order.append(ac)

    print(f"\n2. Processing {len(aircraft_order)} aircraft types...")
    print("   Using parallel processing (up to 8 aircraft simultaneously)")

    # Process all aircraft in parallel
    total_samples = 0
    successful_aircraft = []
    skipped_aircraft = []

    print(f"\n{'='*80}")
    print(f"PROCESSING ALL AIRCRAFT IN PARALLEL")
    print(f"{'='*80}")

    # Use ProcessPoolExecutor to process up to 8 aircraft in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for aircraft_type in aircraft_order:
            future = executor.submit(process_single_aircraft, aircraft_type, ground_truth_path, trajectory_dir, training_data_dir)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"   ❌ Error processing aircraft: {str(e)}")

        print(f"\n   All aircraft processing completed!")

    # Process results
    for aircraft_type, sample_count, was_skipped in results:
        if was_skipped:
            skipped_aircraft.append(aircraft_type)
        elif sample_count > 0:
            successful_aircraft.append(aircraft_type)
            total_samples += sample_count
        # Failed aircraft are not added to any list

    print(f"\n3. Summary:")
    print(f"   ✓ Aircraft processed successfully: {len(successful_aircraft)}")
    print(f"   ✓ Aircraft skipped (already existed): {len(skipped_aircraft)}")
    print(f"   ✓ Total training samples: {total_samples:,}")
    print(f"   ✓ Files saved in: {training_data_dir}/")

    if skipped_aircraft:
        print(f"   ⏭️  Skipped aircraft: {skipped_aircraft}")

    # Create a summary file
    summary_data = []
    for aircraft_type in successful_aircraft + skipped_aircraft:
        file_path = f'{training_data_dir}/point_wise_multiplier_training_data_{aircraft_type}.parquet'
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            summary_data.append({
                'aircraft_type': aircraft_type,
                'samples': len(df),
                'file_path': file_path,
                'file_size_mb': os.path.getsize(file_path) / (1024**2),
                'status': 'skipped' if aircraft_type in skipped_aircraft else 'generated'
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('samples', ascending=False)
        summary_file = f'{training_data_dir}/training_data_summary.csv'
        summary_df.to_csv(summary_file, index=False)

        print(f"   ✓ Summary saved to: {summary_file}")
        print(f"\n   Top aircraft by sample count:")
        for _, row in summary_df.head().iterrows():
            status_icon = "⏭️" if row['status'] == 'skipped' else "✓"
            print(f"     {status_icon} {row['aircraft_type']}: {row['samples']:,} samples")

    print(f"\n{'='*80}")
    print("PER-AIRCRAFT TRAINING DATA GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print("1. Run train_incremental_with_validation.py to train models")

if __name__ == "__main__":
    main()