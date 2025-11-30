"""
Batch Fuel Consumption Calculation for 0.5s Trajectories
=========================================================

Processes all flights from train/rank/final datasets:
1. Loads trajectory files with TAS filled
2. Loads corresponding TOW predictions
3. Calculates OpenAP and Acropole fuel features
4. Saves enhanced trajectories with fuel calculations

Author: PRC 2025
Date: November 14, 2025
"""

import sys
import warnings
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "Acropole"))
sys.path.insert(0, str(WORKSPACE_ROOT / "openap"))

# Import required modules
from acropole.estimator import FuelEstimator
from openap import FuelFlow, Thrust, Drag, WRAP, prop

# Configuration for 1s trajectories
CONFIG_05S = {
    'dt_default_hours': 1.0 / 3600.0,  # 1.0 seconds in hours
    'min_mass_fraction': 0.5  # Don't let mass drop below 50% of TOW
}

# Data paths
DATA_ROOT = WORKSPACE_ROOT / "data" / "processed"
RAW_ROOT = WORKSPACE_ROOT / "data" / "raw"
TRAJ_ROOT = DATA_ROOT / "trajectories_tas_filled"
TOW_ROOT = DATA_ROOT / "tow_predictions"
OUTPUT_ROOT = DATA_ROOT / "trajectories_fuel_calculated"
ENHANCED_PARAMS = DATA_ROOT / "aircraft_params_optimized.csv"

# Dataset configurations  
DATASETS = {
    'train': {
        'traj_dir': TRAJ_ROOT / "train",
        'tow_file': TOW_ROOT / "tow_predictions_train_v2.csv",
        'flightlist_file': RAW_ROOT / "flightlist_train.parquet",
        'output_dir': OUTPUT_ROOT / "train"
    },
    'rank': {
        'traj_dir': TRAJ_ROOT / "rank",
        'tow_file': TOW_ROOT / "tow_predictions_rank_v2.csv",
        'flightlist_file': RAW_ROOT / "flightlist_rank.parquet",
        'output_dir': OUTPUT_ROOT / "rank"
    },
    'final': {
        'traj_dir': TRAJ_ROOT / "final",
        'tow_file': TOW_ROOT / "tow_predictions_final_v2.csv",
        'flightlist_file': RAW_ROOT / "flightlist_final.parquet",
        'output_dir': OUTPUT_ROOT / "final"
    }
}

# Processing configuration
MAX_WORKERS = 15  # Parallel processing workers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(WORKSPACE_ROOT / "logs" / "fuel_calculation_batch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_openap_fuel_05s(
    traj_df: pd.DataFrame,
    aircraft_type: str,
    tow_kg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate fuel consumption using OpenAP for 1s trajectories"""
    n_points = len(traj_df)
    
    try:
        fuelflow = FuelFlow(ac=aircraft_type, use_synonym=True)
        thrust_model = Thrust(ac=aircraft_type, use_synonym=True)
        drag_model = Drag(ac=aircraft_type, use_synonym=True)
    except Exception as e:
        logger.warning(f"OpenAP initialization failed for {aircraft_type}: {e}")
        return (np.full(n_points, np.nan), np.full(n_points, np.nan),
                np.full(n_points, np.nan), np.full(n_points, np.nan))
    
    # Extract trajectory data
    tas_kts = traj_df['TAS'].values
    alt_ft = traj_df['altitude'].values
    vs_fpm = traj_df['vertical_rate'].values
    gs_kts = traj_df['groundspeed'].values if 'groundspeed' in traj_df.columns else None
    
    # Calculate time deltas first (needed for VS calculation)
    if 'timestamp' in traj_df.columns:
        timestamps = pd.to_datetime(traj_df['timestamp'])
        dt_seconds = timestamps.diff().dt.total_seconds().fillna(1.0).values
    else:
        dt_seconds = np.full(n_points, 1.0)
    
    # Calculate vertical speed from altitude changes where VS is NaN
    # VS (ft/min) = (altitude_change_ft / time_seconds) * 60
    vs_calculated = np.full(n_points, np.nan)
    vs_calculated[1:] = (np.diff(alt_ft) / dt_seconds[1:]) * 60.0  # Convert to ft/min
    
    # Fill NaN vertical rates with calculated values, then with 0.0 as fallback
    vs_fpm = np.where(np.isnan(vs_fpm), vs_calculated, vs_fpm)
    vs_fpm = np.where(np.isnan(vs_fpm), 0.0, vs_fpm)
    
    # Handle NaN values for TAS - use groundspeed first, then default
    if gs_kts is not None:
        # Fill NaN TAS with groundspeed where available
        tas_kts = np.where(np.isnan(tas_kts), gs_kts, tas_kts)
    # Fill remaining NaN with default value
    tas_kts = np.where(np.isnan(tas_kts), 250.0, tas_kts)
    
    # Handle NaN values for altitude
    alt_ft = np.where(np.isnan(alt_ft), 30000.0, alt_ft)
    
    dt_hours = dt_seconds / 3600.0
    
    # Initialize arrays
    thrust_n = np.full(n_points, np.nan)
    drag_n = np.full(n_points, np.nan)
    mass_kg = np.full(n_points, np.nan)
    fuel_flow_kgh = np.full(n_points, np.nan)
    
    current_mass = tow_kg
    mass_kg[0] = current_mass
    
    # Calculate for each point
    for i in range(n_points):
        try:
            tas = tas_kts[i]
            alt = alt_ft[i]
            vs = vs_fpm[i]
            
            # Determine phase and calculate thrust
            if vs > 300:
                thrust = thrust_model.climb(tas=tas, alt=alt, roc=vs)
            elif vs < -300:
                thrust = thrust_model.descent_idle(tas=tas, alt=alt)
            else:
                thrust = thrust_model.cruise(tas=tas, alt=alt)
            
            drag = drag_model.clean(mass=current_mass, tas=tas, alt=alt)
            ff_kg_per_sec = fuelflow.enroute(mass=current_mass, tas=tas, alt=alt, vs=vs)
            ff = ff_kg_per_sec * 3600
            
            thrust_n[i] = thrust
            drag_n[i] = drag
            fuel_flow_kgh[i] = ff
            mass_kg[i] = current_mass
            
            if i < n_points - 1:
                fuel_burned_kg = ff_kg_per_sec * dt_seconds[i+1]
                current_mass = max(current_mass - fuel_burned_kg, 
                                 tow_kg * CONFIG_05S['min_mass_fraction'])
        except:
            continue
    
    return thrust_n, drag_n, mass_kg, fuel_flow_kgh


def calculate_acropole_fuel_05s(
    traj_df: pd.DataFrame,
    aircraft_type: str,
    tow_kg: float,
    fe: FuelEstimator
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate fuel consumption using Acropole with enhanced parameters"""
    n_points = len(traj_df)
    
    # CRITICAL: Reset index for proper DataFrame alignment
    traj_df = traj_df.reset_index(drop=True)
    
    # Prepare Acropole input using .values to get clean arrays
    acropole_df = pd.DataFrame({
        'typecode': [aircraft_type] * n_points,
        'groundspeed': traj_df['groundspeed'].fillna(250.0).values,
        'altitude': traj_df['altitude'].fillna(30000.0).values,
        'vertical_rate': traj_df['vertical_rate'].fillna(0.0).values,
    })
    
    estimate_kwargs = {
        'typecode': 'typecode',
        'groundspeed': 'groundspeed',
        'altitude': 'altitude',
        'vertical_rate': 'vertical_rate',
    }
    
    # Add optional features
    if 'timestamp' in traj_df.columns:
        acropole_df['FLIGHT_TIME'] = (traj_df['timestamp'] - traj_df['timestamp'].min()).dt.total_seconds().values
        estimate_kwargs['second'] = 'FLIGHT_TIME'
    
    if 'TAS' in traj_df.columns:
        acropole_df['TRUE_AIR_SPD_KT'] = traj_df['TAS'].fillna(250.0).values
        estimate_kwargs['airspeed'] = 'TRUE_AIR_SPD_KT'
    
    acropole_df['MASS_KG'] = tow_kg
    estimate_kwargs['mass'] = 'MASS_KG'
    
    # Call Acropole
    try:
        estimates = fe.estimate(acropole_df, **estimate_kwargs)
        fuel_flow_kgh = estimates['fuel_flow_kgh'].values
    except Exception as e:
        logger.warning(f"Acropole estimation failed for {aircraft_type}: {e}")
        return np.full(n_points, np.nan), np.full(n_points, np.nan)
    
    # Calculate time deltas
    if 'timestamp' in traj_df.columns:
        timestamps = pd.to_datetime(traj_df['timestamp'])
        dt_seconds = timestamps.diff().dt.total_seconds().fillna(1.0).values
    else:
        dt_seconds = np.full(n_points, 1.0)
    
    dt_hours = dt_seconds / 3600.0
    
    # Track mass
    mass_kg = np.full(n_points, np.nan)
    current_mass = tow_kg
    mass_kg[0] = current_mass
    
    for i in range(1, n_points):
        fuel_burned = fuel_flow_kgh[i-1] * dt_hours[i]
        current_mass = max(current_mass - fuel_burned, 
                          tow_kg * CONFIG_05S['min_mass_fraction'])
        mass_kg[i] = current_mass
    
    return mass_kg, fuel_flow_kgh


def validate_tow(aircraft_type: str, tow_kg: float) -> Tuple[bool, str]:
    """Validate TOW against OpenAP aircraft limits (MTOW and OEW)"""
    if prop is None:
        return True, "Validation skipped (OpenAP prop not available)"
    
    try:
        # Get aircraft properties from OpenAP
        ac_type = aircraft_type.lower()
        ac_data = prop.aircraft(ac_type)
        
        if ac_data is None:
            return True, f"Aircraft type {aircraft_type} not found in OpenAP"
        
        mtow = ac_data.get('limits', {}).get('MTOW', None)
        oew = ac_data.get('limits', {}).get('OEW', None)
        
        if mtow is None or oew is None:
            return True, "MTOW/OEW not available for validation"
        
        # Check if TOW is within valid range
        if tow_kg > mtow:
            return False, f"TOW ({tow_kg:.0f} kg) exceeds MTOW ({mtow:.0f} kg)"
        
        if tow_kg < oew:
            return False, f"TOW ({tow_kg:.0f} kg) below OEW ({oew:.0f} kg)"
        
        return True, "Valid"
        
    except Exception as e:
        return True, f"Validation error: {str(e)}"


def load_flight_metadata(dataset: str) -> Optional[pd.DataFrame]:
    """Load flight metadata from flightlist (aircraft type) and TOW predictions"""
    config = DATASETS[dataset]
    
    # Load flightlist (has aircraft_type, flight_id)
    flightlist_file = config['flightlist_file']
    if not flightlist_file.exists():
        logger.error(f"Flightlist file not found: {flightlist_file}")
        return None
    
    flightlist_df = pd.read_parquet(flightlist_file)
    logger.info(f"Loaded {len(flightlist_df)} flights from flightlist")
    
    # Load TOW predictions
    tow_file = config['tow_file']
    if not tow_file.exists():
        logger.error(f"TOW predictions file not found: {tow_file}")
        return None
    
    tow_df = pd.read_csv(tow_file)
    logger.info(f"Loaded {len(tow_df)} TOW predictions")
    
    # Merge on flight_id
    metadata = pd.merge(
        flightlist_df[['flight_id', 'aircraft_type']],
        tow_df[['flight_id', 'tow']],
        on='flight_id',
        how='inner'
    )
    
    logger.info(f"Merged metadata: {len(metadata)} flights in {dataset} dataset")
    
    # Validate TOWs
    if prop is not None:
        logger.info("Validating TOW against OpenAP aircraft limits...")
        invalid_count = 0
        
        for idx, row in metadata.iterrows():
            valid, msg = validate_tow(row['aircraft_type'], row['tow'])
            if not valid:
                logger.warning(f"Flight {row['flight_id']}: {msg}")
                invalid_count += 1
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} flights with invalid TOW (will still process)")
        else:
            logger.info("All TOW values within valid range")
    
    return metadata


def process_single_flight(
    flight_id: str,
    aircraft_type: str,
    tow_kg: float,
    traj_dir: Path,
    output_dir: Path,
    enhanced_params_path: str
) -> Tuple[str, bool, str]:
    """
    Process fuel calculation for a single flight.
    
    Returns:
        (flight_id, success, error_message)
    """
    try:
        # Input/output paths
        traj_file = traj_dir / f"{flight_id}.parquet"
        output_file = output_dir / f"{flight_id}.parquet"
        
        if not traj_file.exists():
            return (flight_id, False, "Trajectory file not found")
        
        # Skip if already processed
        if output_file.exists():
            return (flight_id, True, "Already processed (skipped)")
        
        # Load trajectory
        traj_df = pd.read_parquet(traj_file)
        
        if len(traj_df) < 10:
            return (flight_id, False, "Trajectory too short")
        
        # Initialize FuelEstimator in worker process (avoids pickle issues)
        fe = FuelEstimator(aircraft_params_path=enhanced_params_path)
        
        # Calculate OpenAP fuel
        thrust_n, drag_n, mass_openap_kg, ff_openap_kgh = calculate_openap_fuel_05s(
            traj_df, aircraft_type, tow_kg
        )
        
        # Calculate Acropole fuel (with enhanced params, no scaling!)
        mass_acropole_kg, ff_acropole_kgh = calculate_acropole_fuel_05s(
            traj_df, aircraft_type, tow_kg, fe
        )
        
        # Add fuel features to dataframe
        traj_df['thrust_openap_n'] = thrust_n
        traj_df['drag_openap_n'] = drag_n
        traj_df['mass_openap_kg'] = mass_openap_kg
        traj_df['fuel_flow_openap_kgh'] = ff_openap_kgh
        traj_df['mass_acropole_kg'] = mass_acropole_kg
        traj_df['fuel_flow_acropole_kgh'] = ff_acropole_kgh
        
        # Save enhanced trajectory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        traj_df.to_parquet(output_file, index=False)
        
        return (flight_id, True, "Success")
        
    except Exception as e:
        return (flight_id, False, f"Error: {str(e)}")


def process_dataset(dataset: str, enhanced_params_path: str) -> Dict:
    """
    Process all flights in a dataset.
    
    Returns:
        Statistics dictionary
    """
    logger.info(f"Processing {dataset} dataset...")
    
    config = DATASETS[dataset]
    
    # Load metadata
    metadata = load_flight_metadata(dataset)
    if metadata is None:
        return {'error': 'Failed to load metadata'}
    
    # Prepare output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'total': len(metadata),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'errors': []
    }
    
    # Process flights in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = []
        for _, row in metadata.iterrows():
            future = executor.submit(
                process_single_flight,
                row['flight_id'],
                row['aircraft_type'],
                row['tow'],
                config['traj_dir'],
                config['output_dir'],
                enhanced_params_path
            )
            futures.append(future)
        
        # Process results with progress bar
        with tqdm(total=len(futures), desc=f"Processing {dataset}") as pbar:
            for future in as_completed(futures):
                flight_id, success, message = future.result()
                
                if success:
                    if "skipped" in message.lower():
                        stats['skipped'] += 1
                    else:
                        stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['errors'].append({
                        'flight_id': flight_id,
                        'error': message
                    })
                
                pbar.update(1)
    
    # Log summary
    logger.info(f"{dataset} Dataset Summary:")
    logger.info(f"  Total: {stats['total']}")
    logger.info(f"  Success: {stats['success']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        logger.warning(f"  First 10 errors:")
        for error in stats['errors'][:10]:
            logger.warning(f"    {error['flight_id']}: {error['error']}")
    
    return stats


def main():
    """Main processing pipeline"""
    logger.info("=" * 80)
    logger.info("Batch Fuel Calculation Pipeline")
    logger.info("=" * 80)
    
    # Create logs directory
    (WORKSPACE_ROOT / "logs").mkdir(exist_ok=True)
    
    # Verify enhanced aircraft params exist
    logger.info("Verifying enhanced aircraft parameters...")
    if not ENHANCED_PARAMS.exists():
        logger.error(f"Enhanced params file not found: {ENHANCED_PARAMS}")
        return
    
    # Test load to verify file is valid
    fe_test = FuelEstimator(aircraft_params_path=str(ENHANCED_PARAMS))
    logger.info(f"Loaded enhanced params for {len(fe_test.aircraft_params)} aircraft types")
    del fe_test  # Clean up test instance
    
    # Process each dataset (FuelEstimator initialized in worker processes)
    all_stats = {}
    for dataset in ['train', 'rank', 'final']:
        logger.info(f"\n{'='*80}")
        stats = process_dataset(dataset, str(ENHANCED_PARAMS))
        all_stats[dataset] = stats
    
    # Overall summary
    logger.info(f"\n{'='*80}")
    logger.info("OVERALL SUMMARY")
    logger.info("=" * 80)
    
    for dataset, stats in all_stats.items():
        if 'error' in stats:
            logger.error(f"{dataset}: {stats['error']}")
        else:
            logger.info(f"{dataset}:")
            logger.info(f"  Total: {stats['total']}")
            logger.info(f"  Success: {stats['success']}")
            logger.info(f"  Skipped: {stats['skipped']}")
            logger.info(f"  Failed: {stats['failed']}")
    
    # Save detailed error report
    error_report_file = WORKSPACE_ROOT / "logs" / "fuel_calculation_errors.json"
    with open(error_report_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    logger.info(f"\nDetailed error report saved to: {error_report_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
