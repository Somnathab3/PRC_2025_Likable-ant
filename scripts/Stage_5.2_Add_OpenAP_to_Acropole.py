"""
Add OpenAP Fuel Features to Existing Acropole Trajectories
===========================================================

Enhances Acropole-only trajectories with OpenAP fuel calculations:
1. Loads existing Acropole trajectory files
2. Calculates OpenAP fuel features (thrust, drag, mass, fuel_flow)
3. Adds these features to the trajectory dataframes
4. Overwrites the original files with enhanced versions

Author: PRC 2025
Date: November 22, 2025
"""

import sys
import warnings
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
import argparse

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "openap"))

# Import OpenAP modules
from openap import FuelFlow, Thrust, Drag, prop

# Configuration
CONFIG = {
    'dt_default_hours': 1.0 / 3600.0,  # 1.0 seconds in hours
    'min_mass_fraction': 0.5  # Don't let mass drop below 50% of TOW
}

# Data paths
DATA_ROOT = WORKSPACE_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
TRAJ_ACROPOLE_DIR = PROCESSED_ROOT / "trajectories_acropole_only"
TOW_PRED_DIR = PROCESSED_ROOT / "tow_predictions"

# Dataset configurations
DATASETS = {
    'train': {
        'traj_dir': TRAJ_ACROPOLE_DIR / "train",
        'tow_file': TOW_PRED_DIR / "tow_predictions_train_v2.csv",
        'flightlist_file': RAW_ROOT / "flightlist_train.parquet",
    },
    'rank': {
        'traj_dir': TRAJ_ACROPOLE_DIR / "rank",
        'tow_file': TOW_PRED_DIR / "tow_predictions_rank_v2.csv",
        'flightlist_file': RAW_ROOT / "flightlist_rank.parquet",
    },
    'final': {
        'traj_dir': TRAJ_ACROPOLE_DIR / "final",
        'tow_file': TOW_PRED_DIR / "tow_predictions_final_v2.csv",
        'flightlist_file': RAW_ROOT / "flightlist_final.parquet",
    }
}

# Processing configuration
MAX_WORKERS = 15

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(WORKSPACE_ROOT / "logs" / "add_openap_to_acropole.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_openap_fuel(
    traj_df: pd.DataFrame,
    aircraft_type: str,
    tow_kg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate fuel consumption using OpenAP for 1s trajectories
    
    Returns:
        thrust_n, drag_n, mass_kg, fuel_flow_kgh
    """
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
    tas_kts = traj_df['TAS'].values if 'TAS' in traj_df.columns else None
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
    vs_calculated = np.full(n_points, np.nan)
    vs_calculated[1:] = (np.diff(alt_ft) / dt_seconds[1:]) * 60.0  # Convert to ft/min
    
    # Fill NaN vertical rates with calculated values, then with 0.0 as fallback
    vs_fpm = np.where(np.isnan(vs_fpm), vs_calculated, vs_fpm)
    vs_fpm = np.where(np.isnan(vs_fpm), 0.0, vs_fpm)
    
    # Handle NaN values for TAS - use groundspeed first, then default
    if tas_kts is None:
        tas_kts = gs_kts.copy() if gs_kts is not None else np.full(n_points, 250.0)
    else:
        if gs_kts is not None:
            tas_kts = np.where(np.isnan(tas_kts), gs_kts, tas_kts)
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
                                 tow_kg * CONFIG['min_mass_fraction'])
        except:
            continue
    
    return thrust_n, drag_n, mass_kg, fuel_flow_kgh


def load_flight_metadata(dataset: str) -> Optional[pd.DataFrame]:
    """Load flight metadata from flightlist and TOW predictions"""
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
    
    return metadata


def process_single_flight(
    flight_id: str,
    aircraft_type: str,
    tow_kg: float,
    traj_dir: Path,
    reprocess: bool = False
) -> Tuple[str, bool, str]:
    """
    Add OpenAP features to a single Acropole trajectory file.
    
    Args:
        flight_id: Flight identifier
        aircraft_type: ICAO aircraft type code
        tow_kg: Takeoff weight in kg
        traj_dir: Directory containing trajectory files
        reprocess: If True, force reprocessing even if OpenAP features exist
    
    Returns:
        (flight_id, success, message)
    """
    try:
        traj_file = traj_dir / f"{flight_id}.parquet"
        
        if not traj_file.exists():
            return (flight_id, False, "Trajectory file not found")
        
        # Load existing Acropole trajectory
        traj_df = pd.read_parquet(traj_file)
        
        if len(traj_df) < 10:
            return (flight_id, False, "Trajectory too short")
        
        # Check if OpenAP features already exist (skip unless reprocess=True)
        if 'thrust_openap_n' in traj_df.columns and not reprocess:
            return (flight_id, True, "Already has OpenAP features (skipped)")
        
        # Calculate OpenAP fuel features
        thrust_n, drag_n, mass_openap_kg, ff_openap_kgh = calculate_openap_fuel(
            traj_df, aircraft_type, tow_kg
        )
        
        # Add OpenAP features to dataframe
        traj_df['thrust_openap_n'] = thrust_n
        traj_df['drag_openap_n'] = drag_n
        traj_df['mass_openap_kg'] = mass_openap_kg
        traj_df['fuel_flow_openap_kgh'] = ff_openap_kgh
        
        # Overwrite the original file with enhanced version
        traj_df.to_parquet(traj_file, index=False)
        
        return (flight_id, True, "Success")
        
    except Exception as e:
        return (flight_id, False, f"Error: {str(e)}")


def process_dataset(dataset: str, reprocess: bool = False) -> Dict:
    """
    Process all flights in a dataset.
    
    Args:
        dataset: Dataset name ('train', 'rank', or 'final')
        reprocess: If True, force reprocessing even if OpenAP features exist
    
    Returns:
        Statistics dictionary
    """
    logger.info(f"Processing {dataset} dataset (reprocess={reprocess})...")
    
    config = DATASETS[dataset]
    
    # Load metadata
    metadata = load_flight_metadata(dataset)
    if metadata is None:
        return {'error': 'Failed to load metadata'}
    
    # Filter to only flights that have trajectory files
    traj_dir = config['traj_dir']
    existing_flights = []
    for _, row in metadata.iterrows():
        traj_file = traj_dir / f"{row['flight_id']}.parquet"
        if traj_file.exists():
            existing_flights.append(row)
    
    metadata = pd.DataFrame(existing_flights)
    logger.info(f"Found {len(metadata)} flights with existing trajectory files")
    
    if len(metadata) == 0:
        logger.warning(f"No trajectory files found for {dataset} dataset")
        return {'error': 'No trajectory files found'}
    
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
                reprocess
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Add OpenAP features to Acropole trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--reprocess',
        action='store_true',
        help='Force reprocessing of all files, even if OpenAP features already exist'
    )
    parser.add_argument(
        '--dataset',
        choices=['train', 'rank', 'final', 'all'],
        default='all',
        help='Which dataset to process (default: all)'
    )
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Add OpenAP Features to Acropole Trajectories")
    if args.reprocess:
        logger.info("REPROCESS MODE: Overwriting existing OpenAP features")
    logger.info("=" * 80)
    
    # Create logs directory
    (WORKSPACE_ROOT / "logs").mkdir(exist_ok=True)
    
    # Determine which datasets to process
    if args.dataset == 'all':
        datasets_to_process = ['train', 'rank', 'final']
    else:
        datasets_to_process = [args.dataset]
    
    # Process each dataset
    all_stats = {}
    for dataset in datasets_to_process:
        logger.info(f"\n{'='*80}")
        stats = process_dataset(dataset, reprocess=args.reprocess)
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
    error_report_file = WORKSPACE_ROOT / "logs" / "add_openap_errors.json"
    with open(error_report_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    logger.info(f"\nDetailed error report saved to: {error_report_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
