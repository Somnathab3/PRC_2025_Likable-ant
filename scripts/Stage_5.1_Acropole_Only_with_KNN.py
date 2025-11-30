"""
Simplified Acropole-Only Fuel Calculation with KNN Gap Filling
===============================================================

Processes all flights from train/rank/final datasets:
1. Loads trajectory files with TAS filled
2. Loads corresponding TOW predictions
3. Calculates ONLY Acropole fuel features
4. Fills missing/NaN predictions using KNN
5. Compares with actual fuel for training data (RMSE)
6. Generates submission files for rank dataset

Author: PRC 2025
Date: November 22, 2025
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "Acropole"))
sys.path.insert(0, str(WORKSPACE_ROOT / "openap"))

# Import required modules
from acropole.estimator import FuelEstimator

# Configuration for 1s trajectories
CONFIG = {
    'dt_default_hours': 1.0 / 3600.0,  # 1.0 seconds in hours
    'min_mass_fraction': 0.5,  # Don't let mass drop below 50% of TOW
    'knn_neighbors': 5,  # Number of neighbors for KNN filling
}

# Data paths
DATA_ROOT = WORKSPACE_ROOT / "data" / "processed"
RAW_ROOT = WORKSPACE_ROOT / "data" / "raw"
TRAJ_ROOT = DATA_ROOT / "trajectories_tas_filled"
TOW_ROOT = DATA_ROOT / "tow_predictions"
OUTPUT_ROOT = DATA_ROOT / "trajectories_acropole_only"
SUBMISSION_ROOT = WORKSPACE_ROOT / "data" / "submissions"
ENHANCED_PARAMS = DATA_ROOT / "aircraft_params_optimized.csv"

# Dataset configurations  
DATASETS = {
    'train': {
        'traj_dir': TRAJ_ROOT / "train",
        'tow_file': TOW_ROOT / "tow_predictions_train_v2.csv",
        'flightlist_file': RAW_ROOT / "flightlist_train.parquet",
        'output_dir': OUTPUT_ROOT / "train",
        'challenge_file': RAW_ROOT / "fuel_train.parquet"
    },
    'rank': {
        'traj_dir': TRAJ_ROOT / "rank",
        'tow_file': TOW_ROOT / "tow_predictions_rank_v2.csv",
        'flightlist_file': RAW_ROOT / "flightlist_rank.parquet",
        'output_dir': OUTPUT_ROOT / "rank",
        'challenge_file': None
    },
    'final': {
        'traj_dir': TRAJ_ROOT / "final",
        'tow_file': TOW_ROOT / "tow_predictions_final_v2.csv",
        'flightlist_file': RAW_ROOT / "flightlist_final.parquet",
        'output_dir': OUTPUT_ROOT / "final",
        'challenge_file': None
    }
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(WORKSPACE_ROOT / "logs" / "acropole_only_calculation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_acropole_fuel(
    traj_df: pd.DataFrame,
    aircraft_type: str,
    tow_kg: float,
    fe: FuelEstimator
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate fuel consumption using Acropole with enhanced parameters
    
    Returns:
        mass_kg, fuel_flow_kgh, total_fuel_kg
    """
    n_points = len(traj_df)
    
    # Reset index for proper DataFrame alignment
    traj_df = traj_df.reset_index(drop=True)
    
    # Calculate time deltas
    if 'timestamp' in traj_df.columns:
        timestamps = pd.to_datetime(traj_df['timestamp'])
        dt_seconds = timestamps.diff().dt.total_seconds().fillna(1.0).values
    else:
        dt_seconds = np.full(n_points, 1.0)
    
    # Calculate VS from altitude changes if NaN
    alt_ft = traj_df['altitude'].fillna(30000.0).values
    vs_fpm = traj_df['vertical_rate'].values
    
    vs_calculated = np.full(n_points, np.nan)
    vs_calculated[1:] = (np.diff(alt_ft) / dt_seconds[1:]) * 60.0
    vs_fpm = np.where(np.isnan(vs_fpm), vs_calculated, vs_fpm)
    vs_fpm = np.where(np.isnan(vs_fpm), 0.0, vs_fpm)
    
    # Prepare Acropole input
    acropole_df = pd.DataFrame({
        'typecode': [aircraft_type] * n_points,
        'groundspeed': traj_df['groundspeed'].fillna(250.0).values,
        'altitude': alt_ft,
        'vertical_rate': vs_fpm,
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
        tas = traj_df['TAS'].values
        gs = traj_df['groundspeed'].values if 'groundspeed' in traj_df.columns else None
        
        # Fill TAS with groundspeed if available
        if gs is not None:
            tas = np.where(np.isnan(tas), gs, tas)
        tas = np.where(np.isnan(tas), 250.0, tas)
        
        acropole_df['TRUE_AIR_SPD_KT'] = tas
        estimate_kwargs['airspeed'] = 'TRUE_AIR_SPD_KT'
    
    acropole_df['MASS_KG'] = tow_kg
    estimate_kwargs['mass'] = 'MASS_KG'
    
    # Call Acropole
    try:
        estimates = fe.estimate(acropole_df, **estimate_kwargs)
        fuel_flow_kgh = estimates['fuel_flow_kgh'].values
    except Exception as e:
        logger.warning(f"Acropole estimation failed for {aircraft_type}: {e}")
        return np.full(n_points, np.nan), np.full(n_points, np.nan), np.nan
    
    dt_hours = dt_seconds / 3600.0
    
    # Track mass and calculate total fuel
    mass_kg = np.full(n_points, np.nan)
    current_mass = tow_kg
    mass_kg[0] = current_mass
    total_fuel = 0.0
    
    for i in range(1, n_points):
        fuel_burned = fuel_flow_kgh[i-1] * dt_hours[i]
        total_fuel += fuel_burned
        current_mass = max(current_mass - fuel_burned, 
                          tow_kg * CONFIG['min_mass_fraction'])
        mass_kg[i] = current_mass
    
    return mass_kg, fuel_flow_kgh, total_fuel


def process_single_flight(
    flight_id: str,
    aircraft_type: str,
    tow_kg: float,
    traj_dir: Path,
    output_dir: Path,
    enhanced_params_path: str
) -> Tuple[str, bool, str, Optional[float]]:
    """
    Process fuel calculation for a single flight.
    
    Returns:
        (flight_id, success, error_message, total_fuel_kg)
    """
    try:
        # Input/output paths
        traj_file = traj_dir / f"{flight_id}.parquet"
        output_file = output_dir / f"{flight_id}.parquet"
        
        if not traj_file.exists():
            return (flight_id, False, "Trajectory file not found", None)
        
        # Skip if already processed
        if output_file.exists():
            # Load existing file to get total fuel
            try:
                existing_df = pd.read_parquet(output_file)
                if 'fuel_flow_acropole_kgh' in existing_df.columns and 'tow_kg' in existing_df.columns:
                    # Recalculate total fuel from existing trajectory
                    if 'timestamp' in existing_df.columns:
                        timestamps = pd.to_datetime(existing_df['timestamp'])
                        dt_seconds = timestamps.diff().dt.total_seconds().fillna(1.0).values
                    else:
                        dt_seconds = np.full(len(existing_df), 1.0)
                    
                    dt_hours = dt_seconds / 3600.0
                    fuel_flow = existing_df['fuel_flow_acropole_kgh'].values
                    total_fuel = np.sum(fuel_flow[:-1] * dt_hours[1:])
                    
                    return (flight_id, True, "Already processed (loaded)", total_fuel)
            except:
                pass
        
        # Load trajectory
        traj_df = pd.read_parquet(traj_file)
        
        if len(traj_df) < 10:
            return (flight_id, False, "Trajectory too short", None)
        
        # Validate timestamp order if timestamps exist
        if 'timestamp' in traj_df.columns:
            timestamps = pd.to_datetime(traj_df['timestamp'])
            if not timestamps.is_monotonic_increasing:
                logger.warning(f"{flight_id}: Timestamps not in order, sorting...")
                traj_df = traj_df.sort_values('timestamp').reset_index(drop=True)
        
        # Initialize FuelEstimator in worker process
        fe = FuelEstimator(aircraft_params_path=enhanced_params_path)
        
        # Calculate Acropole fuel
        mass_acropole_kg, ff_acropole_kgh, total_fuel_kg = calculate_acropole_fuel(
            traj_df, aircraft_type, tow_kg, fe
        )
        
        # Check if calculation was successful
        if np.isnan(total_fuel_kg):
            return (flight_id, False, "Acropole calculation failed", None)
        
        # Add fuel features to dataframe
        traj_df['mass_acropole_kg'] = mass_acropole_kg
        traj_df['fuel_flow_acropole_kgh'] = ff_acropole_kgh
        traj_df['tow_kg'] = tow_kg
        traj_df['aircraft_type'] = aircraft_type
        
        # Save enhanced trajectory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        traj_df.to_parquet(output_file, index=False)
        
        return (flight_id, True, "Success", total_fuel_kg)
        
    except Exception as e:
        return (flight_id, False, f"Error: {str(e)}", None)


def load_flight_metadata(dataset: str) -> Optional[pd.DataFrame]:
    """Load flight metadata from flightlist (aircraft type) and TOW predictions"""
    config = DATASETS[dataset]
    
    # Load flightlist
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


def process_dataset(dataset: str, enhanced_params_path: str) -> Dict:
    """
    Process all flights in a dataset.
    
    Returns:
        Statistics dictionary with fuel predictions
    """
    logger.info(f"Processing {dataset} dataset...")
    
    config = DATASETS[dataset]
    
    # Load metadata
    metadata = load_flight_metadata(dataset)
    if metadata is None:
        return {'error': 'Failed to load metadata'}
    
    # Filter metadata to only include flights with existing trajectory files
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
    
    # Prepare output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'total': len(metadata),
        'success': 0,
        'failed': 0,
        'errors': [],
        'fuel_predictions': []
    }
    
    # Process flights in parallel
    with ProcessPoolExecutor(max_workers=15) as executor:
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
                flight_id, success, message, total_fuel = future.result()
                
                if success:
                    stats['success'] += 1
                    stats['fuel_predictions'].append({
                        'flight_id': flight_id,
                        'fuel_kg': total_fuel
                    })
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
    logger.info(f"  Failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        logger.warning(f"  First 10 errors:")
        for error in stats['errors'][:10]:
            logger.warning(f"    {error['flight_id']}: {error['error']}")
    
    return stats


def fill_missing_with_knn(predictions_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing fuel predictions using KNN based on aircraft type and TOW
    
    Args:
        predictions_df: DataFrame with flight_id and fuel_kg
        metadata_df: DataFrame with flight_id, aircraft_type, tow
    
    Returns:
        DataFrame with missing values filled
    """
    logger.info("Filling missing predictions with KNN...")
    
    # Merge predictions with metadata
    df = pd.merge(metadata_df, predictions_df, on='flight_id', how='left')
    
    # Identify missing predictions
    missing_mask = df['fuel_kg'].isna()
    n_missing = missing_mask.sum()
    
    if n_missing == 0:
        logger.info("No missing predictions to fill!")
        return df
    
    logger.info(f"Found {n_missing} missing predictions ({100*n_missing/len(df):.2f}%)")
    
    # Encode aircraft type
    aircraft_types = df['aircraft_type'].unique()
    aircraft_type_map = {ac: i for i, ac in enumerate(aircraft_types)}
    df['aircraft_type_encoded'] = df['aircraft_type'].map(aircraft_type_map)
    
    # Prepare features for KNN
    features = ['aircraft_type_encoded', 'tow']
    
    # Scale features
    scaler = StandardScaler()
    X = df[features].values
    X_scaled = scaler.fit_transform(X)
    
    # Split into train (non-missing) and predict (missing)
    X_train = X_scaled[~missing_mask]
    y_train = df.loc[~missing_mask, 'fuel_kg'].values
    X_predict = X_scaled[missing_mask]
    
    # Train KNN
    knn = KNeighborsRegressor(n_neighbors=min(CONFIG['knn_neighbors'], len(X_train)))
    knn.fit(X_train, y_train)
    
    # Predict missing values
    y_predict = knn.predict(X_predict)
    
    # Fill missing values
    df.loc[missing_mask, 'fuel_kg'] = y_predict
    
    logger.info(f"Filled {n_missing} missing predictions using KNN")
    logger.info(f"  Mean filled value: {y_predict.mean():.2f} kg")
    logger.info(f"  Std filled value: {y_predict.std():.2f} kg")
    
    return df[['flight_id', 'fuel_kg']]


def calculate_rmse_with_actual(predictions_df: pd.DataFrame, challenge_file: Path) -> Dict:
    """
    Calculate RMSE between predictions and actual fuel consumption
    
    Returns:
        Dictionary with RMSE and other metrics
    """
    logger.info("Calculating RMSE with actual fuel consumption...")
    
    if not challenge_file.exists():
        logger.error(f"Challenge file not found: {challenge_file}")
        return {'error': 'Challenge file not found'}
    
    # Load actual fuel consumption
    challenge_df = pd.read_parquet(challenge_file)
    logger.info(f"Loaded {len(challenge_df)} actual fuel values")
    
    # Aggregate actual fuel by flight_id (sum across all trajectory points)
    actual_by_flight = challenge_df.groupby('flight_id')['fuel_kg'].sum().reset_index()
    actual_by_flight.rename(columns={'fuel_kg': 'actual_fuel_kg'}, inplace=True)
    
    # Merge predictions with actual
    comparison_df = pd.merge(
        actual_by_flight,
        predictions_df[['flight_id', 'fuel_kg']],
        on='flight_id',
        how='inner'
    )
    comparison_df.rename(columns={'fuel_kg': 'predicted_fuel_kg'}, inplace=True)
    
    logger.info(f"Matched {len(comparison_df)} flights for comparison")
    
    if len(comparison_df) == 0:
        logger.error("No flights matched between predictions and actual!")
        return {'error': 'No matches found'}
    
    # Calculate metrics
    actual = comparison_df['actual_fuel_kg'].values
    predicted = comparison_df['predicted_fuel_kg'].values
    
    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # MAE
    mae = np.mean(np.abs(actual - predicted))
    
    # MAPE
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # R²
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'n_flights': len(comparison_df),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'mean_actual': float(actual.mean()),
        'mean_predicted': float(predicted.mean()),
        'std_actual': float(actual.std()),
        'std_predicted': float(predicted.std())
    }
    
    logger.info(f"\n{'='*80}")
    logger.info("RMSE COMPARISON WITH ACTUAL FUEL")
    logger.info(f"{'='*80}")
    logger.info(f"Number of flights: {metrics['n_flights']}")
    logger.info(f"RMSE: {metrics['rmse']:.2f} kg")
    logger.info(f"MAE: {metrics['mae']:.2f} kg")
    logger.info(f"MAPE: {metrics['mape']:.2f}%")
    logger.info(f"R²: {metrics['r2']:.4f}")
    logger.info(f"\nMean actual fuel: {metrics['mean_actual']:.2f} kg")
    logger.info(f"Mean predicted fuel: {metrics['mean_predicted']:.2f} kg")
    logger.info(f"Prediction bias: {metrics['mean_predicted'] - metrics['mean_actual']:.2f} kg")
    logger.info(f"{'='*80}\n")
    
    # Save detailed comparison
    output_file = WORKSPACE_ROOT / "logs" / "acropole_fuel_comparison.csv"
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"Detailed comparison saved to: {output_file}")
    
    return metrics


def create_submission_file(predictions_df: pd.DataFrame, dataset: str):
    """Create submission file for ranking/final datasets"""
    logger.info(f"Creating submission file for {dataset} dataset...")
    
    # Prepare submission format
    submission_df = predictions_df[['flight_id', 'fuel_kg']].copy()
    submission_df.columns = ['flight_id', 'tow']  # Follow submission format
    
    # Save submission
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    submission_file = SUBMISSION_ROOT / f"acropole_only_submission_{dataset}.csv"
    submission_df.to_csv(submission_file, index=False)
    
    logger.info(f"Submission file saved to: {submission_file}")
    logger.info(f"  Flights: {len(submission_df)}")
    logger.info(f"  Mean fuel: {submission_df['tow'].mean():.2f} kg")
    logger.info(f"  Std fuel: {submission_df['tow'].std():.2f} kg")


def main():
    """Main processing pipeline"""
    logger.info("=" * 80)
    logger.info("Acropole-Only Fuel Calculation with KNN Gap Filling")
    logger.info("=" * 80)
    
    # Create directories
    (WORKSPACE_ROOT / "logs").mkdir(exist_ok=True)
    
    # Verify enhanced aircraft params exist
    logger.info("Verifying enhanced aircraft parameters...")
    if not ENHANCED_PARAMS.exists():
        logger.error(f"Enhanced params file not found: {ENHANCED_PARAMS}")
        return
    
    # Test load
    fe_test = FuelEstimator(aircraft_params_path=str(ENHANCED_PARAMS))
    logger.info(f"Loaded enhanced params for {len(fe_test.aircraft_params)} aircraft types")
    del fe_test
    
    # Process datasets
    all_stats = {}
    all_metrics = {}
    
    for dataset in ['train', 'rank', 'final']:
        logger.info(f"\n{'='*80}")
        
        # Process flights
        stats = process_dataset(dataset, str(ENHANCED_PARAMS))
        if 'error' in stats:
            logger.error(f"{dataset}: {stats['error']}")
            continue
        
        all_stats[dataset] = stats
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame(stats['fuel_predictions'])
        
        # Load metadata for KNN filling
        metadata = load_flight_metadata(dataset)
        
        # Fill missing predictions with KNN
        if len(stats['errors']) > 0 and metadata is not None:
            predictions_df = fill_missing_with_knn(predictions_df, metadata)
        
        # For training data, calculate RMSE with actual
        if dataset == 'train':
            challenge_file = DATASETS[dataset]['challenge_file']
            metrics = calculate_rmse_with_actual(predictions_df, challenge_file)
            all_metrics['train'] = metrics
        
        # Create submission file for rank and final datasets
        if dataset in ['rank', 'final']:
            create_submission_file(predictions_df, dataset)
    
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
            logger.info(f"  Failed: {stats['failed']}")
            
            if dataset in all_metrics:
                logger.info(f"  RMSE: {all_metrics[dataset]['rmse']:.2f} kg")
                logger.info(f"  MAE: {all_metrics[dataset]['mae']:.2f} kg")
                logger.info(f"  R²: {all_metrics[dataset]['r2']:.4f}")
    
    # Save summary
    summary = {
        'stats': {k: {key: val for key, val in v.items() if key != 'fuel_predictions'} 
                  for k, v in all_stats.items()},
        'metrics': all_metrics
    }
    
    summary_file = WORKSPACE_ROOT / "logs" / "acropole_only_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
