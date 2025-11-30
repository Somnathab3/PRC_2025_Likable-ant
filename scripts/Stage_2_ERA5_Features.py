#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2: ERA5 Feature Integration (Optimized Single-Day Processing)
====================================================================

This optimized version processes ERA5 data ONE DATE AT A TIME to minimize memory usage:
- Loads ERA5 data for ONLY the current date (no 3-day window)
- Filters flights to only those that have data on the current date
- Splits multi-day flights at midnight boundaries
- Processes date-specific segments and merges them back

Benefits:
- 3x less ERA5 data in memory (1 day vs 3 days)
- Faster KDTree interpolation (smaller search space)
- Better parallelization (independent dates)
- Cleaner code (no cross-day interpolation complexity)

Input: data/processed/trajectories_unified/{dataset}/*.parquet
       flight_date_segments.parquet (date-flight mapping)
Output: data/processed/trajectories_era5/{dataset}/*.parquet

Usage:
    python Stage_2_ERA5_Features_Optimized.py --dataset train --date 2025-04-15
    python Stage_2_ERA5_Features_Optimized.py --dataset train --start_date 2025-04-10 --end_date 2025-04-15
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
ERA5_TEMP_DIR = Path("D:/Projects/PRC_2025_ERA5/temp")
PROCESSED_DIR = BASE_DIR / "data/processed"

# Pipeline stage directories
STAGE_1_OUTPUT = PROCESSED_DIR / "trajectories_unified"
STAGE_2_OUTPUT = PROCESSED_DIR / "trajectories_era5"

# Mapping file
FLIGHT_DATE_SEGMENTS = BASE_DIR / "flight_date_segments.parquet"

# Variable mapping
VAR_NAME_MAPPING = {
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'temperature': 't',
    'specific_humidity': 'q'
}

PRESSURE_VARS = [
    'u_component_of_wind',
    'v_component_of_wind',
    'temperature',
    'specific_humidity'
]


def pressure_hpa_from_altitude_ft(alt_ft):
    """Convert altitude (feet) to pressure (hPa) using ISA formula."""
    z = np.array(alt_ft, dtype=float) * 0.3048
    p0, T0, g, L, R = 101325.0, 288.15, 9.80665, 0.0065, 287.05
    trop = z <= 11000
    p = np.empty_like(z, dtype=float)
    p[trop] = p0 * (1 - L * z[trop] / T0) ** (g / (R * L))
    p11 = p0 * (1 - L * 11000 / T0) ** (g / (R * L))
    T11 = T0 - L * 11000
    p[~trop] = p11 * np.exp(-(g * (z[~trop] - 11000)) / (R * T11))
    return p / 100.0


def load_era5_data_for_date(date: datetime, era5_temp_dir: Path) -> Dict[str, xr.DataArray]:
    """
    Load ERA5 data for a SINGLE date (not 3-day window).
    
    Args:
        date: Date to load
        era5_temp_dir: ERA5 data directory
    
    Returns:
        Dictionary of variable_name -> DataArray
    """
    date_str = date.strftime('%Y%m%d')
    date_dir = era5_temp_dir / date_str
    
    if not date_dir.exists():
        print(f"Warning: No ERA5 data for {date_str}")
        return {}
    
    data_dict = {}
    
    for var_name in PRESSURE_VARS:
        var_dir = date_dir / var_name
        
        if not var_dir.exists():
            print(f"Warning: No {var_name} data for {date_str}")
            continue
        
        nc_files = list(var_dir.glob("*.nc"))
        if not nc_files:
            nc_files = list(var_dir.glob("*.tmp"))
            if not nc_files:
                print(f"Warning: No NetCDF files for {var_name} on {date_str}")
                continue
        
        try:
            levels = [int(f.stem.replace('.tmp', '')) for f in nc_files]
            sorted_pairs = sorted(zip(levels, nc_files), key=lambda x: x[0], reverse=True)
            levels = [pair[0] for pair in sorted_pairs]
            nc_files = [pair[1] for pair in sorted_pairs]
            
            ds = xr.open_mfdataset(
                nc_files,
                combine='nested',
                concat_dim='level',
                engine='netcdf4',
                coords='minimal',
                compat='override'
            )
            
            ds = ds.assign_coords(level=('level', levels))
            nc_var_name = VAR_NAME_MAPPING.get(var_name, var_name)
            
            if nc_var_name in ds.data_vars:
                data_dict[var_name] = ds[nc_var_name].load()
                print(f"  [OK] {var_name}: {len(levels)} levels ({min(levels)}-{max(levels)} hPa)")
            else:
                print(f"Warning: {nc_var_name} not in dataset")
        
        except Exception as e:
            print(f"Error loading {var_name} for {date_str}: {e}")
            continue
    
    return data_dict


def interpolate_era5_kdtree(var_name: str, var_data: xr.DataArray,
                            lat: np.ndarray, lon: np.ndarray,
                            tim: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
    """
    Fast vectorized ERA5 interpolation using KDTree (spatial) + log-pressure (vertical).
    
    Args:
        var_name: Variable name
        var_data: DataArray with dims (level, time, latitude, longitude)
        lat, lon, tim, pressure_hpa: Trajectory point arrays
    
    Returns:
        Interpolated values
    """
    if var_data is None or var_data.size == 0:
        return np.full(len(lat), np.nan, dtype=np.float32)
    
    try:
        # Convert longitude to [0, 360]
        lon_360 = np.where(lon < 0, lon + 360, lon)
        
        # Convert timestamps
        tim_naive = pd.to_datetime(tim)
        if isinstance(tim_naive, pd.DatetimeIndex) and tim_naive.tz is not None:
            tim_naive = tim_naive.tz_localize(None)
        
        n_points = len(lat)
        result = np.full(n_points, np.nan, dtype=np.float32)
        
        # Build spatial KDTree
        lon_grid = var_data.longitude.values
        lat_grid = var_data.latitude.values
        lon_mg, lat_mg = np.meshgrid(lon_grid, lat_grid)
        grid_points = np.column_stack([lat_mg.ravel(), lon_mg.ravel()])
        tree = cKDTree(grid_points)
        
        # Pressure levels (descending order: 1000 -> 150 hPa)
        levels = var_data.level.values
        
        # Process by unique hour
        unique_hours = pd.to_datetime(tim_naive).floor('h').unique()
        
        for hour in unique_hours:
            hour_mask = (pd.to_datetime(tim_naive).floor('h') == hour)
            if not np.any(hour_mask):
                continue
            
            indices = np.where(hour_mask)[0]
            lat_hour = lat[hour_mask]
            lon_hour = lon_360[hour_mask]
            p_hour = pressure_hpa[hour_mask]
            
            # Spatial nearest neighbors
            query_points = np.column_stack([lat_hour, lon_hour])
            _, nn_idx = tree.query(query_points, k=1)
            iy, ix = np.divmod(nn_idx, lon_grid.size)
            
            # Time index
            try:
                time_idx = var_data.time.to_index().get_indexer([hour], method='nearest')[0]
                if time_idx < 0 or time_idx >= len(var_data.time):
                    continue
            except:
                continue
            
            # Pressure interpolation indices
            p_clipped = np.clip(p_hour, levels.min(), levels.max())
            j_hi = np.searchsorted(-levels, -p_clipped, side='right') - 1  # Higher pressure (warmer)
            j_hi = np.clip(j_hi, 0, len(levels) - 1)
            j_lo = np.clip(j_hi + 1, 0, len(levels) - 1)  # Lower pressure (colder)
            
            # Extract data - OPTIMIZED direct numpy indexing
            try:
                vals_at_hi_p = var_data.values[j_hi, time_idx, iy, ix]
                vals_at_lo_p = var_data.values[j_lo, time_idx, iy, ix]
                
                # Vectorized vertical interpolation
                exact_match = (j_hi == j_lo)
                result[indices[exact_match]] = vals_at_hi_p[exact_match]
                
                interp_mask = ~exact_match
                if np.any(interp_mask):
                    p_hi = levels[j_hi[interp_mask]]
                    p_lo = levels[j_lo[interp_mask]]
                    
                    log_denom = np.log(p_lo) - np.log(p_hi)
                    safe_denom = np.where(np.abs(log_denom) > 1e-12, log_denom, 1.0)
                    weight = (np.log(p_clipped[interp_mask]) - np.log(p_hi)) / safe_denom
                    weight = np.clip(weight, 0.0, 1.0)
                    
                    result[indices[interp_mask]] = \
                        vals_at_hi_p[interp_mask] * (1 - weight) + \
                        vals_at_lo_p[interp_mask] * weight
            
            except Exception as e:
                continue
        
        return result
    
    except Exception as e:
        print(f"      Warning: KDTree interpolation failed for {var_name}: {e}")
        return np.full(len(lat), np.nan, dtype=np.float32)


def process_flight_for_date(flight_file: Path, date: datetime, era5_data: Dict,
                            segment_info: pd.Series, output_file: Path,
                            skip_existing: bool) -> Tuple[str, str, str]:
    """
    Process a single flight for a specific date (or date segment).
    
    Args:
        flight_file: Path to trajectory parquet
        date: Date being processed
        era5_data: ERA5 data dictionary
        segment_info: Row from flight_date_segments with time bounds
        output_file: Output path
        skip_existing: Whether to skip if output exists
    
    Returns:
        (status, flight_id, message)
    """
    flight_id = flight_file.stem
    
    # Check if THIS specific date segment temp file already exists
    if skip_existing and segment_info is not None:
        date_str = date.strftime('%Y%m%d')
        temp_filename = f"{flight_id}_{date_str}.parquet"
        temp_file = output_file.parent / "temp_segments" / temp_filename
        
        if temp_file.exists():
            try:
                existing_df = pd.read_parquet(temp_file)
                if len(existing_df) > 0:
                    return ('skipped', flight_id, f'Date segment {date_str} already processed')
            except:
                # If error reading temp file, delete and reprocess
                temp_file.unlink()
    
    try:
        # Load full trajectory
        df = pd.read_parquet(flight_file)
        
        if len(df) == 0:
            return ('failed', flight_id, 'Empty trajectory')
        
        # Convert timestamps
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        df = df.dropna(subset=['latitude', 'longitude', 'altitude', 'timestamp'])
        
        if len(df) == 0:
            return ('failed', flight_id, 'No valid data')
        
        # Filter to date segment (if segment_info provided)
        if segment_info is not None:
            segment_start = pd.Timestamp(segment_info['segment_start'])
            segment_end = pd.Timestamp(segment_info['segment_end'])
            df = df[(df['timestamp'] >= segment_start) & (df['timestamp'] <= segment_end)]
        
        if len(df) == 0:
            return ('failed', flight_id, 'No data in date segment')
        
        # Convert altitude to pressure
        pressure_hpa = pressure_hpa_from_altitude_ft(df['altitude'].values)
        df['pressure_hpa'] = pressure_hpa
        
        # Prepare arrays
        lat = np.asarray(df['latitude'].values, dtype=np.float64)
        lon = np.asarray(df['longitude'].values, dtype=np.float64)
        tim = df['timestamp'].values
        
        # Interpolate ERA5 variables
        for var_name in PRESSURE_VARS:
            if var_name in era5_data:
                interpolated = interpolate_era5_kdtree(
                    var_name, era5_data[var_name], lat, lon, tim, pressure_hpa
                )
                df[f'{var_name}_pl'] = interpolated
            else:
                df[f'{var_name}_pl'] = np.nan
        
        # Save to temporary folder with date-specific filename
        # Format: flightid_YYYYMMDD.parquet
        date_str = date.strftime('%Y%m%d')
        temp_filename = f"{flight_id}_{date_str}.parquet"
        temp_file = output_file.parent / "temp_segments" / temp_filename
        
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(temp_file, index=False)
        
        return ('success', flight_id, f'{len(df)} points')
    
    except Exception as e:
        return ('failed', flight_id, str(e))


def process_date_optimized(date: datetime, dataset: str, era5_temp_dir: Path,
                           input_dir: Path, output_dir: Path,
                           flight_segments_df: pd.DataFrame,
                           skip_existing: bool = True) -> Dict:
    """
    Process all flights for a specific date (SINGLE-DAY processing).
    
    Args:
        date: Date to process
        dataset: Dataset name
        era5_temp_dir: ERA5 data directory
        input_dir: Input trajectory directory
        output_dir: Output directory
        flight_segments_df: DataFrame with flight-date segments
        skip_existing: Skip processed flights
    
    Returns:
        Processing statistics
    """
    print(f"\n{'='*70}")
    print(f"Processing {date.strftime('%Y-%m-%d')} [{dataset.upper()}]")
    print(f"{'='*70}")
    
    # Load ERA5 for THIS DATE ONLY
    print(f"Loading ERA5 data for {date.strftime('%Y-%m-%d')}...")
    era5_data = load_era5_data_for_date(date, era5_temp_dir)
    
    if not era5_data:
        print(f"⚠ No ERA5 data for {date.strftime('%Y-%m-%d')}")
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    # Filter to flights on this date
    date_str = date.strftime('%Y-%m-%d')
    date_flights = flight_segments_df[
        (flight_segments_df['date'] == date_str) &
        (flight_segments_df['dataset'] == dataset)
    ]
    
    if len(date_flights) == 0:
        print(f"No flights for {date_str}")
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    print(f"\nProcessing {len(date_flights)} flight segments...")
    
    results = {'success': 0, 'failed': 0, 'skipped': 0, 'failed_flights': []}
    
    for _, segment in tqdm(date_flights.iterrows(), total=len(date_flights),
                           desc="Processing flights"):
        flight_id = segment['flight_id']
        flight_file = input_dir / f"{flight_id}.parquet"
        output_file = output_dir / f"{flight_id}.parquet"
        
        if not flight_file.exists():
            results['failed'] += 1
            results['failed_flights'].append((flight_id, 'File not found'))
            continue
        
        status, fid, msg = process_flight_for_date(
            flight_file, date, era5_data, segment, output_file, skip_existing
        )
        
        if status == 'success':
            results['success'] += 1
        elif status == 'skipped':
            results['skipped'] += 1
        else:
            results['failed'] += 1
            results['failed_flights'].append((fid, msg))
    
    print(f"\n✓ Complete:")
    print(f"  Success: {results['success']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Failed: {results['failed']}")
    
    if results['failed_flights']:
        print(f"\nFirst 10 failures:")
        for fid, msg in results['failed_flights'][:10]:
            print(f"  {fid}: {msg}")
    
    return results


def merge_flight_segments(output_dir: Path, dataset: str):
    """
    Merge temporary date-specific segments into final flight files.
    
    Args:
        output_dir: Output directory for dataset
        dataset: Dataset name
    """
    temp_dir = output_dir / "temp_segments"
    
    if not temp_dir.exists():
        print("No temporary segments to merge")
        return
    
    # Get all temporary segment files
    temp_files = list(temp_dir.glob("*.parquet"))
    
    if not temp_files:
        print("No temporary segments found")
        return
    
    print(f"Found {len(temp_files)} temporary segment files")
    
    # Group by flight_id (extract from filename: flightid_YYYYMMDD.parquet)
    from collections import defaultdict
    flight_segments = defaultdict(list)
    
    for temp_file in temp_files:
        # Parse filename: flightid_YYYYMMDD.parquet
        name = temp_file.stem  # Remove .parquet
        parts = name.rsplit('_', 1)  # Split from right, max 1 split
        
        if len(parts) == 2:
            flight_id, date_str = parts
            flight_segments[flight_id].append(temp_file)
        else:
            print(f"⚠ Warning: Unexpected filename format: {temp_file.name}")
    
    print(f"Merging segments for {len(flight_segments)} flights...")
    
    merged_count = 0
    single_count = 0
    failed_count = 0
    duplicate_rows_removed = 0
    
    for flight_id, segment_files in tqdm(flight_segments.items(), desc="Merging flights"):
        try:
            if len(segment_files) == 1:
                # Single segment - just copy to output
                segment_df = pd.read_parquet(segment_files[0])
                output_file = output_dir / f"{flight_id}.parquet"
                segment_df.to_parquet(output_file, index=False)
                single_count += 1
            else:
                # Multiple segments - merge them
                dfs = []
                for seg_file in sorted(segment_files):  # Sort to maintain order
                    df = pd.read_parquet(seg_file)
                    dfs.append(df)
                
                # Concatenate all segments
                merged_df = pd.concat(dfs, ignore_index=True)
                
                # Sort by timestamp
                merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
                
                # Check for and remove duplicate rows
                initial_len = len(merged_df)
                
                # Remove duplicates based on timestamp (keep first occurrence)
                merged_df = merged_df.drop_duplicates(subset=['timestamp'], keep='first')
                
                duplicates_removed = initial_len - len(merged_df)
                if duplicates_removed > 0:
                    duplicate_rows_removed += duplicates_removed
                
                # Save merged file
                output_file = output_dir / f"{flight_id}.parquet"
                merged_df.to_parquet(output_file, index=False)
                merged_count += 1
                
        except Exception as e:
            print(f"⚠ Failed to merge {flight_id}: {e}")
            failed_count += 1
    
    print(f"\n✓ Merge complete:")
    print(f"  Single-segment flights: {single_count}")
    print(f"  Multi-segment flights merged: {merged_count}")
    print(f"  Failed: {failed_count}")
    if duplicate_rows_removed > 0:
        print(f"  Duplicate rows removed: {duplicate_rows_removed}")
    
    # Clean up temporary directory
    print(f"\nCleaning up temporary segments...")
    for temp_file in temp_files:
        temp_file.unlink()
    
    try:
        temp_dir.rmdir()
        print(f"✓ Removed temporary directory: {temp_dir}")
    except:
        print(f"⚠ Could not remove temporary directory: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: ERA5 Feature Integration (Optimized Single-Day)'
    )
    
    parser.add_argument('--dataset', choices=['train', 'rank', 'final', 'all'],
                       default='train', help='Dataset to process')
    parser.add_argument('--date', help='Single date to process (YYYY-MM-DD)')
    parser.add_argument('--start_date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess all flights')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    # Parse dates
    if args.date:
        dates = [datetime.strptime(args.date, '%Y-%m-%d')]
    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = datetime.strptime(args.end_date, '%Y-%m-%d')
        dates = pd.date_range(start, end, freq='D').to_pydatetime().tolist()
    else:
        print("Error: Must provide --date or --start_date + --end_date")
        sys.exit(1)
    
    # Load flight-date segments mapping
    if not FLIGHT_DATE_SEGMENTS.exists():
        print(f"Error: Flight-date segments not found: {FLIGHT_DATE_SEGMENTS}")
        print("Run: python build_flight_date_mapping.py --dataset all")
        sys.exit(1)
    
    print(f"Loading flight-date segments...")
    flight_segments = pd.read_parquet(FLIGHT_DATE_SEGMENTS)
    print(f"  Loaded {len(flight_segments)} segments")
    
    datasets = ['train', 'rank', 'final'] if args.dataset == 'all' else [args.dataset]
    skip_existing = not args.no_skip
    n_workers = min(args.workers, len(dates))  # Don't use more workers than dates
    
    print(f"\n{'#'*70}")
    print(f"# ERA5 Feature Integration (Optimized)")
    print(f"{'#'*70}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Dates: {len(dates)} ({dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')})")
    print(f"ERA5 directory: {ERA5_TEMP_DIR}")
    print(f"Skip existing: {skip_existing}")
    print(f"Parallel workers: {n_workers}")
    print(f"{'#'*70}\n")
    
    # Process each dataset
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*70}")
        
        input_dir = STAGE_1_OUTPUT / dataset
        output_dir = STAGE_2_OUTPUT / dataset
        
        if not input_dir.exists():
            print(f"⚠ Input directory not found: {input_dir}")
            continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Process dates in parallel
        if n_workers > 1 and len(dates) > 1:
            print(f"\nProcessing {len(dates)} dates with {n_workers} parallel workers...")
            
            # Prepare arguments for parallel processing
            process_args = [
                (date, dataset, ERA5_TEMP_DIR, input_dir, output_dir,
                 flight_segments, skip_existing)
                for date in dates
            ]
            
            # Use spawn method to avoid memory issues on Windows
            mp.set_start_method('spawn', force=True)
            
            with Pool(processes=n_workers) as pool:
                results = pool.starmap(process_date_optimized, process_args)
            
            # Aggregate results
            for stats in results:
                total_stats['success'] += stats['success']
                total_stats['failed'] += stats['failed']
                total_stats['skipped'] += stats['skipped']
        else:
            # Sequential processing (single worker or single date)
            print(f"\nProcessing {len(dates)} dates sequentially...")
            for date in dates:
                stats = process_date_optimized(
                    date, dataset, ERA5_TEMP_DIR, input_dir, output_dir,
                    flight_segments, skip_existing
                )
                
                total_stats['success'] += stats['success']
                total_stats['failed'] += stats['failed']
                total_stats['skipped'] += stats['skipped']
        
        print(f"\n{'='*70}")
        print(f"DATASET {dataset.upper()} COMPLETE")
        print(f"{'='*70}")
        print(f"Total success: {total_stats['success']}")
        print(f"Total skipped: {total_stats['skipped']}")
        print(f"Total failed: {total_stats['failed']}")
        
        # MERGE temporary segments into final flight files
        print(f"\n{'='*70}")
        print(f"MERGING SEGMENTS FOR {dataset.upper()}")
        print(f"{'='*70}")
        merge_flight_segments(output_dir, dataset)
    
    print(f"\n{'#'*70}")
    print(f"# ALL PROCESSING COMPLETE")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
