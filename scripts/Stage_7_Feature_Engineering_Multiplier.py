"""
Stage 7: Multiplier-Aware Feature Engineering (Acropole + OpenAP)
=================================================================

Extracts interval-level features from trajectories enhanced with
per-aircraft multipliers (Stage 6.3 + multiplier correction):
1. Loads Stage 6.3 trajectories that already include multiplier outputs
2. Computes flight-/interval-/segment-level features (200+ columns)
3. Adds multiplier statistics and corrected fuel flow integrations
4. Consolidates all per-flight features into single dataset files

FEATURE CATEGORIES (200+ features):

Flight-Level Features:
  - Aircraft metadata (type, origin, destination)
  - TOW predictions and kinematic features
  - Flight timing and duration
  - Altitude, speed, vertical rate statistics
  - Flight phase time allocations (climb/cruise/descent)
  - Total fuel consumption (Acropole, OpenAP)
  - Acropole vs OpenAP comparison (fuel diff, ratio, % error)
  - METAR weather data (origin/destination airports at takeoff/landing)
  - METAR differences (destination - origin)

Interval-Level Features:
  - Temporal positioning (relative position, time since takeoff)
  - Altitude features (start, end, change, statistics)
  - Speed features (groundspeed, TAS statistics)
  - Vertical rate statistics
  - Flight phase distribution (climb/cruise/descent fractions)
  - Mass features (Acropole, OpenAP, start/end/change)
  - Fuel flow statistics (Acropole, OpenAP)
  - Fuel consumption (integrated Acropole, OpenAP)
  
OpenAP Features:
  - Thrust statistics (min, max, mean, median, std, p10, p90, iqr)
  - Drag statistics (min, max, mean, median, std, p10, p90, iqr)
  - Net force (thrust - drag)
  - Thrust-drag ratio

Acropole vs OpenAP Comparisons:
  - Mass comparison (diff, ratio, % error)
  - Fuel flow comparison (diff, ratio, % error)
  - Fuel consumed comparison (diff, ratio, % error)
  - Squared differences, agreement indicators

Advanced Interaction Features:
  - Thrust power (thrust × TAS)
  - Drag power (drag × TAS)
  - Net power (thrust power - drag power)
  - Fuel efficiency metrics (fuel/distance, fuel/time, specific range)
  - Fuel normalized by mass and TOW
  - Energy change features (PE, KE, total energy)
  - Power proxies

Atmosphere & Environment:
  - ISA deviation (mean, std, min, max)
  - Air density (mean, std, min, max, sigma)
  - Dynamic pressure (qbar)
  - Specific humidity
  - Wind features (speed, headwind, crosswind, shear)

Variability & Smoothness Features:
  - TAS acceleration patterns (mean, std, max, min)
  - Altitude rate changes
  - Vertical rate variability
  - Fuel flow variability (Acropole, OpenAP)
  - Coefficient of variation metrics

Energy & Momentum Features:
  - Speed × altitude interaction (energy state proxy)
  - Flight path angle (FPA)
  - Climb-descent ratio
  - Weighted cruise altitude

Segment-Level Features (per 20 segments):
  - Altitude, groundspeed, vertical rate means
  - Fuel flow (Acropole, OpenAP)
  - Mass (Acropole, OpenAP)
  - Thrust, drag (OpenAP)
  - Atmosphere: density, dynamic pressure (qbar), temperature
  - Wind: headwind, crosswind (projected onto track)
  - Geometry: track mean/std, flight path angle (FPA)
  - Motion: TAS acceleration
  - Efficiency ratios: TW, DW, SFC (specific fuel consumption)
  - Fuel flow differences and ratios (Acropole vs OpenAP)

Cross-Segment Shape Features:
  - Fuel distribution: share in first/last half, center of mass index
  - Altitude profile: linear trend slope, peak/valley segments
  - Multiplier variability: std and range across segments

Cross-Segment Transition Features:
  - Delta groundspeed between consecutive segments
  - Delta altitude between consecutive segments
  - Acceleration/deceleration patterns across segments
  - Thrust power per segment (thrust × speed)
  - Drag power per segment (drag × speed)
  - Aggregate thrust/drag power (total, mean, std)
  - Fuel flow variability across segments (std, range, CV)
  - Vertical rate variability across segments
  - Mass variability across segments

Data Quality Features:
  - Fraction of reconstructed points
  - Sampling quality (mean dt, dt std)
  - Number of trajectory points

Temporal Features:
  - Time of day (cyclical encoding: sin, cos)
  - Day of week

Output: 
  - Per-flight parquet files in Stage_7_Multiplier_features/{dataset}/
  - Consolidated parquet files in Stage_8_Consolidated_Features/ (all intervals for each dataset)


Author: PRC 2025
Date: November 22, 2025
"""

import sys
import warnings
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import pandas as pd
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

# Setup paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

# Data paths
DATA_ROOT = WORKSPACE_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"

# Input directories (Stage 6.3 trajectories augmented with multipliers)
TRAJ_MULTIPLIER_DIR = PROCESSED_ROOT / "Stage_6.3_Multiplier_Trajectory"
TOW_PRED_DIR = PROCESSED_ROOT / "Stage_4_tow_predictions"
TOW_FEATURES_DIR = PROCESSED_ROOT / "Stage_4_tow_predictions" / "tow_features"

# Aircraft features
AIRCRAFT_FEATURES_FILE = PROCESSED_ROOT / "aircraft_features_competition.csv"
AIRCRAFT_FEATURES_FALLBACK = PROCESSED_ROOT / "aircraft_features_comprehensive.csv"

# Airport metadata
APT_FILE = RAW_ROOT / "apt.parquet"

# Output directory
OUTPUT_ROOT = PROCESSED_ROOT / "Stage_7_Multiplier_features"

# Dataset configurations
DATASETS = {
    'train': {
        'traj_dir': TRAJ_MULTIPLIER_DIR / "train",
        'fuel_submission': RAW_ROOT / "fuel_train.parquet",
        'flightlist': RAW_ROOT / "flightlist_train.parquet",
        'tow_predictions': TOW_PRED_DIR / "tow_predictions_train_v2.csv",
        'output_dir': OUTPUT_ROOT / "train"
    },
    'rank': {
        'traj_dir': TRAJ_MULTIPLIER_DIR / "rank",
        'fuel_submission': RAW_ROOT / "fuel_rank_submission.parquet",
        'flightlist': RAW_ROOT / "flightlist_rank.parquet",
        'tow_predictions': TOW_PRED_DIR / "tow_predictions_rank_v2.csv",
        'output_dir': OUTPUT_ROOT / "rank"
    },
    'final': {
        'traj_dir': TRAJ_MULTIPLIER_DIR / "final",
        'fuel_submission': RAW_ROOT / "fuel_final_submission.parquet",
        'flightlist': RAW_ROOT / "flightlist_final.parquet",
        'tow_predictions': TOW_PRED_DIR / "tow_predictions_final_v2.csv",
        'output_dir': OUTPUT_ROOT / "final"
    }
}

# Constants
NUM_SEGMENTS = 20  # Number of time segments per interval for profiling
ISA_TEMP_K = 288.15
G = 9.80665  # m/s²

KINEMATIC_TOW_PREFIXES = (
    'total_flight_duration_sec', 'climb_', 'cruise_', 'descent_', 'landing_', 'takeoff_'
)

# Aircraft exclusion lists for corrected fuel application
ACROPOLE_EXCLUDE = ['A319','B748']
OPENAP_EXCLUDE = ['B748']

# Setup logging
log_file = WORKSPACE_ROOT / "logs" / "stage7_multiplier_feature_engineering.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# METAR Data Functions
# ============================================================================

def load_metar_for_date(date_str):
    """Load METAR data for a specific date."""
    metar_path = DATA_ROOT / "raw" / "metars" / f"{date_str}.csv"
    if not metar_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(metar_path, comment='#')
        df['valid'] = pd.to_datetime(df['valid'], utc=True)
        return df
    except Exception as e:
        logger.warning(f"Error loading METAR for {date_str}: {e}")
        return pd.DataFrame()

def find_closest_metar(metar_df, icao, target_time, time_window_hours=2):
    """Find METAR observation closest to target_time for a station."""
    if metar_df.empty:
        return pd.Series(dtype=float)

    # Map ICAO to station code
    station = icao[1:] if icao.startswith('K') else icao

    # Filter by station
    station_metar = metar_df[metar_df['station'] == station].copy()
    if station_metar.empty:
        return pd.Series(dtype=float)

    # Filter by time window
    time_window = pd.Timedelta(hours=time_window_hours)
    window_metar = station_metar[
        (station_metar['valid'] >= target_time - time_window) &
        (station_metar['valid'] <= target_time + time_window)
    ]

    if window_metar.empty:
        return pd.Series(dtype=float)

    # Find closest
    time_diffs = abs((window_metar['valid'] - target_time).dt.total_seconds())
    closest_idx = time_diffs.idxmin()
    return window_metar.loc[closest_idx]

def add_metar_features_to_flight(flight_id, takeoff_ts, landing_ts, origin_icao, dest_icao, metar_cache):
    """Add METAR features for a single flight."""
    metar_features = [
        'elevation', 'tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'p01i', 'alti','vsby', 'feel']

    features = {}

    if pd.isna(takeoff_ts) or pd.isna(landing_ts):
        # Initialize with NaN values
        for prefix in ['origin_metar_', 'dest_metar_']:
            for feat in metar_features:
                features[f'{prefix}{feat}'] = np.nan

        # Difference features
        diff_features = ['elevation', 'tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'p01i', 'alti',  'vsby', 'feel']
        for feat in diff_features:
            features[f'metar_diff_{feat}'] = np.nan
        return features

    takeoff_date = takeoff_ts.strftime('%Y%m%d')
    landing_date = landing_ts.strftime('%Y%m%d')

    # Load METAR if not cached
    if takeoff_date not in metar_cache:
        metar_cache[takeoff_date] = load_metar_for_date(takeoff_date)
    if landing_date not in metar_cache and landing_date != takeoff_date:
        metar_cache[landing_date] = load_metar_for_date(landing_date)

    origin_metar = {}
    dest_metar = {}

    # Origin METAR (at takeoff)
    origin_series = find_closest_metar(metar_cache[takeoff_date], origin_icao, takeoff_ts)
    if not origin_series.empty:
        for feat in metar_features:
            origin_metar[f'origin_metar_{feat}'] = origin_series.get(feat, np.nan)
    else:
        # Fallback: set to 0 for missing METAR data
        for feat in metar_features:
            origin_metar[f'origin_metar_{feat}'] = 0.0

    # Add explicit handling for drct
    if pd.isna(origin_metar.get('origin_metar_drct', np.nan)):
        origin_metar['origin_metar_drct'] = 0.0

    # Destination METAR (at landing)
    dest_series = find_closest_metar(metar_cache[landing_date], dest_icao, landing_ts)
    if not dest_series.empty:
        for feat in metar_features:
            dest_metar[f'dest_metar_{feat}'] = dest_series.get(feat, np.nan)
    else:
        # Fallback: set to 0 for missing METAR data
        for feat in metar_features:
            dest_metar[f'dest_metar_{feat}'] = 0.0

    # Add explicit handling for drct
    if pd.isna(dest_metar.get('dest_metar_drct', np.nan)):
        dest_metar['dest_metar_drct'] = 0.0

    # Calculate differences
    diff_metar = {}
    diff_features = ['elevation', 'tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'p01i', 'alti',  'vsby', 'feel']
    for feat in diff_features:
        origin_val = origin_metar.get(f'origin_metar_{feat}', 0.0)
        dest_val = dest_metar.get(f'dest_metar_{feat}', 0.0)
        diff_metar[f'metar_diff_{feat}'] = dest_val - origin_val

    # Combine all features
    features.update(origin_metar)
    features.update(dest_metar)
    features.update(diff_metar)

    return features

def calculate_isa_deviation(temp_k: float, altitude_ft: float) -> float:
    """Calculate ISA temperature deviation in Kelvin"""
    alt_m = altitude_ft * 0.3048
    isa_temp_at_alt = ISA_TEMP_K - 0.0065 * alt_m
    return temp_k - isa_temp_at_alt


def calculate_air_density(pressure_hpa: float, temp_k: float, specific_humidity: float = 0.0) -> float:
    """Calculate air density from pressure, temperature, and humidity
    
    Args:
        pressure_hpa: Pressure in hPa
        temp_k: Temperature in Kelvin
        specific_humidity: Specific humidity in kg/kg (default 0 for dry air)
    
    Returns:
        Air density in kg/m³
    """
    # Gas constants
    R_dry = 287.05  # J/(kg·K) for dry air
    R_vapor = 461.5  # J/(kg·K) for water vapor
    
    # Convert pressure to Pa
    pressure_pa = pressure_hpa * 100.0
    
    # Virtual temperature (accounts for moisture)
    T_virtual = temp_k * (1 + 0.608 * specific_humidity)
    
    # Density using ideal gas law with virtual temperature
    rho = pressure_pa / (R_dry * T_virtual)
    
    return rho


def calculate_dynamic_pressure(density: float, tas_kts: float) -> float:
    """Calculate dynamic pressure q̄ = 0.5 * ρ * V²
    
    Args:
        density: Air density in kg/m³
        tas_kts: True airspeed in knots
    
    Returns:
        Dynamic pressure in Pa
    """
    tas_ms = tas_kts * 0.514444  # Convert knots to m/s
    return 0.5 * density * (tas_ms ** 2)


def calculate_wind_components(u_wind: float, v_wind: float, track_deg: float) -> Dict[str, float]:
    """Calculate wind magnitude, headwind, and crosswind components
    
    Args:
        u_wind: East-west wind component (m/s)
        v_wind: North-south wind component (m/s)
        track_deg: Aircraft track in degrees
    
    Returns:
        Dictionary with wind_speed, headwind, crosswind
    """
    # Wind magnitude
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    
    # Convert track to radians
    track_rad = np.deg2rad(track_deg)
    
    # Headwind component (negative of wind in direction of travel)
    # Positive = headwind, negative = tailwind
    headwind = -(u_wind * np.sin(track_rad) + v_wind * np.cos(track_rad))
    
    # Crosswind component (perpendicular)
    crosswind = np.abs(-u_wind * np.cos(track_rad) + v_wind * np.sin(track_rad))
    
    # Convert to knots
    ms_to_kts = 1.94384
    
    return {
        'wind_speed_kts': wind_speed * ms_to_kts,
        'headwind_kts': headwind * ms_to_kts,
        'crosswind_kts': crosswind * ms_to_kts
    }


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in nautical miles."""
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    lat1_rad, lon1_rad = np.deg2rad([lat1, lon1])
    lat2_rad, lon2_rad = np.deg2rad([lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    earth_radius_nm = 3440.065
    return float(earth_radius_nm * c)


def integrate_fuel(fuel_flow_kgh: pd.Series, timestamps: pd.Series) -> float:
    """Integrate fuel flow over time using trapezoidal rule"""
    if fuel_flow_kgh is None or timestamps is None or len(fuel_flow_kgh) < 2:
        return 0.0

    t_hours = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
    fuel_kg = np.trapz(fuel_flow_kgh.to_numpy(dtype=float), t_hours.to_numpy(dtype=float))
    return float(max(0.0, fuel_kg))


def get_flight_phase(vertical_rate_fpm: float, altitude_ft: float) -> str:
    """Determine flight phase - must match training logic exactly"""
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


def add_phase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure phase label + one-hot columns match training logic"""
    if df.empty or not {'vertical_rate', 'altitude'}.issubset(df.columns):
        for col in ['phase', 'phase_climb', 'phase_cruise', 'phase_descent', 'phase_ground']:
            if col not in df:
                df[col] = np.nan if col == 'phase' else 0
        return df

    df['phase'] = df.apply(
        lambda row: get_flight_phase(row['vertical_rate'], row['altitude']),
        axis=1
    )
    df['phase_climb'] = (df['phase'] == 'climb').astype(int)
    df['phase_cruise'] = (df['phase'] == 'cruise').astype(int)
    df['phase_descent'] = (df['phase'] == 'descent').astype(int)
    df['phase_ground'] = (df['phase'] == 'ground').astype(int)
    return df


def select_fuel_flow_column(
    df: Optional[pd.DataFrame],
    aircraft_type: str,
    source: str = "acropole"
) -> str:
    """Select appropriate fuel flow column (corrected vs original)."""
    columns = set(df.columns) if df is not None else set()
    if source not in {"acropole", "openap"}:
        raise ValueError(f"Unknown fuel source: {source}")

    if source == "acropole":
        use_corrected = aircraft_type not in ACROPOLE_EXCLUDE
        column = 'fuel_flow_acropole_corrected_kgh' if use_corrected else 'fuel_flow_acropole_kgh'
    else:
        use_corrected = aircraft_type not in OPENAP_EXCLUDE
        column = 'fuel_flow_openap_corrected_kgh' if use_corrected else 'fuel_flow_openap_kgh'

    if df is not None and column not in columns:
        fallback = 'fuel_flow_acropole_kgh' if source == 'acropole' else 'fuel_flow_openap_kgh'
        logger.warning(
            "Column %s missing for aircraft %s (source=%s). Falling back to %s",
            column,
            aircraft_type,
            source,
            fallback,
        )
        column = fallback

    # Acceptance check: log unexpected mappings
    if source == 'acropole':
        expected = 'fuel_flow_acropole_kgh' if aircraft_type in ACROPOLE_EXCLUDE else 'fuel_flow_acropole_corrected_kgh'
    else:
        expected = 'fuel_flow_openap_kgh' if aircraft_type in OPENAP_EXCLUDE else 'fuel_flow_openap_corrected_kgh'

    if column != expected:
        logger.warning(
            "Fuel column mismatch for %s (%s). Expected %s, using %s",
            aircraft_type,
            source,
            expected,
            column,
        )

    return column


def aggregate_series(series: pd.Series, prefix: str) -> Dict[str, float]:
    """Compute summary statistics for numeric series"""
    stats = {}
    if series.empty or series.isna().all():
        for stat in ['min', 'p10', 'median', 'mean', 'p90', 'max', 'std', 'iqr']:
            stats[f"{prefix}_{stat}"] = np.nan
        return stats

    stats[f"{prefix}_min"] = series.min()
    stats[f"{prefix}_max"] = series.max()
    stats[f"{prefix}_mean"] = series.mean()
    stats[f"{prefix}_median"] = series.median()
    stats[f"{prefix}_std"] = series.std()
    stats[f"{prefix}_p10"] = series.quantile(0.10)
    stats[f"{prefix}_p90"] = series.quantile(0.90)
    stats[f"{prefix}_iqr"] = series.quantile(0.75) - series.quantile(0.25)
    return stats


# ============================================================================
# Load Reference Data
# ============================================================================

def load_reference_data(dataset: str) -> Dict[str, Any]:
    """Load all reference data needed for feature extraction"""
    config = DATASETS[dataset]
    
    logger.info(f"Loading reference data for {dataset}...")
    
    # Load fuel submission intervals
    fuel_df = pd.read_parquet(config['fuel_submission'])
    logger.info(f"  > Fuel intervals: {len(fuel_df):,} rows")
    
    # Load flightlist
    flightlist = pd.read_parquet(config['flightlist'])
    logger.info(f"  > Flightlist: {len(flightlist):,} flights")
    
    # Load TOW predictions
    tow_pred = pd.read_csv(config['tow_predictions'])
    logger.info(f"  > TOW predictions: {len(tow_pred):,} flights")
    
    # Load aircraft features (fall back if needed)
    aircraft_features = pd.DataFrame()
    aircraft_path = None
    if AIRCRAFT_FEATURES_FILE.exists():
        aircraft_path = AIRCRAFT_FEATURES_FILE
    elif AIRCRAFT_FEATURES_FALLBACK.exists():
        aircraft_path = AIRCRAFT_FEATURES_FALLBACK
        logger.warning("Primary aircraft feature file missing, using fallback comprehensive file")
    
    if aircraft_path is not None:
        aircraft_features = pd.read_csv(aircraft_path)
        logger.info(
            "  > Aircraft features: %s aircraft types (%s columns) from %s",
            f"{len(aircraft_features):,}",
            len(aircraft_features.columns),
            aircraft_path.name,
        )
    else:
        logger.warning("  > No aircraft feature file detected")
    
    # Load airport metadata
    if APT_FILE.exists():
        apt_df = pd.read_parquet(APT_FILE)
        logger.info(f"  > Airport metadata: {len(apt_df):,} airports")
    else:
        logger.warning(f"  > Airport file missing: {APT_FILE}")
        apt_df = pd.DataFrame()
    
    # Initialize METAR cache
    metar_cache = {}
    
    return {
        'fuel': fuel_df,
        'flightlist': flightlist,
        'tow': tow_pred,
        'aircraft': aircraft_features,
        'apt': apt_df,
        'metar_cache': metar_cache,
        'dataset': dataset
    }


# ============================================================================
# Load Flight Trajectory Data
# ============================================================================

def load_tow_features(flight_id: str, dataset: str) -> Optional[pd.DataFrame]:
    """
    Load pre-calculated TOW features for a flight
    
    Returns DataFrame with TOW-related features or None if not found
    """
    subdir = dataset  # 'train', 'rank', or 'final'
    tow_feat_file = TOW_FEATURES_DIR / subdir / f"{flight_id}.parquet"
    
    if not tow_feat_file.exists():
        return None
    
    try:
        df = pd.read_parquet(tow_feat_file)
        keep_cols = [
            col for col in df.columns
            if col in {'flight_id', 'tow'} or any(col.startswith(prefix) for prefix in KINEMATIC_TOW_PREFIXES)
        ]
        if keep_cols:
            df = df[keep_cols]
        return df
    except Exception as e:
        logger.warning(f"Error loading TOW features for {flight_id}: {e}")
        return None


def load_flight_data(flight_id: str, dataset: str) -> Optional[pd.DataFrame]:
    """Load trajectory data for a single flight from Stage 6.3 output
    
    For 'final' dataset, also checks 'rank' directory since final = rank + other
    """
    traj_dir = DATASETS[dataset]['traj_dir']
    traj_file = traj_dir / f"{flight_id}.parquet"
    
    # For final dataset, also check rank directory
    if not traj_file.exists() and dataset == 'final':
        rank_traj_dir = DATASETS['rank']['traj_dir']
        rank_traj_file = rank_traj_dir / f"{flight_id}.parquet"
        if rank_traj_file.exists():
            traj_file = rank_traj_file
    
    if not traj_file.exists():
        return None
    
    try:
        df = pd.read_parquet(traj_file)
        
        # Handle empty or very small trajectories
        if len(df) == 0:
            logger.warning(f"{flight_id}: Empty trajectory")
            return None
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            logger.warning(f"{flight_id}: No timestamp column")
            return None
        
        # Ensure timestamp is datetime with UTC timezone
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        # Sort by timestamp and ensure phase indicators match training logic
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = add_phase_columns(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data for {flight_id}: {e}")
        return None


# ============================================================================
# Feature Extraction: Flight Level
# ============================================================================

def extract_flight_level_features(
    df: pd.DataFrame,
    flight_id: str,
    ref_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract flight-level features"""

    features: Dict[str, Any] = {
        'flight_id': flight_id,
        'dataset': ref_data.get('dataset', 'train')
    }

    # Flight metadata
    flight_df = ref_data['flightlist'][ref_data['flightlist']['flight_id'] == flight_id]
    if not flight_df.empty:
        info = flight_df.iloc[0]
        features['aircraft_type'] = info.get('aircraft_type', 'UNKNOWN')
        features['origin_icao'] = info.get('origin_icao', 'UNKNOWN')
        features['destination_icao'] = info.get('destination_icao', 'UNKNOWN')
        # Takeoff and landing times from flightlist
        features['takeoff_ts'] = info.get('takeoff', pd.NaT)
        features['landing_ts'] = info.get('landed', pd.NaT)
        
        # Convert timestamps if they are not datetime (handle Unix timestamps)
        if pd.notna(features['takeoff_ts']) and not pd.api.types.is_datetime64_any_dtype(type(features['takeoff_ts'])):
            if isinstance(features['takeoff_ts'], (int, float)):
                # Assume milliseconds since epoch
                features['takeoff_ts'] = pd.to_datetime(features['takeoff_ts'], unit='ms', utc=True)
            else:
                features['takeoff_ts'] = pd.to_datetime(features['takeoff_ts'], utc=True)
        
        if pd.notna(features['landing_ts']) and not pd.api.types.is_datetime64_any_dtype(type(features['landing_ts'])):
            if isinstance(features['landing_ts'], (int, float)):
                features['landing_ts'] = pd.to_datetime(features['landing_ts'], unit='ms', utc=True)
            else:
                features['landing_ts'] = pd.to_datetime(features['landing_ts'], utc=True)
        
        # Ensure timestamps are timezone-aware (assume UTC for aviation data)
        if pd.notna(features['takeoff_ts']) and features['takeoff_ts'].tz is None:
            features['takeoff_ts'] = features['takeoff_ts'].tz_localize('UTC')
        if pd.notna(features['landing_ts']) and features['landing_ts'].tz is None:
            features['landing_ts'] = features['landing_ts'].tz_localize('UTC')
    else:
        features['aircraft_type'] = 'UNKNOWN'
        features['origin_icao'] = 'UNKNOWN'
        features['destination_icao'] = 'UNKNOWN'
        features['takeoff_ts'] = pd.NaT
        features['landing_ts'] = pd.NaT

    # Fallback to fuel intervals if flightlist doesn't have takeoff/landing times
    if pd.isna(features['takeoff_ts']) or pd.isna(features['landing_ts']):
        fuel_df = ref_data['fuel'][ref_data['fuel']['flight_id'] == flight_id]
        if not fuel_df.empty:
            # Use earliest interval start as takeoff, latest interval end as landing
            min_start = fuel_df['start'].min()
            max_end = fuel_df['end'].max()
            if pd.isna(features['takeoff_ts']) and pd.notna(min_start):
                features['takeoff_ts'] = min_start
                if features['takeoff_ts'].tz is None:
                    features['takeoff_ts'] = features['takeoff_ts'].tz_localize('UTC')
            if pd.isna(features['landing_ts']) and pd.notna(max_end):
                features['landing_ts'] = max_end
                if features['landing_ts'].tz is None:
                    features['landing_ts'] = features['landing_ts'].tz_localize('UTC')
            logger.info(f"Using fuel intervals for takeoff/landing times for flight {flight_id}")

    aircraft_type = features['aircraft_type']

    # Airport enrichment
    apt_df = ref_data.get('apt', pd.DataFrame())
    if not apt_df.empty:
        for loc_key, prefix in [(features['origin_icao'], 'origin'), (features['destination_icao'], 'dest')]:
            if isinstance(loc_key, str) and loc_key in apt_df['icao'].values:
                apt_row = apt_df[apt_df['icao'] == loc_key].iloc[0]
                features[f'{prefix}_elevation_ft'] = apt_row.get('elevation')
                features[f'{prefix}_latitude_deg'] = apt_row.get('latitude')
                features[f'{prefix}_longitude_deg'] = apt_row.get('longitude')

        if all(k in features for k in ['origin_latitude_deg', 'origin_longitude_deg', 'dest_latitude_deg', 'dest_longitude_deg']):
            lat1 = features['origin_latitude_deg']
            lon1 = features['origin_longitude_deg']
            lat2 = features['dest_latitude_deg']
            lon2 = features['dest_longitude_deg']
            if not any(pd.isna([lat1, lon1, lat2, lon2])):
                features['origin_dest_gcdist_nm'] = haversine_nm(lat1, lon1, lat2, lon2)

    # Aircraft characteristics
    aircraft_df = ref_data.get('aircraft', pd.DataFrame())
    if not aircraft_df.empty:
        ac_row = aircraft_df[aircraft_df['aircraft_type'] == aircraft_type]
        if not ac_row.empty:
            ac_row = ac_row.iloc[0]
            numeric_cols = aircraft_df.select_dtypes(include=[np.number]).columns
            excluded = {
                'aircraft_type', 'WTC_encoded', 'acropole_engine_type',
                'configuration_index', 'engine_usage_percentage',
                'Wingspan_ft_with_winglets_sharklets', 'wingspan_with_winglets_m',
                'Wingspan_ft_without_winglets_sharklets', 'wingspan_without_winglets_m'
            }
            for col in numeric_cols:
                if not str(col).startswith('Unnamed') and col not in excluded:
                    features[f'ac_{col}'] = ac_row[col]

            for feat in ['manufacturer_encoded', 'model_family_encoded', 'engine_type_encoded', 'size_category_encoded']:
                if feat in ac_row:
                    features[f'ac_{feat}'] = ac_row[feat]

            # Wingspan convenience features
            for src, target in [
                ('Wingspan_ft_with_winglets_sharklets', 'ac_wingspan_ft'),
                ('Wingspan_ft_without_winglets_sharklets', 'ac_wingspan_ft'),
                ('wingspan_with_winglets_m', 'ac_wingspan_m'),
                ('wingspan_without_winglets_m', 'ac_wingspan_m')
            ]:
                if target not in features and src in ac_row and pd.notna(ac_row[src]):
                    features[target] = ac_row[src]

    # TOW prediction
    tow_df = ref_data['tow'][ref_data['tow']['flight_id'] == flight_id]
    features['tow_pred_kg'] = tow_df.iloc[0].get('tow', 70000) if not tow_df.empty else 70000

    # Load TOW kinematic features
    dataset_name = ref_data.get('dataset', 'train')
    tow_features_df = load_tow_features(flight_id, dataset_name)
    has_tow_features = tow_features_df is not None and not tow_features_df.empty
    features['has_tow_features'] = has_tow_features
    if has_tow_features and tow_features_df is not None:
        row = tow_features_df.iloc[0]
        # Exclude phase-related columns to calculate them directly from trajectory
        phase_related_cols = {
            'climb_time_sec', 'climb_distance_nm', 'climb_avg_speed', 'climb_max_speed', 'climb_median_speed',
            'climb_avg_tas', 'climb_median_tas', 'climb_avg_altitude', 'climb_max_altitude', 'climb_median_altitude',
            'climb_std_altitude', 'climb_avg_vr', 'climb_median_vr', 'climb_accel',
            # Add other phases if needed: descent, cruise, takeoff, landing
        }
        for col in tow_features_df.columns:
            if col in {'flight_id', 'tow', 'fuel_kg_actual'}:
                continue
            if '_num_points' in col:
                continue
            if col in phase_related_cols:
                continue  # Skip phase-related columns
            features[col] = row[col]

    # Timing: Primary source is flightlist, fallback to trajectory
    if pd.isna(features['takeoff_ts']) or pd.isna(features['landing_ts']):
        # Fallback to trajectory data if flightlist doesn't have takeoff/landing times
        if not df.empty and 'timestamp' in df.columns:
            features['takeoff_ts'] = df['timestamp'].iloc[0]
            features['landing_ts'] = df['timestamp'].iloc[-1]
            logger.info(f"Using trajectory timestamps for flight {flight_id} (flightlist missing)")
        else:
            logger.warning(f"No takeoff/landing times available for flight {flight_id}")

    # Calculate flight duration and calendar features
    if pd.notna(features['takeoff_ts']) and pd.notna(features['landing_ts']):
        features['total_flight_duration_sec'] = (features['landing_ts'] - features['takeoff_ts']).total_seconds()
        # Calendar encodings
        takeoff_ts = features['takeoff_ts']
        features['flt_time_of_day_sin'] = np.sin(2 * np.pi * (takeoff_ts.hour + takeoff_ts.minute / 60.0) / 24.0)
        features['flt_time_of_day_cos'] = np.cos(2 * np.pi * (takeoff_ts.hour + takeoff_ts.minute / 60.0) / 24.0)
        features['flt_day_of_week'] = takeoff_ts.dayofweek
    else:
        features['total_flight_duration_sec'] = np.nan
        features['flt_time_of_day_sin'] = np.nan
        features['flt_time_of_day_cos'] = np.nan
        features['flt_day_of_week'] = np.nan

    # Add METAR features
    metar_cache = ref_data.get('metar_cache', {})
    metar_features = add_metar_features_to_flight(
        flight_id,
        features['takeoff_ts'],
        features['landing_ts'],
        features['origin_icao'],
        features['destination_icao'],
        metar_cache
    )
    features.update(metar_features)

    # Temporal features: Month, Season, Time since midnight, Day of year
    if pd.notna(features['takeoff_ts']):
        takeoff_ts = features['takeoff_ts']
        month = takeoff_ts.month
        features['flt_month_sin'] = np.sin(2 * np.pi * month / 12)
        features['flt_month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Season: 0=winter, 1=spring, 2=summer, 3=fall
        if month in [12, 1, 2]:
            features['flt_season'] = 0  # winter
        elif month in [3, 4, 5]:
            features['flt_season'] = 1  # spring
        elif month in [6, 7, 8]:
            features['flt_season'] = 2  # summer
        else:
            features['flt_season'] = 3  # fall
        
        # Time since midnight (normalized 0-1)
        seconds_since_midnight = takeoff_ts.hour * 3600 + takeoff_ts.minute * 60 + takeoff_ts.second
        features['flt_time_since_midnight_norm'] = seconds_since_midnight / 86400.0
        
        # Day of year (cyclical)
        day_of_year = takeoff_ts.dayofyear
        features['flt_dayofyear_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        features['flt_dayofyear_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Flight-level statistics
    if not df.empty:
        if 'altitude' in df:
            features['flt_altitude_min_ft'] = df['altitude'].min()
            features['flt_altitude_max_ft'] = df['altitude'].max()
            features['flt_altitude_mean_ft'] = df['altitude'].mean()
            features['flt_altitude_std_ft'] = df['altitude'].std()

        if 'groundspeed' in df:
            features['flt_groundspeed_mean_kts'] = df['groundspeed'].mean()
            features['flt_groundspeed_std_kts'] = df['groundspeed'].std()
        if 'TAS' in df:
            features['flt_tas_mean_kts'] = df['TAS'].mean()
            features['flt_tas_std_kts'] = df['TAS'].std()
        if 'vertical_rate' in df:
            features['flt_vertical_rate_mean_fpm'] = df['vertical_rate'].mean()
            features['flt_vertical_rate_std_fpm'] = df['vertical_rate'].std()
            features['flt_max_climb_rate_fpm'] = df['vertical_rate'].max()
            features['flt_max_descent_rate_fpm'] = df['vertical_rate'].min()

        # Phase distribution
        if {'phase_climb', 'phase_cruise', 'phase_descent', 'phase_ground'}.issubset(df.columns):
            total_points = max(len(df), 1)
            for phase in ['climb', 'cruise', 'descent', 'ground']:
                features[f'flt_phase_{phase}_fraction'] = df[f'phase_{phase}'].sum() / total_points

            # Direct phase-based aggregations from trajectory data
            for phase in ['climb', 'cruise', 'descent']:
                phase_mask = df[f'phase_{phase}'] == 1
                phase_df = df[phase_mask]
                
                if not phase_df.empty:
                    # Time and distance
                    features[f'{phase}_time_sec'] = (phase_df['timestamp'].max() - phase_df['timestamp'].min()).total_seconds() if len(phase_df) > 1 else 0.0
                    features[f'{phase}_distance_nm'] = phase_df['distance_cum_nm'].max() - phase_df['distance_cum_nm'].min() if 'distance_cum_nm' in phase_df.columns and len(phase_df) > 1 else 0.0
                    
                    # Speed aggregations
                    if 'groundspeed' in phase_df.columns:
                        features[f'{phase}_avg_speed'] = phase_df['groundspeed'].mean()
                        features[f'{phase}_max_speed'] = phase_df['groundspeed'].max()
                        features[f'{phase}_median_speed'] = phase_df['groundspeed'].median()
                    
                    # TAS aggregations
                    if 'TAS' in phase_df.columns:
                        features[f'{phase}_avg_tas'] = phase_df['TAS'].mean()
                        features[f'{phase}_median_tas'] = phase_df['TAS'].median()
                    
                    # Altitude aggregations
                    if 'altitude' in phase_df.columns:
                        features[f'{phase}_avg_altitude'] = phase_df['altitude'].mean()
                        features[f'{phase}_max_altitude'] = phase_df['altitude'].max()
                        features[f'{phase}_median_altitude'] = phase_df['altitude'].median()
                        features[f'{phase}_std_altitude'] = phase_df['altitude'].std()
                    
                    # Vertical rate aggregations
                    if 'vertical_rate' in phase_df.columns:
                        features[f'{phase}_avg_vr'] = phase_df['vertical_rate'].mean()
                        features[f'{phase}_median_vr'] = phase_df['vertical_rate'].median()
                    
                    # Acceleration (calculated from groundspeed if available)
                    if 'groundspeed' in phase_df.columns and len(phase_df) > 1:
                        gs_diff = phase_df['groundspeed'].diff()
                        time_diff_sec = phase_df['timestamp'].diff().dt.total_seconds()
                        valid_mask = (time_diff_sec > 0) & (~np.isnan(gs_diff)) & (~np.isnan(time_diff_sec))
                        if valid_mask.any():
                            accel_values = gs_diff[valid_mask] / time_diff_sec[valid_mask]
                            # Handle any remaining NaN/inf in accel_values
                            accel_values = accel_values.replace([np.inf, -np.inf], np.nan).dropna()
                            if not accel_values.empty:
                                mean_accel = accel_values.mean()
                                features[f'{phase}_accel'] = mean_accel if not np.isnan(mean_accel) else 0.0
                            else:
                                features[f'{phase}_accel'] = 0.0
                        else:
                            features[f'{phase}_accel'] = 0.0
                    else:
                        features[f'{phase}_accel'] = 0.0
                else:
                    # Set to 0.0 for phases with no activity
                    features[f'{phase}_time_sec'] = 0.0
                    features[f'{phase}_distance_nm'] = 0.0
                    features[f'{phase}_avg_speed'] = 0.0
                    features[f'{phase}_max_speed'] = 0.0
                    features[f'{phase}_median_speed'] = 0.0
                    features[f'{phase}_avg_tas'] = 0.0
                    features[f'{phase}_median_tas'] = 0.0
                    features[f'{phase}_avg_altitude'] = 0.0
                    features[f'{phase}_max_altitude'] = 0.0
                    features[f'{phase}_median_altitude'] = 0.0
                    features[f'{phase}_std_altitude'] = 0.0
                    features[f'{phase}_avg_vr'] = 0.0
                    features[f'{phase}_median_vr'] = 0.0

        # Ensure acceleration features are always set
        for phase in ['climb', 'cruise', 'descent']:
            if f'{phase}_accel' not in features or pd.isna(features[f'{phase}_accel']):
                features[f'{phase}_accel'] = 0.0

        # Total track change for flight
        if 'track' in df.columns and len(df) > 1:
            track_changes = df['track'].diff().abs()
            track_changes = track_changes.apply(lambda x: min(x, 360 - x) if not np.isnan(x) else 0)
            features['flt_total_track_change_deg'] = track_changes.sum()

        # Flight-level mass
        if 'mass_acropole_kg' in df:
            features['flt_mass_acropole_start_kg'] = df['mass_acropole_kg'].iloc[0]
            features['flt_mass_acropole_end_kg'] = df['mass_acropole_kg'].iloc[-1]
            features['flt_mass_acropole_change_kg'] = features['flt_mass_acropole_start_kg'] - features['flt_mass_acropole_end_kg']

    # Flight-level fuel totals (corrected selection)
    acro_col = select_fuel_flow_column(df if not df.empty else None, aircraft_type, 'acropole')
    if acro_col in df:
        features['flt_acropole_fuel_total_kg'] = integrate_fuel(df[acro_col], df['timestamp'])

    openap_col = select_fuel_flow_column(df if not df.empty else None, aircraft_type, 'openap')
    if openap_col in df:
        features['flt_openap_fuel_total_kg'] = integrate_fuel(df[openap_col], df['timestamp'])

    if 'flt_acropole_fuel_total_kg' in features and 'flt_openap_fuel_total_kg' in features:
        acropole_fuel = features['flt_acropole_fuel_total_kg']
        openap_fuel = features['flt_openap_fuel_total_kg']
        if openap_fuel > 0:
            features['flt_fuel_ratio_acropole_openap'] = acropole_fuel / openap_fuel
            features['flt_fuel_pct_error_acropole_openap'] = ((acropole_fuel - openap_fuel) / openap_fuel) * 100
        features['flt_fuel_diff_acropole_openap_kg'] = acropole_fuel - openap_fuel

    # ISA deviation
    if not df.empty and {'temperature_pl', 'altitude'}.issubset(df.columns):
        isa_deviations = [
            calculate_isa_deviation(temp, alt)
            for temp, alt in zip(df['temperature_pl'], df['altitude'])
            if not np.isnan(temp) and not np.isnan(alt)
        ]
        if isa_deviations:
            features['flt_isa_deviation_mean_K'] = float(np.mean(isa_deviations))
            features['flt_isa_deviation_std_K'] = float(np.std(isa_deviations))

    return features


# ============================================================================
# Feature Extraction: Interval Level
# ============================================================================

def extract_interval_features(
    df: pd.DataFrame,
    interval_start_ts: pd.Timestamp,
    interval_end_ts: pd.Timestamp,
    interval_idx: int,
    flight_features: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract aggregated features for a specific interval"""
    
    features = {}
    
    # Handle empty DataFrame (no trajectory data)
    if df.empty or 'timestamp' not in df.columns:
        # Still calculate relative positions if possible
        takeoff_ts = flight_features.get('takeoff_ts')
        landing_ts = flight_features.get('landing_ts')
        total_duration = flight_features.get('total_flight_duration_sec')
        
        rel_features = {}
        if takeoff_ts is not None and landing_ts is not None and pd.notna(takeoff_ts) and pd.notna(landing_ts) and total_duration:
            mid_ts = interval_start_ts + (interval_end_ts - interval_start_ts) / 2
            
            time_start = (interval_start_ts - takeoff_ts).total_seconds()
            time_end = (interval_end_ts - takeoff_ts).total_seconds()
            time_mid = (mid_ts - takeoff_ts).total_seconds()
            
            rel_features = {
                'interval_rel_position_start': time_start / total_duration,
                'interval_rel_position_end': time_end / total_duration,
                'interval_rel_position_midpoint': time_mid / total_duration,
                'interval_rel_span': (time_end - time_start) / total_duration,
                'time_since_takeoff_min': time_start / 60.0,
                'time_to_landing_min': (landing_ts - interval_end_ts).total_seconds() / 60.0,
                'int_time_since_takeoff_sec': time_start,
                'int_time_before_landing_sec': (landing_ts - interval_end_ts).total_seconds(),
                'int_relative_position_in_flight': time_mid / total_duration,
                'interval_rel_position': time_mid / total_duration
            }
        else:
            nan_dict = {k: np.nan for k in [
                'interval_rel_position_start', 'interval_rel_position_end', 'interval_rel_position_midpoint',
                'interval_rel_span', 'time_since_takeoff_min', 'time_to_landing_min',
                'int_time_since_takeoff_sec', 'int_time_before_landing_sec', 'int_relative_position_in_flight',
                'interval_rel_position'
            ]}
            rel_features = nan_dict
        
        return {
            'interval_idx': interval_idx,
            'interval_start_ts': interval_start_ts,
            'interval_end_ts': interval_end_ts,
            'interval_duration_sec': (interval_end_ts - interval_start_ts).total_seconds(),
            'has_trajectory_data': False,
            'int_num_points': 0,
            **rel_features
        }
    
    # Filter data to interval
    mask = (df['timestamp'] >= interval_start_ts) & (df['timestamp'] <= interval_end_ts)
    interval_df = df[mask].copy()
    aircraft_type = flight_features.get('aircraft_type', 'UNKNOWN')
    features['int_num_points'] = len(interval_df)
    
    # Basic metadata
    features['interval_idx'] = interval_idx
    features['interval_start_ts'] = interval_start_ts
    features['interval_end_ts'] = interval_end_ts
    features['interval_duration_sec'] = (interval_end_ts - interval_start_ts).total_seconds()
    features['has_trajectory_data'] = len(interval_df) > 0
    
    # Relative position in flight (enhanced) - calculate regardless of trajectory data
    takeoff_ts = flight_features.get('takeoff_ts')
    landing_ts = flight_features.get('landing_ts')
    total_duration = flight_features.get('total_flight_duration_sec')
    
    if takeoff_ts is not None and landing_ts is not None and pd.notna(takeoff_ts) and pd.notna(landing_ts) and total_duration:
        mid_ts = interval_start_ts + (interval_end_ts - interval_start_ts) / 2
        
        # Start, end, and midpoint positions
        time_start = (interval_start_ts - takeoff_ts).total_seconds()
        time_end = (interval_end_ts - takeoff_ts).total_seconds()
        time_mid = (mid_ts - takeoff_ts).total_seconds()
        
        features['interval_rel_position_start'] = time_start / total_duration
        features['interval_rel_position_end'] = time_end / total_duration
        features['interval_rel_position_midpoint'] = time_mid / total_duration
        features['interval_rel_span'] = (time_end - time_start) / total_duration
        
        # Time since takeoff and until landing (minutes)
        features['time_since_takeoff_min'] = time_start / 60.0
        features['time_to_landing_min'] = (landing_ts - interval_end_ts).total_seconds() / 60.0
        features['int_time_since_takeoff_sec'] = time_start
        features['int_time_before_landing_sec'] = (landing_ts - interval_end_ts).total_seconds()
        features['int_relative_position_in_flight'] = features['interval_rel_position_midpoint']
        
        # Backward compatibility
        features['interval_rel_position'] = features['interval_rel_position_midpoint']
    else:
        features['interval_rel_position_start'] = np.nan
        features['interval_rel_position_end'] = np.nan
        features['interval_rel_position_midpoint'] = np.nan
        features['interval_rel_span'] = np.nan
        features['interval_rel_position'] = np.nan
        features['time_since_takeoff_min'] = np.nan
        features['time_to_landing_min'] = np.nan
        features['int_time_since_takeoff_sec'] = np.nan
        features['int_time_before_landing_sec'] = np.nan
        features['int_relative_position_in_flight'] = np.nan
    
    if len(interval_df) == 0:
        return features
    
    dt_seconds = interval_df['timestamp'].diff().dt.total_seconds().fillna(0) if 'timestamp' in interval_df else pd.Series(dtype=float)
    
    # Altitude features
    if 'altitude' in interval_df.columns:
        features['int_alt_start_ft'] = interval_df['altitude'].iloc[0]
        features['int_alt_end_ft'] = interval_df['altitude'].iloc[-1]
        features['int_alt_change_ft'] = features['int_alt_end_ft'] - features['int_alt_start_ft']
        features.update(aggregate_series(interval_df['altitude'], 'int_altitude'))
        alt_mean = features.get('int_altitude_mean')
        alt_std = features.get('int_altitude_std')
        if alt_mean is not None and alt_std is not None and not np.isnan(alt_mean) and abs(alt_mean) > 1e-6:
            features['int_altitude_cv'] = alt_std / abs(alt_mean)
    
    # Speed features
    if 'groundspeed' in interval_df.columns:
        features.update(aggregate_series(interval_df['groundspeed'], 'int_groundspeed'))
    
    if 'TAS' in interval_df.columns:
        features.update(aggregate_series(interval_df['TAS'], 'int_tas'))
        tas_mean = features.get('int_tas_mean')
        tas_std = features.get('int_tas_std')
        if tas_mean is not None and tas_std is not None and not np.isnan(tas_mean) and abs(tas_mean) > 1e-6:
            features['int_tas_cv'] = tas_std / abs(tas_mean)

        if not dt_seconds.empty and len(interval_df) > 1:
            tas_series = pd.to_numeric(interval_df['TAS'], errors='coerce')
            tas_diff = tas_series.diff()
            valid_dt = pd.to_numeric(dt_seconds, errors='coerce').replace(0, np.nan)
            tas_acc = tas_diff / valid_dt
            tas_acc = tas_acc.replace([np.inf, -np.inf], np.nan).dropna()
            if not tas_acc.empty:
                features.update(aggregate_series(tas_acc, 'int_tas_acc'))
    
    # Vertical rate
    if 'vertical_rate' in interval_df.columns:
        features.update(aggregate_series(interval_df['vertical_rate'], 'int_vertical_rate'))
        
        # Set VR to 0 for cruise phases (if majority of points are cruise)
        if 'phase_cruise' in interval_df.columns:
            cruise_fraction = interval_df['phase_cruise'].mean()
            if cruise_fraction > 0.5:  # Majority cruise
                features['int_vertical_rate_mean'] = 0.0
        
        # Calculate VR from altitude if still missing and altitude available
        if pd.isna(features.get('int_vertical_rate_mean', np.nan)) and 'altitude' in interval_df.columns and len(interval_df) >= 2:
            alt_start = interval_df['altitude'].iloc[0]
            alt_end = interval_df['altitude'].iloc[-1]
            duration_sec = features.get('interval_duration_sec', 0)
            if duration_sec > 0:
                vr_approx = (alt_end - alt_start) / duration_sec * 60  # fpm
                features['int_vertical_rate_mean'] = vr_approx
        
        vr_mean = features.get('int_vertical_rate_mean')
        vr_std = features.get('int_vertical_rate_std')
        if vr_mean is not None and vr_std is not None and not np.isnan(vr_mean):
            if abs(vr_mean) > 1e-6:
                features['int_vertical_rate_cv'] = vr_std / abs(vr_mean)
            else:
                features['int_vertical_rate_cv'] = 0.0  # No variability if mean is ~0

        vr_abs = interval_df['vertical_rate'].abs()
        if not vr_abs.empty:
            features.update(aggregate_series(vr_abs, 'int_vertical_rate_abs'))
        
        # ========================================================================
        # Climb and Descent Rate Features
        # ========================================================================
        
        # Separate climb and descent vertical rates
        climb_rates = interval_df[interval_df['vertical_rate'] > 0]['vertical_rate']
        descent_rates = interval_df[interval_df['vertical_rate'] < 0]['vertical_rate']
        
        if not climb_rates.empty:
            features['int_max_climb_rate_fpm'] = climb_rates.max()
            features['int_mean_climb_rate_fpm'] = climb_rates.mean()
            features['int_std_climb_rate_fpm'] = climb_rates.std()
            features['int_min_climb_rate_fpm'] = climb_rates.min()
        else:
            features['int_max_climb_rate_fpm'] = 0.0  # Changed from np.nan
            features['int_mean_climb_rate_fpm'] = 0.0
            features['int_std_climb_rate_fpm'] = 0.0
            features['int_min_climb_rate_fpm'] = 0.0
        
        if not descent_rates.empty:
            features['int_max_descent_rate_fpm'] = descent_rates.max()  # Most negative (least descent)
            features['int_mean_descent_rate_fpm'] = descent_rates.mean()
            features['int_std_descent_rate_fpm'] = descent_rates.std()
            features['int_min_descent_rate_fpm'] = descent_rates.min()  # Most negative
        else:
            features['int_max_descent_rate_fpm'] = 0.0  # Changed from np.nan
            features['int_mean_descent_rate_fpm'] = 0.0
            features['int_std_descent_rate_fpm'] = 0.0
            features['int_min_descent_rate_fpm'] = 0.0
    
    # ========================================================================
    # TAS Acceleration Positive/Negative Components
    # ========================================================================
    
    if 'TAS' in interval_df.columns and not dt_seconds.empty and len(interval_df) > 1:
        tas_series = pd.to_numeric(interval_df['TAS'], errors='coerce')
        tas_diff = tas_series.diff()
        valid_dt = pd.to_numeric(dt_seconds, errors='coerce').replace(0, np.nan)
        tas_acc = tas_diff / valid_dt
        tas_acc = tas_acc.replace([np.inf, -np.inf], np.nan).dropna()
        
        if not tas_acc.empty:
            # Positive accelerations (speeding up)
            pos_acc = tas_acc[tas_acc > 0]
            if not pos_acc.empty:
                features['int_tas_accel_pos_mean'] = pos_acc.mean()
                features['int_tas_accel_pos_std'] = pos_acc.std()
                features['int_tas_accel_pos_max'] = pos_acc.max()
                features['int_tas_accel_pos_min'] = pos_acc.min()
            else:
                # Changed: Set to 0 instead of np.nan to reduce imputation
                features['int_tas_accel_pos_mean'] = 0
                features['int_tas_accel_pos_std'] = 0
                features['int_tas_accel_pos_max'] = 0
                features['int_tas_accel_pos_min'] = 0
            
            # Negative accelerations (slowing down)
            neg_acc = tas_acc[tas_acc < 0]
            if not neg_acc.empty:
                features['int_tas_accel_neg_mean'] = neg_acc.mean()
                features['int_tas_accel_neg_std'] = neg_acc.std()
                features['int_tas_accel_neg_max'] = neg_acc.max()  # Least negative
                features['int_tas_accel_neg_min'] = neg_acc.min()  # Most negative
            else:
                # Changed: Set to 0 instead of np.nan to reduce imputation
                features['int_tas_accel_neg_mean'] = 0
                features['int_tas_accel_neg_std'] = 0
                features['int_tas_accel_neg_max'] = 0
                features['int_tas_accel_neg_min'] = 0
    
    phase_fracs: Dict[str, float] = {}
    # Flight phase distribution
    total_points = max(len(interval_df), 1)
    if {'phase_climb', 'phase_cruise', 'phase_descent', 'phase_ground'}.issubset(interval_df.columns):
        features['int_frac_climb'] = interval_df['phase_climb'].sum() / total_points
        features['int_frac_cruise'] = interval_df['phase_cruise'].sum() / total_points
        features['int_frac_descent'] = interval_df['phase_descent'].sum() / total_points
        features['int_frac_ground'] = interval_df['phase_ground'].sum() / total_points
        phase_fracs = {
            'climb': features['int_frac_climb'],
            'cruise': features['int_frac_cruise'],
            'descent': features['int_frac_descent'],
            'ground': features['int_frac_ground']
        }
    else:
        phases = interval_df.apply(
            lambda row: get_flight_phase(row['vertical_rate'], row['altitude']),
            axis=1
        ) if {'vertical_rate', 'altitude'}.issubset(interval_df.columns) else pd.Series([], dtype=str)
        phase_counts = phases.value_counts()
        features['int_frac_climb'] = phase_counts.get('climb', 0) / total_points
        features['int_frac_cruise'] = phase_counts.get('cruise', 0) / total_points
        features['int_frac_descent'] = phase_counts.get('descent', 0) / total_points
        features['int_frac_ground'] = phase_counts.get('ground', 0) / total_points
        phase_fracs = {
            'climb': features['int_frac_climb'],
            'cruise': features['int_frac_cruise'],
            'descent': features['int_frac_descent'],
            'ground': features['int_frac_ground']
        }
    
    # Mass features (from Acropole)
    if 'mass_acropole_kg' in interval_df.columns:
        features['int_mass_start_kg'] = interval_df['mass_acropole_kg'].iloc[0]
        features['int_mass_end_kg'] = interval_df['mass_acropole_kg'].iloc[-1]
        features['int_mass_change_kg'] = features['int_mass_start_kg'] - features['int_mass_end_kg']
        features.update(aggregate_series(interval_df['mass_acropole_kg'], 'int_mass'))

    fuel_col_acro = select_fuel_flow_column(interval_df, aircraft_type, 'acropole') if len(interval_df) else 'fuel_flow_acropole_kgh'
    fuel_col_openap = select_fuel_flow_column(interval_df, aircraft_type, 'openap') if len(interval_df) else 'fuel_flow_openap_kgh'
    features['int_acropole_corrected_used'] = int(aircraft_type not in ACROPOLE_EXCLUDE)
    features['int_openap_corrected_used'] = int(aircraft_type not in OPENAP_EXCLUDE)
    
    # Fuel flow features (from Acropole)
    if fuel_col_acro in interval_df.columns:
        acro_stats = aggregate_series(interval_df[fuel_col_acro], 'int_fuel_flow_acropole')
        features.update(acro_stats)

        features['int_fuel_consumed_acropole_kg'] = integrate_fuel(
            interval_df[fuel_col_acro],
            interval_df['timestamp']
        )
    
    # ========================================================================
    # OpenAP Features
    # ========================================================================
    
    # OpenAP thrust
    if 'thrust_openap_n' in interval_df.columns:
        features.update(aggregate_series(interval_df['thrust_openap_n'], 'int_thrust_openap'))
    
    # OpenAP drag
    if 'drag_openap_n' in interval_df.columns:
        features.update(aggregate_series(interval_df['drag_openap_n'], 'int_drag_openap'))
    
    # OpenAP mass
    if 'mass_openap_kg' in interval_df.columns:
        features['int_mass_openap_start_kg'] = interval_df['mass_openap_kg'].iloc[0]
        features['int_mass_openap_end_kg'] = interval_df['mass_openap_kg'].iloc[-1]
        features['int_mass_openap_change_kg'] = features['int_mass_openap_start_kg'] - features['int_mass_openap_end_kg']
        features.update(aggregate_series(interval_df['mass_openap_kg'], 'int_mass_openap'))
    
    # OpenAP fuel flow and consumption
    if fuel_col_openap in interval_df.columns:
        openap_stats = aggregate_series(interval_df[fuel_col_openap], 'int_fuel_flow_openap')
        features.update(openap_stats)
        features['int_fuel_consumed_openap_kg'] = integrate_fuel(
            interval_df[fuel_col_openap], 
            interval_df['timestamp']
        )

    # Multiplier statistics
    if 'acropole_multiplier' in interval_df.columns:
        features.update(aggregate_series(interval_df['acropole_multiplier'], 'int_acropole_multiplier'))
    if 'openap_multiplier' in interval_df.columns:
        features.update(aggregate_series(interval_df['openap_multiplier'], 'int_openap_multiplier'))
    
    # ========================================================================
    # Acropole vs OpenAP Comparison Features
    # ========================================================================
    
    # Mass comparison
    if 'int_mass_start_kg' in features and 'int_mass_openap_start_kg' in features:
        features['int_mass_diff_acropole_openap_kg'] = features['int_mass_start_kg'] - features['int_mass_openap_start_kg']
        if features['int_mass_openap_start_kg'] > 0:
            features['int_mass_ratio_acropole_openap'] = features['int_mass_start_kg'] / features['int_mass_openap_start_kg']
            features['int_mass_pct_error_acropole_openap'] = ((features['int_mass_start_kg'] - features['int_mass_openap_start_kg']) / features['int_mass_openap_start_kg']) * 100
    
    # Fuel flow comparison
    if 'int_fuel_flow_acropole_mean' in features and 'int_fuel_flow_openap_mean' in features:
        features['int_ff_diff_acropole_openap_kgh'] = features['int_fuel_flow_acropole_mean'] - features['int_fuel_flow_openap_mean']
        if features['int_fuel_flow_openap_mean'] > 0:
            features['int_ff_ratio_acropole_openap'] = features['int_fuel_flow_acropole_mean'] / features['int_fuel_flow_openap_mean']
            features['int_ff_pct_error_acropole_openap'] = ((features['int_fuel_flow_acropole_mean'] - features['int_fuel_flow_openap_mean']) / features['int_fuel_flow_openap_mean']) * 100
    
    # Fuel consumed comparison
    if 'int_fuel_consumed_acropole_kg' in features and 'int_fuel_consumed_openap_kg' in features:
        features['int_fuel_diff_acropole_openap_kg'] = features['int_fuel_consumed_acropole_kg'] - features['int_fuel_consumed_openap_kg']
        if features['int_fuel_consumed_openap_kg'] > 0:
            features['int_fuel_ratio_acropole_openap'] = features['int_fuel_consumed_acropole_kg'] / features['int_fuel_consumed_openap_kg']
            features['int_fuel_pct_error_acropole_openap'] = ((features['int_fuel_consumed_acropole_kg'] - features['int_fuel_consumed_openap_kg']) / features['int_fuel_consumed_openap_kg']) * 100
    
    # Net force comparison (thrust - drag)
    if 'int_thrust_openap_mean' in features and 'int_drag_openap_mean' in features:
        features['int_net_force_openap_n'] = features['int_thrust_openap_mean'] - features['int_drag_openap_mean']
        features['int_thrust_drag_ratio_openap'] = features['int_thrust_openap_mean'] / features['int_drag_openap_mean'] if features['int_drag_openap_mean'] > 0 else np.nan
    
    # ========================================================================
    # Advanced Interaction Features
    # ========================================================================
    
    # Thrust × TAS (power-like feature)
    if 'int_thrust_openap_mean' in features and 'int_tas_mean' in features:
        features['int_thrust_power'] = features['int_thrust_openap_mean'] * features['int_tas_mean']
    
    # Drag × TAS (drag power)
    if 'int_drag_openap_mean' in features and 'int_tas_mean' in features:
        features['int_drag_power'] = features['int_drag_openap_mean'] * features['int_tas_mean']
    
    # Net power (thrust power - drag power)
    if 'int_thrust_power' in features and 'int_drag_power' in features:
        features['int_net_power'] = features['int_thrust_power'] - features['int_drag_power']
    
    # Acropole fuel efficiency metrics
    if 'int_fuel_consumed_acropole_kg' in features:
        # Fuel per second
        if features.get('interval_duration_sec', 0) > 0:
            features['int_acropole_fuel_rate'] = features['int_fuel_consumed_acropole_kg'] / (features['interval_duration_sec'] / 3600)
        
        # Fuel per distance
        if 'int_ground_distance_nm' in features and features.get('int_ground_distance_nm', 0) > 0:
            features['int_acropole_fuel_per_nm'] = features['int_fuel_consumed_acropole_kg'] / features['int_ground_distance_nm']
            features['int_specific_range'] = features['int_ground_distance_nm'] / (features['int_fuel_consumed_acropole_kg'] + 0.001)
    
    # OpenAP fuel efficiency metrics
    if 'int_fuel_consumed_openap_kg' in features:
        # OpenAP fuel per second
        if features.get('interval_duration_sec', 0) > 0:
            features['int_openap_fuel_rate'] = features['int_fuel_consumed_openap_kg'] / (features['interval_duration_sec'] / 3600)
        
        # OpenAP fuel per distance
        if 'int_ground_distance_nm' in features and features.get('int_ground_distance_nm', 0) > 0:
            features['int_openap_fuel_per_nm'] = features['int_fuel_consumed_openap_kg'] / features['int_ground_distance_nm']
    
    # Squared differences (penalize large errors more)
    if 'int_fuel_diff_acropole_openap_kg' in features:
        features['int_fuel_sq_diff_acropole_openap'] = features['int_fuel_diff_acropole_openap_kg'] ** 2
    
    # Agreement indicator (exponential decay with difference)
    if 'int_fuel_diff_acropole_openap_kg' in features:
        features['int_acropole_openap_agreement'] = np.exp(-np.abs(features['int_fuel_diff_acropole_openap_kg']) / 100)
    
    # Fuel normalized by mass
    if 'int_fuel_consumed_acropole_kg' in features and 'int_mass_mean' in features:
        if features.get('int_mass_mean', 0) > 0:
            features['int_acropole_fuel_per_mass'] = features['int_fuel_consumed_acropole_kg'] / features['int_mass_mean']
    
    # Fuel normalized by TOW
    tow_kg = flight_features.get('tow_pred_kg', 70000)
    if 'int_fuel_consumed_acropole_kg' in features:
        if tow_kg > 0:
            features['int_acropole_fuel_per_tow'] = features['int_fuel_consumed_acropole_kg'] / tow_kg
    
    # Energy proxies
    
    if 'altitude' in interval_df.columns and len(interval_df) >= 2:
        alt_change_m = (interval_df['altitude'].iloc[-1] - interval_df['altitude'].iloc[0]) * 0.3048
        dPE_J = tow_kg * G * alt_change_m
        features['int_dPE_MJ'] = dPE_J / 1e6
    
    if 'groundspeed' in interval_df.columns and len(interval_df) >= 2:
        v_start_ms = interval_df['groundspeed'].iloc[0] * 0.514444
        v_end_ms = interval_df['groundspeed'].iloc[-1] * 0.514444
        dKE_J = 0.5 * tow_kg * (v_end_ms**2 - v_start_ms**2)
        features['int_dKE_MJ'] = dKE_J / 1e6
    
    if 'int_dPE_MJ' in features and 'int_dKE_MJ' in features:
        features['int_total_energy_change_MJ'] = features['int_dPE_MJ'] + features['int_dKE_MJ']
    
    # Distance calculation
    if 'groundspeed' in interval_df.columns and 'timestamp' in interval_df.columns:
        time_diffs_hours = interval_df['timestamp'].diff().dt.total_seconds() / 3600
        features['int_ground_distance_nm'] = (interval_df['groundspeed'] * time_diffs_hours).sum()
    
    # Track changes (turning indicator)
    if 'track' in interval_df.columns:
        track_changes = interval_df['track'].diff().abs()
        track_changes = track_changes.apply(lambda x: min(x, 360 - x) if not np.isnan(x) else 0)
        features['int_total_track_change_deg'] = track_changes.sum()

    # Lift-to-Drag Ratio (L/D) proxy
    if 'int_tas_mean' in features and 'int_thrust_openap_mean' in features and 'int_fuel_flow_acropole_mean' in features:
        if features['int_thrust_openap_mean'] > 0 and features['int_fuel_flow_acropole_mean'] > 0:
            features['int_ld_ratio_proxy'] = features['int_tas_mean'] / (features['int_fuel_flow_acropole_mean'] / features['int_thrust_openap_mean'])

    # Fuel burn rate per phase (kg/h) - Acropole
    if fuel_col_acro in interval_df.columns and phase_fracs:
        for phase in ['climb', 'cruise', 'descent']:
            phase_mask = interval_df['phase'] == phase
            if phase_mask.any():
                phase_fuel_flow = interval_df.loc[phase_mask, fuel_col_acro].mean()
                features[f'int_fuel_rate_{phase}_acropole_kgh'] = phase_fuel_flow if not np.isnan(phase_fuel_flow) else 0.0
            else:
                features[f'int_fuel_rate_{phase}_acropole_kgh'] = 0.0

    # Fuel density corrections (adjust for temperature)
    if 'int_fuel_consumed_acropole_kg' in features and 'int_temperature_pl_mean' in features:
        temp_c = features['int_temperature_pl_mean'] - 273.15  # Convert K to C
        # Fuel density decreases with temperature: approx 0.8 kg/L at 15°C, adjust linearly
        density_adjustment = 0.8 * (1 - 0.0007 * (temp_c - 15))
        features['int_fuel_density_adjusted'] = density_adjustment
        features['int_fuel_consumed_density_corrected_kg'] = features['int_fuel_consumed_acropole_kg'] * (0.8 / density_adjustment)

    # Fuel efficiency vs. wind
    if 'int_acropole_fuel_per_nm' in features and 'int_headwind_mean' in features:
        features['int_fuel_efficiency_wind'] = features['int_acropole_fuel_per_nm'] / (features['int_headwind_mean'] + 1)

    # Altitude gradient (slope of altitude profile)
    if 'altitude' in interval_df.columns and len(interval_df) >= 2:
        from scipy import stats
        time_seconds = (interval_df['timestamp'] - interval_df['timestamp'].iloc[0]).dt.total_seconds()
        alt_ft = interval_df['altitude']
        if len(time_seconds) >= 2:
            if alt_ft.std() > 0:
                slope, _, _, _, _ = stats.linregress(time_seconds, alt_ft)
                features['int_alt_gradient_ft_per_sec'] = slope
            else:
                features['int_alt_gradient_ft_per_sec'] = 0.0  # No change, slope is 0

    # Time-series autocorrelation
    if 'TAS' in interval_df.columns and len(interval_df) > 1:
        tas_series = interval_df['TAS']
        if tas_series.std() > 0:
            features['int_tas_autocorr_lag1'] = tas_series.autocorr(lag=1)
        else:
            features['int_tas_autocorr_lag1'] = 0.0  # No variability, no correlation
    else:
        features['int_tas_autocorr_lag1'] = 0.0  # Insufficient data
    
    if 'vertical_rate' in interval_df.columns and len(interval_df) > 1:
        vr_series = interval_df['vertical_rate']
        if vr_series.std() > 0:
            features['int_vr_autocorr_lag1'] = vr_series.autocorr(lag=1)
        else:
            features['int_vr_autocorr_lag1'] = 0.0  # No variability, no correlation
    else:
        features['int_vr_autocorr_lag1'] = 0.0  # Insufficient data

    # Higher-order polynomials
    if 'int_tas_mean' in features:
        features['int_tas_mean_sq'] = features['int_tas_mean'] ** 2
        features['int_tas_mean_cube'] = features['int_tas_mean'] ** 3
    if 'int_mass_mean' in features:
        features['int_mass_mean_sq'] = features['int_mass_mean'] ** 2
    if 'int_fuel_flow_acropole_mean' in features:
        features['int_ff_acropole_sq'] = features['int_fuel_flow_acropole_mean'] ** 2

    # Feature ratios
    if 'int_mass_mean' in features and 'int_altitude_mean' in features and features['int_altitude_mean'] > 0:
        features['int_mass_per_altitude'] = features['int_mass_mean'] / features['int_altitude_mean']
    if 'int_fuel_flow_acropole_mean' in features and 'int_density_mean' in features and features['int_density_mean'] > 0:
        features['int_ff_per_density'] = features['int_fuel_flow_acropole_mean'] / features['int_density_mean']
    if 'int_thrust_openap_mean' in features and 'int_mass_mean' in features and features['int_mass_mean'] > 0:
        features['int_thrust_per_mass'] = features['int_thrust_openap_mean'] / features['int_mass_mean']

    # ========================================================================
    # Atmosphere & Density Features
    # ========================================================================
    
    # ISA temperature deviation
    if 'temperature_pl' in interval_df.columns and 'altitude' in interval_df.columns:
        isa_devs = [
            calculate_isa_deviation(temp, alt)
            for temp, alt in zip(interval_df['temperature_pl'], interval_df['altitude'])
            if not np.isnan(temp) and not np.isnan(alt)
        ]
        if isa_devs:
            features['int_isa_deviation_mean_K'] = np.mean(isa_devs)
            features['int_isa_deviation_std_K'] = np.std(isa_devs)
            features['int_isa_deviation_min_K'] = np.min(isa_devs)
            features['int_isa_deviation_max_K'] = np.max(isa_devs)
    
    # Air density
    if all(col in interval_df.columns for col in ['pressure_hpa', 'temperature_pl']):
        specific_hum = interval_df.get('specific_humidity_pl', pd.Series([0.0] * len(interval_df)))
        densities = [
            calculate_air_density(p, t, h)
            for p, t, h in zip(interval_df['pressure_hpa'], interval_df['temperature_pl'], specific_hum)
            if not np.isnan(p) and not np.isnan(t)
        ]
        if densities:
            features['int_density_mean'] = np.mean(densities)
            features['int_density_std'] = np.std(densities)
            features['int_density_min'] = np.min(densities)
            features['int_density_max'] = np.max(densities)
            features['int_sigma_density'] = features['int_density_mean'] / 1.225  # Ratio to sea-level
    
    # Dynamic pressure
    if 'TAS' in interval_df.columns and 'int_density_mean' in features:
        density_mean = float(features['int_density_mean']) if pd.notna(features['int_density_mean']) else np.nan
        qbars = [
            calculate_dynamic_pressure(density_mean, float(tas))
            for tas in interval_df['TAS']
            if not np.isnan(tas) and pd.notna(density_mean)
        ]
        if qbars:
            features['int_qbar_mean'] = np.mean(qbars)
            features['int_qbar_std'] = np.std(qbars)
    
    # Specific humidity stats
    if 'specific_humidity_pl' in interval_df.columns:
        features['int_specific_humidity_mean'] = interval_df['specific_humidity_pl'].mean()
        features['int_specific_humidity_std'] = interval_df['specific_humidity_pl'].std()
    
    # ========================================================================
    # Wind Features
    # ========================================================================
    
    if all(col in interval_df.columns for col in ['u_component_of_wind_pl', 'v_component_of_wind_pl']):
        # Wind magnitude
        wind_speeds = np.sqrt(
            interval_df['u_component_of_wind_pl']**2 + 
            interval_df['v_component_of_wind_pl']**2
        ) * 1.94384  # Convert m/s to knots
        features['int_wind_speed_mean'] = wind_speeds.mean()
        features['int_wind_speed_std'] = wind_speeds.std()
        features['int_wind_speed_max'] = wind_speeds.max()
        
        # Headwind and crosswind components
        if 'track' in interval_df.columns:
            headwinds = []
            crosswinds = []
            for u, v, trk in zip(
                interval_df['u_component_of_wind_pl'],
                interval_df['v_component_of_wind_pl'],
                interval_df['track']
            ):
                if not any(np.isnan([u, v, trk])):
                    wind_comp = calculate_wind_components(u, v, trk)
                    headwinds.append(wind_comp['headwind_kts'])
                    crosswinds.append(wind_comp['crosswind_kts'])
            
            if headwinds:
                features['int_headwind_mean'] = np.mean(headwinds)
                features['int_headwind_std'] = np.std(headwinds)
                features['int_headwind_p10'] = np.percentile(headwinds, 10)
                features['int_headwind_p90'] = np.percentile(headwinds, 90)
                features['int_headwind_min'] = np.min(headwinds)
                features['int_headwind_max'] = np.max(headwinds)
                features['int_headwind_range'] = features['int_headwind_max'] - features['int_headwind_min']
                features['int_wind_shear_mag'] = features['int_headwind_p90'] - features['int_headwind_p10']
            
            if crosswinds:
                features['int_crosswind_mean'] = np.mean(crosswinds)
                features['int_crosswind_std'] = np.std(crosswinds)
                features['int_crosswind_max'] = np.max(crosswinds)
    
    # ========================================================================
    # Fuel Efficiency & Energy-Normalized Features
    # ========================================================================
    
    # Specific fuel consumption (distance per fuel)
    if 'int_ground_distance_nm' in features and 'int_fuel_consumed_acropole_kg' in features:
        if features['int_fuel_consumed_acropole_kg'] > 0.1:  # Avoid division by near-zero
            features['int_nm_per_kg'] = features['int_ground_distance_nm'] / features['int_fuel_consumed_acropole_kg']
            features['int_kg_per_nm'] = features['int_fuel_consumed_acropole_kg'] / max(features['int_ground_distance_nm'], 0.01)
    
    # Fuel normalized by TOW/mass
    tow_kg = flight_features.get('tow_pred_kg', 70000)
    if 'int_fuel_consumed_acropole_kg' in features:
        if tow_kg > 0:
            features['int_acropole_fuel_per_tow'] = features['int_fuel_consumed_acropole_kg'] / tow_kg
        
        if 'int_mass_mean' in features and features.get('int_mass_mean', 0) > 0:
            features['int_acropole_fuel_per_mass'] = features['int_fuel_consumed_acropole_kg'] / features['int_mass_mean']
    
    # Energy change per fuel
    if 'int_total_energy_change_MJ' in features and 'int_fuel_consumed_acropole_kg' in features:
        if features['int_fuel_consumed_acropole_kg'] > 0.1:
            features['int_MJ_per_kg'] = features['int_total_energy_change_MJ'] / features['int_fuel_consumed_acropole_kg']
    
    # Power proxy
    if 'int_total_energy_change_MJ' in features and features.get('interval_duration_sec', 0) > 0:
        features['int_power_MW'] = features['int_total_energy_change_MJ'] / (features['interval_duration_sec'] / 60.0)
    
    if 'int_power_MW' in features and 'int_fuel_flow_acropole_mean' in features and features.get('int_fuel_flow_acropole_mean', 0) > 0:
        features['int_power_per_acropole_fuelflow'] = features['int_power_MW'] / features['int_fuel_flow_acropole_mean']
    
    # ========================================================================
    # Flight Phase Dynamics
    # ========================================================================
    
    # Dominant phase
    if phase_fracs:
        features['int_phase_mode'] = max(phase_fracs.items(), key=lambda kv: kv[1])[0]
    
    # Phase transitions
    if 'vertical_rate' in interval_df.columns and 'altitude' in interval_df.columns:
        phases = interval_df.apply(
            lambda row: get_flight_phase(row['vertical_rate'], row['altitude']),
            axis=1
        )
        phase_changes = (phases != phases.shift()).sum() - 1  # Subtract 1 for first point
        features['int_num_phase_changes'] = max(0, phase_changes)
        
        # Detect TOC (climb to cruise) and TOD (cruise to descent)
        phase_list = phases.tolist()
        features['int_contains_toc'] = ('climb' in phase_list and 'cruise' in phase_list)
        features['int_contains_tod'] = ('cruise' in phase_list and 'descent' in phase_list)
    
    # Vertical profile shape
    if 'altitude' in interval_df.columns:
        features['int_altitude_range_ft'] = interval_df['altitude'].max() - interval_df['altitude'].min()
    
    if 'vertical_rate' in interval_df.columns:
        features['int_vertical_rate_range_fpm'] = interval_df['vertical_rate'].max() - interval_df['vertical_rate'].min()
    
    # Flight path angle
    if 'vertical_rate' in interval_df.columns and 'TAS' in interval_df.columns:
        # Convert vertical rate from fpm to m/s
        v_vert_ms = interval_df['vertical_rate'] * 0.00508  # fpm to m/s
        v_horiz_ms = interval_df['TAS'] * 0.514444  # knots to m/s
        
        # Calculate FPA in degrees
        fpa_rad = np.arctan2(v_vert_ms, v_horiz_ms)
        fpa_deg = np.rad2deg(fpa_rad)
        
        features['int_fpa_mean_deg'] = fpa_deg.mean()
        features['int_fpa_std_deg'] = fpa_deg.std()
    
    # ========================================================================
    # Data Quality / Reconstruction Features
    # ========================================================================
    
    if 'is_reconstructed' in interval_df.columns:
        total_points = len(interval_df)
        recon_points = interval_df['is_reconstructed'].sum()
        features['int_frac_reconstructed'] = recon_points / total_points if total_points > 0 else 0.0
    
    # Sampling quality
    if not dt_seconds.empty and len(interval_df) > 1:
        features['int_mean_dt_sec'] = float(dt_seconds.mean())
        features['int_dt_std_sec'] = float(dt_seconds.std())
    
    # ========================================================================
    # Time-of-Flight and Calendar Effects
    # ========================================================================
    
    # Time of day (cyclical encoding)
    if 'interval_start_ts' in features and pd.notna(features['interval_start_ts']):
        hour = features['interval_start_ts'].hour + features['interval_start_ts'].minute / 60.0
        features['int_hour_of_day_sin'] = np.sin(2 * np.pi * hour / 24.0)
        features['int_hour_of_day_cos'] = np.cos(2 * np.pi * hour / 24.0)
        features['int_day_of_week'] = features['interval_start_ts'].dayofweek  # 0=Monday
    
    # ========================================================================
    # Advanced Variability & Smoothness Features
    # ========================================================================
    
    # TAS variability and acceleration patterns
    if 'TAS' in interval_df.columns and len(interval_df) > 1:
        tas_diff = interval_df['TAS'].diff()
        features['int_tas_acceleration_mean'] = tas_diff.mean()
        features['int_tas_acceleration_std'] = tas_diff.std()
        features['int_tas_acceleration_max'] = tas_diff.max()
        features['int_tas_deceleration_min'] = tas_diff.min()
        features['int_tas_variability'] = interval_df['TAS'].std() / (interval_df['TAS'].mean() + 1.0)
    
    # Altitude variability and climb patterns
    if 'altitude' in interval_df.columns and len(interval_df) > 1:
        alt_diff = interval_df['altitude'].diff()
        features['int_altitude_rate_change_mean'] = alt_diff.mean()
        features['int_altitude_rate_change_std'] = alt_diff.std()
        features['int_altitude_smoothness'] = interval_df['altitude'].std() / (interval_df['altitude'].mean() + 1.0)
    
    # Vertical rate variability
    if 'vertical_rate' in interval_df.columns:
        features['int_vertical_rate_variability'] = interval_df['vertical_rate'].std() / (np.abs(interval_df['vertical_rate'].mean()) + 1.0)
    
    # Fuel flow variability (Acropole)
    if fuel_col_acro in interval_df.columns:
        series_acro = interval_df[fuel_col_acro]
        if not series_acro.empty:
            features['int_ff_acropole_std'] = float(series_acro.std())
            features['int_ff_acropole_range'] = float(series_acro.max() - series_acro.min())
            denom = features.get('int_fuel_flow_acropole_mean', 1.0) + 0.01
            features['int_ff_acropole_cv'] = features['int_ff_acropole_std'] / denom if denom else np.nan
    
    # Fuel flow variability (OpenAP)
    if fuel_col_openap in interval_df.columns:
        series_openap = interval_df[fuel_col_openap]
        if not series_openap.empty:
            features['int_ff_openap_std'] = float(series_openap.std())
            features['int_ff_openap_range'] = float(series_openap.max() - series_openap.min())
            denom = features.get('int_fuel_flow_openap_mean', 1.0) + 0.01
            features['int_ff_openap_cv'] = features['int_ff_openap_std'] / denom if denom else np.nan
    
    # ========================================================================
    # Speed × Altitude Interaction (Energy State)
    # ========================================================================
    
    if 'TAS' in interval_df.columns and 'altitude' in interval_df.columns:
        # Mean speed-altitude product (energy state proxy)
        speed_alt_product = interval_df['TAS'] * interval_df['altitude']
        features['int_speed_alt_mean'] = speed_alt_product.mean()
        features['int_speed_alt_std'] = speed_alt_product.std()
        features['int_speed_alt_max'] = speed_alt_product.max()
    
    # ========================================================================
    # Climb-Descent Ratio and Balance
    # ========================================================================
    
    if 'int_frac_climb' in features and 'int_frac_descent' in features:
        if features['int_frac_descent'] > 0:
            features['int_climb_descent_ratio'] = features['int_frac_climb'] / features['int_frac_descent']
        else:
            features['int_climb_descent_ratio'] = features['int_frac_climb'] * 10  # Large value if no descent
    
    # ========================================================================
    # Weighted Cruise Altitude (Duration-Weighted)
    # ========================================================================
    
    # This would require segment-level duration data, which we'll compute from timestamps
    # For now, use simple altitude weighting by time in cruise phase
    if 'altitude' in interval_df.columns and 'vertical_rate' in interval_df.columns and len(interval_df) > 1:
        # Identify cruise points (low vertical rate)
        cruise_mask = np.abs(interval_df['vertical_rate']) < 200  # fpm threshold
        if cruise_mask.sum() > 0:
            cruise_altitudes = interval_df.loc[cruise_mask, 'altitude']
            features['int_weighted_cruise_alt_ft'] = cruise_altitudes.mean()
            features['int_cruise_alt_std_ft'] = cruise_altitudes.std()
        else:
            features['int_weighted_cruise_alt_ft'] = features.get('int_altitude_mean', np.nan)
            features['int_cruise_alt_std_ft'] = 0.0
    
    # ========================================================================
    # A. Distance & Geometry Features
    # ========================================================================
    
    # Horizontal geometry
    if 'latitude' in interval_df.columns and 'longitude' in interval_df.columns and len(interval_df) >= 2:
        # Great-circle distance between start and end points
        lat1, lon1 = interval_df['latitude'].iloc[0], interval_df['longitude'].iloc[0]
        lat2, lon2 = interval_df['latitude'].iloc[-1], interval_df['longitude'].iloc[-1]
        if not any(np.isnan([lat1, lon1, lat2, lon2])):
            features['int_haversine_start_end_nm'] = haversine_nm(lat1, lon1, lat2, lon2)
            
            # Path inefficiency ratio
            if 'int_ground_distance_nm' in features and features['int_haversine_start_end_nm'] > 0.1:
                features['int_path_inefficiency_ratio'] = features['int_ground_distance_nm'] / features['int_haversine_start_end_nm']
    
    # Track statistics
    if 'track' in interval_df.columns:
        features['int_mean_track_deg'] = interval_df['track'].mean()
        features['int_track_std_deg'] = interval_df['track'].std()
        features['int_track_min_deg'] = interval_df['track'].min()
        features['int_track_max_deg'] = interval_df['track'].max()
        
        # Turn rates
        if len(interval_df) > 1:
            track_diffs = interval_df['track'].diff()
            time_diffs = interval_df['timestamp'].diff().dt.total_seconds()
            
            # Handle track angle wrapping (0-360 degrees)
            track_diffs = track_diffs.apply(lambda x: min(x % 360, 360 - (x % 360)) if not np.isnan(x) else 0)
            track_diffs = track_diffs.apply(lambda x: x if x <= 180 else x - 360)  # Handle large turns
            
            valid_mask = (~np.isnan(track_diffs)) & (~np.isnan(time_diffs)) & (time_diffs > 0)
            if valid_mask.any():
                turn_rates = track_diffs[valid_mask] / time_diffs[valid_mask]
                features['int_mean_turn_rate_deg_per_sec'] = turn_rates.mean()
                features['int_max_turn_rate_deg_per_sec'] = np.abs(turn_rates).max()
    
    # Direction-specific distance decomposition
    if 'track' in interval_df.columns and 'int_ground_distance_nm' in features and len(interval_df) >= 2:
        # Use track at interval midpoint for approximation
        mid_idx = len(interval_df) // 2
        mid_track = interval_df['track'].iloc[mid_idx]
        
        if not np.isnan(mid_track):
            # Approximate along-track vs cross-track components
            # This is a simplification - in reality would need proper vector decomposition
            total_distance = features['int_ground_distance_nm']
            
            # For now, assume most movement is along-track
            # A more accurate implementation would integrate along the path
            features['int_alongtrack_distance_nm'] = total_distance * 0.95  # Conservative estimate
            features['int_crosstrack_distance_nm'] = total_distance * 0.05   # Conservative estimate
    
    # Vertical geometry
    if 'altitude' in interval_df.columns and len(interval_df) >= 2:
        alt_start = interval_df['altitude'].iloc[0]
        alt_end = interval_df['altitude'].iloc[-1]
        
        features['int_net_vertical_distance_ft'] = abs(alt_end - alt_start)  # Rename for clarity (net displacement)
        
        # Climb and descent distances
        alt_changes = interval_df['altitude'].diff()
        climb_distance = alt_changes[alt_changes > 0].sum()
        descent_distance = abs(alt_changes[alt_changes < 0].sum())
        
        features['int_climb_distance_ft'] = climb_distance if not np.isnan(climb_distance) else 0
        features['int_descent_distance_ft'] = descent_distance if not np.isnan(descent_distance) else 0
        
        # Vertical fractions (fix: divide by total vertical movement, not net)
        total_vertical_movement = climb_distance + descent_distance
        if total_vertical_movement > 0:
            features['int_climb_vertical_fraction'] = climb_distance / total_vertical_movement
            features['int_descent_vertical_fraction'] = descent_distance / total_vertical_movement
        else:
            features['int_climb_vertical_fraction'] = 0.0
            features['int_descent_vertical_fraction'] = 0.0
    
    # ========================================================================
    # B. Speed Regime Features
    # ========================================================================
    
    # Process each speed variable
    speed_vars = ['groundspeed', 'TAS']
    if 'CAS' in interval_df.columns:
        speed_vars.append('CAS')
    
    for speed_var in speed_vars:
        if speed_var in interval_df.columns:
            speed_series = interval_df[speed_var]
            prefix = f'int_{speed_var.lower()}'
            
            # Quantiles & extremes
            features[f'{prefix}_p05'] = speed_series.quantile(0.05)
            features[f'{prefix}_p25'] = speed_series.quantile(0.25)
            features[f'{prefix}_p50'] = speed_series.quantile(0.50)
            features[f'{prefix}_p75'] = speed_series.quantile(0.75)
            features[f'{prefix}_p95'] = speed_series.quantile(0.95)
            features[f'{prefix}_range'] = speed_series.max() - speed_series.min()
            features[f'{prefix}_iqr'] = features[f'{prefix}_p75'] - features[f'{prefix}_p25']
            
            # Statistical moments
            features[f'{prefix}_skew'] = speed_series.skew()
            features[f'{prefix}_kurtosis'] = speed_series.kurtosis()
            
            # Time fractions in regimes
            total_points = len(speed_series)
            if total_points > 0:
                # Low speed (< 250 kt)
                features[f'{prefix}_frac_low'] = (speed_series < 250).sum() / total_points
                # Mid speed (250-450 kt)
                features[f'{prefix}_frac_mid'] = ((speed_series >= 250) & (speed_series < 450)).sum() / total_points
                # High speed (>= 450 kt)
                features[f'{prefix}_frac_high'] = (speed_series >= 450).sum() / total_points
    
    # Mach number regimes (if TAS and altitude available for Mach calculation)
    if 'TAS' in interval_df.columns and 'altitude' in interval_df.columns:
        # Simplified Mach calculation (would need proper atmospheric model)
        # For now, approximate Mach from TAS and altitude
        alt_km = interval_df['altitude'] * 0.0003048  # ft to km
        # Rough approximation: Mach = TAS / (661 * sqrt(T + 273.15) / 15) where T is in °C
        temp_approx = 15 - 6.5 * alt_km  # Simplified ISA
        speed_sound_approx = 661 * np.sqrt(temp_approx + 273.15) / 15  # knots
        mach_numbers = interval_df['TAS'] / speed_sound_approx
        
        features['int_mach_mean'] = mach_numbers.mean()
        features['int_mach_std'] = mach_numbers.std()
        features['int_mach_max'] = mach_numbers.max()
        
        # Mach regime fractions
        total_points = len(mach_numbers)
        if total_points > 0:
            features['int_frac_mach_subsonic'] = (mach_numbers < 0.7).sum() / total_points
            features['int_frac_mach_transonic'] = ((mach_numbers >= 0.7) & (mach_numbers < 0.8)).sum() / total_points
            features['int_frac_mach_supersonic'] = (mach_numbers >= 0.8).sum() / total_points
    
    # Acceleration statistics (separate accel/decel)
    if 'TAS' in interval_df.columns and len(interval_df) > 1:
        tas_accel = interval_df['TAS'].diff()
        valid_accel = tas_accel.dropna()
        
        if len(valid_accel) > 0:
            pos_accel = valid_accel[valid_accel > 0]
            neg_accel = valid_accel[valid_accel < 0]
            
            if len(pos_accel) > 0:
                features['int_tas_accel_pos_mean'] = pos_accel.mean()
                features['int_tas_accel_pos_std'] = pos_accel.std()
                features['int_tas_accel_pos_max'] = pos_accel.max()
            
            if len(neg_accel) > 0:
                features['int_tas_accel_neg_mean'] = neg_accel.mean()
                features['int_tas_accel_neg_std'] = neg_accel.std()
                features['int_tas_accel_neg_min'] = neg_accel.min()
            
            # Speed jump counts
            features['int_num_speed_jumps_gt_10kt'] = (np.abs(valid_accel) > 10).sum()
            features['int_num_speed_jumps_gt_20kt'] = (np.abs(valid_accel) > 20).sum()
    
    # ========================================================================
    # C. Vertical Rate & Altitude Structure
    # ========================================================================
    
    # Level-off detection
    if 'vertical_rate' in interval_df.columns and 'timestamp' in interval_df.columns:
        vr_series = interval_df['vertical_rate']
        time_diffs = interval_df['timestamp'].diff().dt.total_seconds()
        
        # Find segments where |VR| < 100 fpm for > N seconds
        level_off_mask = np.abs(vr_series) < 100
        level_off_durations = []
        current_duration = 0
        
        for i in range(len(level_off_mask)):
            if level_off_mask.iloc[i]:
                current_duration += time_diffs.iloc[i] if i > 0 else 0
            else:
                if current_duration > 30:  # > 30 seconds
                    level_off_durations.append(current_duration)
                current_duration = 0
        
        # Check last segment
        if current_duration > 30:
            level_off_durations.append(current_duration)
        
        features['int_num_leveloffs'] = len(level_off_durations)
        features['int_total_leveloff_time_sec'] = sum(level_off_durations) if level_off_durations else 0
        features['int_frac_leveloff_time'] = features['int_total_leveloff_time_sec'] / features.get('interval_duration_sec', 1) if features.get('interval_duration_sec', 0) > 0 else 0
    
    # Climb/descent episode metrics
    if 'vertical_rate' in interval_df.columns:
        vr_series = interval_df['vertical_rate']
        
        # Identify episodes (sequences of similar sign VR)
        vr_sign = np.sign(vr_series)
        episode_changes = np.where(vr_sign != np.roll(vr_sign, 1))[0]
        
        climb_episodes = []
        descent_episodes = []
        
        for start_idx in range(len(episode_changes) - 1):
            end_idx = episode_changes[start_idx + 1]
            episode_vr = vr_series.iloc[start_idx:end_idx]
            
            if len(episode_vr) > 5:  # Minimum episode length
                mean_vr = episode_vr.mean()
                if mean_vr > 200:  # Climb
                    climb_episodes.append(mean_vr)
                elif mean_vr < -200:  # Descent
                    descent_episodes.append(mean_vr)
        
        features['int_num_climb_episodes'] = len(climb_episodes)
        features['int_num_descent_episodes'] = len(descent_episodes)
        
        if climb_episodes:
            features['int_mean_climb_rate_fpm'] = np.mean(climb_episodes)
            features['int_max_climb_rate_fpm'] = np.max(climb_episodes)
        
        if descent_episodes:
            features['int_mean_descent_rate_fpm'] = np.mean(descent_episodes)
            features['int_max_descent_rate_fpm'] = np.max(descent_episodes)
    
    # Altitude band occupancy
    if 'altitude' in interval_df.columns:
        alt_series = interval_df['altitude']
        total_points = len(alt_series)
        
        if total_points > 0:
            features['int_frac_alt_below_10k'] = (alt_series < 10000).sum() / total_points
            features['int_frac_alt_10_20k'] = ((alt_series >= 10000) & (alt_series < 20000)).sum() / total_points
            features['int_frac_alt_20_30k'] = ((alt_series >= 20000) & (alt_series < 30000)).sum() / total_points
            features['int_frac_alt_above_30k'] = (alt_series >= 30000).sum() / total_points
    
    # ========================================================================
    # D. Track/Heading & Wind Interaction
    # ========================================================================
    
    # Turn structure
    if 'track' in interval_df.columns and len(interval_df) > 1:
        track_diffs = interval_df['track'].diff()
        # Handle angle wrapping
        track_diffs = track_diffs.apply(lambda x: min(x % 360, 360 - (x % 360)) if not np.isnan(x) else 0)
        track_diffs = track_diffs.apply(lambda x: x if x <= 180 else x - 360)
        
        abs_turns = np.abs(track_diffs.dropna())
        features['int_num_turns_gt_5deg'] = (abs_turns > 5).sum()
        features['int_num_turns_gt_10deg'] = (abs_turns > 10).sum()
        features['int_num_turns_gt_25deg'] = (abs_turns > 25).sum()
        
        # Split by direction
        left_turns = track_diffs[track_diffs > 0].sum()
        right_turns = abs(track_diffs[track_diffs < 0].sum())
        total_turns = left_turns + right_turns
        
        features['int_total_left_turn_deg'] = left_turns
        features['int_total_right_turn_deg'] = right_turns
        
        if total_turns > 0:
            features['int_left_right_turn_ratio'] = left_turns / total_turns
    
    # Wind interaction
    if 'int_headwind_mean' in features and 'int_tas_mean' in features:
        # Correlation between headwind and TAS
        if 'int_headwind_std' in features and 'int_tas_std' in features and features['int_headwind_std'] > 0 and features['int_tas_std'] > 0:
            # Would need point-wise data for proper correlation - approximate with available stats
            features['int_corr_headwind_tas'] = 0.0  # Placeholder - would need full time series
        
        # Effective TAS (TAS corrected for headwind)
        features['int_effective_TAS'] = features['int_tas_mean'] - features['int_headwind_mean']
    
    # Time fractions with strong winds
    if 'int_headwind_mean' in features:
        # These would need point-wise wind data - using approximations
        features['int_frac_headwind_gt_40kts'] = 0.0  # Placeholder
        features['int_frac_tailwind_gt_40kts'] = 0.0  # Placeholder
    
    # Crosswind statistics
    if 'int_crosswind_mean' in features:
        features['int_frac_crosswind_gt_20kts'] = 0.0  # Placeholder
        features['int_crosswind_p95'] = features['int_crosswind_mean'] * 1.5  # Approximation
    
    return features


# ============================================================================
# Feature Extraction: Segment Profile
# ============================================================================

# ============================================================================
# Feature Extraction: Segment Profile
# ============================================================================

def compute_cross_segment_features(row_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute cross-segment transition and aggregate features
    
    This adds features that look at relationships between segments:
    - Segment-to-segment transitions (velocity, altitude changes)
    - Acceleration/deceleration patterns
    - Thrust-drag interactions per segment
    - Aggregate statistics across all segments
    """
    new_features = {}
    
    # Get number of segments from existing features
    seg_nums = list(range(NUM_SEGMENTS))
    
    # ========================================================================
    # CATEGORY 1: Segment-to-Segment Transitions
    # ========================================================================
    
    # Delta TAS between consecutive segments
    tas_deltas = []
    for i in range(len(seg_nums) - 1):
        seg_curr = f'seg{seg_nums[i]:02d}_groundspeed_mean'
        seg_next = f'seg{seg_nums[i+1]:02d}_groundspeed_mean'
        if seg_curr in row_features and seg_next in row_features:
            val_curr = row_features.get(seg_curr, np.nan)
            val_next = row_features.get(seg_next, np.nan)
            if not np.isnan(val_curr) and not np.isnan(val_next):
                delta = val_next - val_curr
                new_features[f'delta_gs_{i:02d}_{i+1:02d}'] = delta
                tas_deltas.append(delta)
    
    # Delta Altitude between consecutive segments
    alt_deltas = []
    for i in range(len(seg_nums) - 1):
        seg_curr = f'seg{seg_nums[i]:02d}_altitude_mean'
        seg_next = f'seg{seg_nums[i+1]:02d}_altitude_mean'
        if seg_curr in row_features and seg_next in row_features:
            val_curr = row_features.get(seg_curr, np.nan)
            val_next = row_features.get(seg_next, np.nan)
            if not np.isnan(val_curr) and not np.isnan(val_next):
                delta = val_next - val_curr
                new_features[f'delta_alt_{i:02d}_{i+1:02d}'] = delta
                alt_deltas.append(delta)
    
    # ========================================================================
    # CATEGORY 2: Acceleration Patterns
    # ========================================================================
    
    if len(tas_deltas) > 0:
        new_features['mean_gs_acceleration'] = np.mean(tas_deltas)
        new_features['max_gs_acceleration'] = np.max(tas_deltas)
        new_features['min_gs_deceleration'] = np.min(tas_deltas)
        new_features['gs_acceleration_std'] = np.std(tas_deltas)
    
    # ========================================================================
    # CATEGORY 3: Thrust × TAS and Drag × TAS per segment
    # ========================================================================
    
    thrust_power_list = []
    drag_power_list = []
    
    for seg_num in seg_nums:
        seg_str = f'{seg_num:02d}'
        thrust_col = f'seg{seg_str}_thrust_openap_mean'
        drag_col = f'seg{seg_str}_drag_openap_mean'
        gs_col = f'seg{seg_str}_groundspeed_mean'
        
        if thrust_col in row_features and gs_col in row_features:
            thrust = row_features.get(thrust_col, np.nan)
            gs = row_features.get(gs_col, np.nan)
            if not np.isnan(thrust) and not np.isnan(gs):
                thrust_power = thrust * gs
                new_features[f'thrust_power_{seg_str}'] = thrust_power
                thrust_power_list.append(thrust_power)
        
        if drag_col in row_features and gs_col in row_features:
            drag = row_features.get(drag_col, np.nan)
            gs = row_features.get(gs_col, np.nan)
            if not np.isnan(drag) and not np.isnan(gs):
                drag_power = drag * gs
                new_features[f'drag_power_{seg_str}'] = drag_power
                drag_power_list.append(drag_power)
    
    # Aggregate thrust and drag power
    if len(thrust_power_list) > 0:
        new_features['total_thrust_power'] = np.sum(thrust_power_list)
        new_features['avg_thrust_power'] = np.mean(thrust_power_list)
        new_features['thrust_power_std'] = np.std(thrust_power_list)
    
    if len(drag_power_list) > 0:
        new_features['total_drag_power'] = np.sum(drag_power_list)
        new_features['avg_drag_power'] = np.mean(drag_power_list)
        new_features['drag_power_std'] = np.std(drag_power_list)
    
    # ========================================================================
    # CATEGORY 4: Fuel Flow Variability Across Segments
    # ========================================================================
    
    ff_acropole_list = []
    ff_openap_list = []
    
    for seg_num in seg_nums:
        seg_str = f'{seg_num:02d}'
        ff_acropole_col = f'seg{seg_str}_fuel_flow_acropole_mean'
        ff_openap_col = f'seg{seg_str}_fuel_flow_openap_mean'
        
        if ff_acropole_col in row_features:
            val = row_features.get(ff_acropole_col, np.nan)
            if not np.isnan(val):
                ff_acropole_list.append(val)
        
        if ff_openap_col in row_features:
            val = row_features.get(ff_openap_col, np.nan)
            if not np.isnan(val):
                ff_openap_list.append(val)
    
    if len(ff_acropole_list) > 0:
        new_features['seg_ff_acropole_std'] = np.std(ff_acropole_list)
        new_features['seg_ff_acropole_range'] = np.max(ff_acropole_list) - np.min(ff_acropole_list)
        new_features['seg_ff_acropole_cv'] = np.std(ff_acropole_list) / (np.mean(ff_acropole_list) + 0.01)
    
    if len(ff_openap_list) > 0:
        new_features['seg_ff_openap_std'] = np.std(ff_openap_list)
        new_features['seg_ff_openap_range'] = np.max(ff_openap_list) - np.min(ff_openap_list)
        new_features['seg_ff_openap_cv'] = np.std(ff_openap_list) / (np.mean(ff_openap_list) + 0.01)
    
    # ========================================================================
    # CATEGORY 5: Vertical Rate Variability
    # ========================================================================
    
    vr_list = []
    for seg_num in seg_nums:
        seg_str = f'{seg_num:02d}'
        vr_col = f'seg{seg_str}_vertical_rate_mean'
        if vr_col in row_features:
            val = row_features.get(vr_col, np.nan)
            if not np.isnan(val):
                vr_list.append(val)
    
    if len(vr_list) > 0:
        new_features['seg_vr_std'] = np.std(vr_list)
        new_features['seg_vr_range'] = np.max(vr_list) - np.min(vr_list)
    
    # ========================================================================
    # CATEGORY 6: Mass Variability
    # ========================================================================
    
    mass_acropole_list = []
    mass_openap_list = []
    
    for seg_num in seg_nums:
        seg_str = f'{seg_num:02d}'
        mass_acropole_col = f'seg{seg_str}_mass_mean'
        mass_openap_col = f'seg{seg_str}_mass_openap_mean'
        
        if mass_acropole_col in row_features:
            val = row_features.get(mass_acropole_col, np.nan)
            if not np.isnan(val):
                mass_acropole_list.append(val)
        
        if mass_openap_col in row_features:
            val = row_features.get(mass_openap_col, np.nan)
            if not np.isnan(val):
                mass_openap_list.append(val)
    
    if len(mass_acropole_list) > 0:
        new_features['seg_mass_acropole_range'] = np.max(mass_acropole_list) - np.min(mass_acropole_list)
        new_features['seg_mass_acropole_std'] = np.std(mass_acropole_list)
    
    if len(mass_openap_list) > 0:
        new_features['seg_mass_openap_range'] = np.max(mass_openap_list) - np.min(mass_openap_list)
        new_features['seg_mass_openap_std'] = np.std(mass_openap_list)
    
    # ========================================================================
    # (d) Cross-segment "shape" features
    # ========================================================================
    
    # Fuel distribution over segments
    ff_acropole_list = []
    ff_openap_list = []
    for seg_num in seg_nums:
        seg_str = f'{seg_num:02d}'
        ff_acro_col = f'seg{seg_str}_fuel_flow_acropole_mean'
        ff_openap_col = f'seg{seg_str}_fuel_flow_openap_mean'
        
        if ff_acro_col in row_features:
            val = row_features.get(ff_acro_col, np.nan)
            if not np.isnan(val):
                ff_acropole_list.append(val)
        
        if ff_openap_col in row_features:
            val = row_features.get(ff_openap_col, np.nan)
            if not np.isnan(val):
                ff_openap_list.append(val)
    
    # Fuel share in first vs last half
    if len(ff_acropole_list) >= 4:  # Need at least 4 segments for meaningful split
        mid_point = len(ff_acropole_list) // 2
        first_half = ff_acropole_list[:mid_point]
        last_half = ff_acropole_list[mid_point:]
        
        total_ff = sum(ff_acropole_list)
        if total_ff > 0:
            new_features['fuel_share_first_half'] = sum(first_half) / total_ff
            new_features['fuel_share_last_half'] = sum(last_half) / total_ff
        
        # Fuel center of mass index
        weighted_sum = sum((i + 1) * ff for i, ff in enumerate(ff_acropole_list))
        if total_ff > 0:
            new_features['fuel_com_index'] = weighted_sum / total_ff
    
    # Altitude profile shape
    alt_list = []
    for seg_num in seg_nums:
        seg_str = f'{seg_num:02d}'
        alt_col = f'seg{seg_str}_altitude_mean'
        if alt_col in row_features:
            val = row_features.get(alt_col, np.nan)
            if not np.isnan(val):
                alt_list.append(val)
    
    if len(alt_list) >= 3:  # Need at least 3 points for meaningful regression
        # Linear regression slope of altitude vs segment index
        x = np.arange(len(alt_list))
        y = np.array(alt_list)
        try:
            slope, intercept = np.polyfit(x, y, 1)
            new_features['alt_trend_per_interval'] = slope
        except (np.linalg.LinAlgError, ValueError):
            pass  # Skip if regression fails
        
        # Peak and valley segments
        if len(alt_list) > 0:
            alt_array = np.array(alt_list)
            peak_idx = np.argmax(alt_array)
            valley_idx = np.argmin(alt_array)
            new_features['alt_peak_segment'] = peak_idx
            new_features['alt_valley_segment'] = valley_idx
    
    # Multiplier profile variability
    acropole_mult_list = []
    openap_mult_list = []
    for seg_num in seg_nums:
        seg_str = f'{seg_num:02d}'
        acro_mult_col = f'seg{seg_str}_acropole_multiplier_mean'
        openap_mult_col = f'seg{seg_str}_openap_multiplier_mean'
        
        if acro_mult_col in row_features:
            val = row_features.get(acro_mult_col, np.nan)
            if not np.isnan(val):
                acropole_mult_list.append(val)
        
        if openap_mult_col in row_features:
            val = row_features.get(openap_mult_col, np.nan)
            if not np.isnan(val):
                openap_mult_list.append(val)
    
    if len(acropole_mult_list) > 1:
        new_features['seg_acropole_multiplier_std'] = np.std(acropole_mult_list)
        new_features['seg_acropole_multiplier_range'] = np.max(acropole_mult_list) - np.min(acropole_mult_list)
    
    if len(openap_mult_list) > 1:
        new_features['seg_openap_multiplier_std'] = np.std(openap_mult_list)
        new_features['seg_openap_multiplier_range'] = np.max(openap_mult_list) - np.min(openap_mult_list)
    
    return new_features


def extract_segment_features(
    df: pd.DataFrame,
    interval_start_ts: pd.Timestamp,
    interval_end_ts: pd.Timestamp,
    flight_features: Optional[Dict[str, Any]] = None,
    num_segments: int = NUM_SEGMENTS
) -> Dict[str, Any]:
    """
    Extract time-slice features within an interval
    Creates a profile of the interval by dividing it into segments
    """
    
    segment_duration = (interval_end_ts - interval_start_ts) / num_segments
    segment_features = {}
    
    # Handle empty DataFrame or missing timestamp column
    if df.empty or 'timestamp' not in df.columns:
        for seg_idx in range(num_segments):
            prefix = f"seg{seg_idx:02d}"
            segment_features[f"{prefix}_has_data"] = False
        return segment_features
    
    aircraft_type = (flight_features or {}).get('aircraft_type', 'UNKNOWN')
    fuel_col_acro = select_fuel_flow_column(df, aircraft_type, 'acropole')
    fuel_col_openap = select_fuel_flow_column(df, aircraft_type, 'openap')

    for seg_idx in range(num_segments):
        seg_start_ts = interval_start_ts + segment_duration * seg_idx
        seg_end_ts = seg_start_ts + segment_duration
        
        mask = (df['timestamp'] >= seg_start_ts) & (df['timestamp'] <= seg_end_ts)
        seg_df = df[mask]
        
        prefix = f"seg{seg_idx:02d}"
        
        if len(seg_df) < 2:
            segment_features[f"{prefix}_has_data"] = False
            continue
        
        segment_features[f"{prefix}_has_data"] = True
        # Removed: segment_features[f"{prefix}_num_points"] = len(seg_df)
        
        # Key features for each segment
        if 'altitude' in seg_df.columns:
            segment_features[f"{prefix}_altitude_mean"] = seg_df['altitude'].mean()
        
        if 'groundspeed' in seg_df.columns:
            segment_features[f"{prefix}_groundspeed_mean"] = seg_df['groundspeed'].mean()
        
        if 'vertical_rate' in seg_df.columns:
            segment_features[f"{prefix}_vertical_rate_mean"] = seg_df['vertical_rate'].mean()
        
        # Add VR fallback for segments
        if f"{prefix}_vertical_rate_mean" not in segment_features or pd.isna(segment_features[f"{prefix}_vertical_rate_mean"]):
            # Fallback: Calculate VR from altitude change
            if 'altitude' in seg_df.columns and len(seg_df) >= 2:
                alt_start = seg_df['altitude'].iloc[0]
                alt_end = seg_df['altitude'].iloc[-1]
                duration_sec = (seg_df['timestamp'].max() - seg_df['timestamp'].min()).total_seconds()
                if duration_sec > 0:
                    vr_approx = (alt_end - alt_start) / duration_sec * 60  # fpm
                    segment_features[f"{prefix}_vertical_rate_mean"] = vr_approx
                else:
                    segment_features[f"{prefix}_vertical_rate_mean"] = 0.0
            else:
                segment_features[f"{prefix}_vertical_rate_mean"] = 0.0
        
        if fuel_col_acro in seg_df.columns:
            segment_features[f"{prefix}_fuel_flow_acropole_mean"] = seg_df[fuel_col_acro].mean()
        
        if 'mass_acropole_kg' in seg_df.columns:
            segment_features[f"{prefix}_mass_mean"] = seg_df['mass_acropole_kg'].mean()
        
        # OpenAP segment features
        if fuel_col_openap in seg_df.columns:
            segment_features[f"{prefix}_fuel_flow_openap_mean"] = seg_df[fuel_col_openap].mean()
        
        if 'thrust_openap_n' in seg_df.columns:
            segment_features[f"{prefix}_thrust_openap_mean"] = seg_df['thrust_openap_n'].mean()
        
        if 'drag_openap_n' in seg_df.columns:
            segment_features[f"{prefix}_drag_openap_mean"] = seg_df['drag_openap_n'].mean()
        
        if 'mass_openap_kg' in seg_df.columns:
            segment_features[f"{prefix}_mass_openap_mean"] = seg_df['mass_openap_kg'].mean()

        if 'acropole_multiplier' in seg_df.columns:
            segment_features[f"{prefix}_acropole_multiplier_mean"] = seg_df['acropole_multiplier'].mean()
            segment_features[f"{prefix}_acropole_multiplier_std"] = seg_df['acropole_multiplier'].std()

        if 'openap_multiplier' in seg_df.columns:
            segment_features[f"{prefix}_openap_multiplier_mean"] = seg_df['openap_multiplier'].mean()
            segment_features[f"{prefix}_openap_multiplier_std"] = seg_df['openap_multiplier'].std()
        
        # ========================================================================
        # (a) Segment-level atmosphere & wind features
        # ========================================================================
        
        # Air density from pressure and temperature
        if 'pressure_hpa' in seg_df.columns and 'temperature_pl' in seg_df.columns:
            pressures = seg_df['pressure_hpa']
            temps = seg_df['temperature_pl']
            densities = []
            for p, t in zip(pressures, temps):
                if not (np.isnan(p) or np.isnan(t)):
                    rho = calculate_air_density(p, t + 273.15)  # Convert °C to K
                    densities.append(rho)
            if densities:
                segment_features[f"{prefix}_density_mean"] = np.mean(densities)
        
        # Dynamic pressure (qbar)
        if f"{prefix}_density_mean" in segment_features and 'TAS' in seg_df.columns:
            density = segment_features[f"{prefix}_density_mean"]
            tas_mean = seg_df['TAS'].mean()
            if not np.isnan(tas_mean):
                qbar = calculate_dynamic_pressure(density, tas_mean)
                segment_features[f"{prefix}_qbar_mean"] = qbar
        
        # Wind components projected onto track
        if all(col in seg_df.columns for col in ['u_component_of_wind_pl', 'v_component_of_wind_pl', 'track']):
            u_winds = seg_df['u_component_of_wind_pl']
            v_winds = seg_df['v_component_of_wind_pl']
            tracks = seg_df['track']
            
            headwinds = []
            crosswinds = []
            for u, v, track in zip(u_winds, v_winds, tracks):
                if not any(np.isnan([u, v, track])):
                    wind_comps = calculate_wind_components(u, v, track)
                    headwinds.append(wind_comps['headwind_kts'])
                    crosswinds.append(wind_comps['crosswind_kts'])
            
            if headwinds:
                segment_features[f"{prefix}_headwind_mean"] = np.mean(headwinds)
            if crosswinds:
                segment_features[f"{prefix}_crosswind_mean"] = np.mean(crosswinds)
        
        # Temperature
        if 'temperature_pl' in seg_df.columns:
            segment_features[f"{prefix}_temperature_mean"] = seg_df['temperature_pl'].mean()
        
        # ========================================================================
        # (b) Segment-level geometry & motion features
        # ========================================================================
        
        # Track (heading) statistics
        if 'track' in seg_df.columns:
            segment_features[f"{prefix}_track_mean"] = seg_df['track'].mean()
            segment_features[f"{prefix}_track_std"] = seg_df['track'].std()
        
        # Flight path angle from vertical rate and TAS
        if 'vertical_rate' in seg_df.columns and 'TAS' in seg_df.columns:
            vr_series = seg_df['vertical_rate'].dropna()
            tas_series = seg_df['TAS'].dropna()
            if len(vr_series) > 0 and len(tas_series) > 0:
                vr_mean = vr_series.mean()
                tas_mean = tas_series.mean()
                # Clean outliers >3σ
                vr_std = vr_series.std()
                tas_std = tas_series.std()
                if not np.isnan(vr_std) and vr_std > 0:
                    vr_clean = vr_series[(vr_series - vr_mean).abs() <= 3 * vr_std]
                    if len(vr_clean) > 0:
                        vr_mean = vr_clean.mean()
                if not np.isnan(tas_std) and tas_std > 0:
                    tas_clean = tas_series[(tas_series - tas_mean).abs() <= 3 * tas_std]
                    if len(tas_clean) > 0:
                        tas_mean = tas_clean.mean()
                if tas_mean != 0:
                    ratio = vr_mean / (tas_mean + 1e-6)
                    ratio = np.clip(ratio, -1, 1)
                    fpa_rad = np.arcsin(ratio)
                    fpa_deg = np.degrees(fpa_rad)
                    segment_features[f"{prefix}_fpa_mean_deg"] = fpa_deg
        
        # TAS acceleration (optional)
        if 'TAS' in seg_df.columns and len(seg_df) > 1:
            tas_series = seg_df['TAS']
            time_diffs = seg_df['timestamp'].diff().dt.total_seconds()
            tas_diff = tas_series.diff()
            valid_mask = (~np.isnan(time_diffs)) & (~np.isnan(tas_diff)) & (time_diffs > 0)
            if valid_mask.any():
                accel_values = tas_diff[valid_mask] / time_diffs[valid_mask]
                segment_features[f"{prefix}_tas_accel_mean"] = accel_values.mean()

        # Specific excess power (rate of climb potential)
        if f"{prefix}_thrust_openap_mean" in segment_features and f"{prefix}_drag_openap_mean" in segment_features and f"{prefix}_mass_mean" in segment_features:
            thrust = segment_features[f"{prefix}_thrust_openap_mean"]
            drag = segment_features[f"{prefix}_drag_openap_mean"]
            mass = segment_features[f"{prefix}_mass_mean"]
            if mass > 0:
                segment_features[f"{prefix}_specific_excess_power"] = (thrust - drag) / mass

        # Angle of attack proxy
        if 'vertical_rate' in seg_df.columns and 'TAS' in seg_df.columns:
            vr_series = seg_df['vertical_rate'].dropna()
            tas_series = seg_df['TAS'].dropna()
            if len(vr_series) > 0 and len(tas_series) > 0:
                vr_mean = vr_series.mean()
                tas_mean = tas_series.mean()
                # Clean outliers >3σ
                vr_std = vr_series.std()
                tas_std = tas_series.std()
                if not np.isnan(vr_std) and vr_std > 0:
                    vr_clean = vr_series[(vr_series - vr_mean).abs() <= 3 * vr_std]
                    if len(vr_clean) > 0:
                        vr_mean = vr_clean.mean()
                if not np.isnan(tas_std) and tas_std > 0:
                    tas_clean = tas_series[(tas_series - tas_mean).abs() <= 3 * tas_std]
                    if len(tas_clean) > 0:
                        tas_mean = tas_clean.mean()
                if tas_mean != 0:
                    ratio = vr_mean / (tas_mean + 1e-6)
                    ratio = np.clip(ratio, -1, 1)
                    aoa_rad = np.arcsin(ratio)
                    aoa_deg = np.degrees(aoa_rad)
                    segment_features[f"{prefix}_aoa_proxy_deg"] = aoa_deg

        # Flight path curvature (rate of heading change)
        if 'track' in seg_df.columns and len(seg_df) > 1:
            track_series = seg_df['track']
            time_diffs = seg_df['timestamp'].diff().dt.total_seconds()
            track_diff = track_series.diff()
            valid_mask = (~np.isnan(time_diffs)) & (~np.isnan(track_diff)) & (time_diffs > 0)
            if valid_mask.any():
                # Approximate distance traveled in segment
                if 'groundspeed' in seg_df.columns:
                    gs_mean = seg_df['groundspeed'].mean()
                    if not np.isnan(gs_mean):
                        distance_nm = gs_mean * (time_diffs.sum() / 3600)  # nm
                        if distance_nm > 0:
                            curvature = np.abs(track_diff[valid_mask]).sum() / distance_nm
                            segment_features[f"{prefix}_curvature_deg_per_nm"] = curvature

        # ========================================================================
        # (c) Segment-level efficiency ratios
        # ========================================================================
        
        # Thrust-to-weight ratio
        if f"{prefix}_thrust_openap_mean" in segment_features and f"{prefix}_mass_mean" in segment_features:
            thrust = segment_features[f"{prefix}_thrust_openap_mean"]
            mass = segment_features[f"{prefix}_mass_mean"]
            if not (np.isnan(thrust) or np.isnan(mass) or mass == 0):
                segment_features[f"{prefix}_TW"] = thrust / (mass * G)
        
        # Drag-to-weight ratio
        if f"{prefix}_drag_openap_mean" in segment_features and f"{prefix}_mass_mean" in segment_features:
            drag = segment_features[f"{prefix}_drag_openap_mean"]
            mass = segment_features[f"{prefix}_mass_mean"]
            if not (np.isnan(drag) or np.isnan(mass) or mass == 0):
                segment_features[f"{prefix}_DW"] = drag / (mass * G)
        
        # Specific fuel consumption (SFC) for OpenAP
        if f"{prefix}_fuel_flow_openap_mean" in segment_features and f"{prefix}_thrust_openap_mean" in segment_features:
            ff = segment_features[f"{prefix}_fuel_flow_openap_mean"]
            thrust = segment_features[f"{prefix}_thrust_openap_mean"]
            if not (np.isnan(ff) or np.isnan(thrust) or thrust <= 0):
                segment_features[f"{prefix}_sfc_openap"] = ff / thrust
        
        # Specific fuel consumption (SFC) for Acropole
        if f"{prefix}_fuel_flow_acropole_mean" in segment_features and f"{prefix}_thrust_openap_mean" in segment_features:
            ff = segment_features[f"{prefix}_fuel_flow_acropole_mean"]
            thrust = segment_features[f"{prefix}_thrust_openap_mean"]
            if not (np.isnan(ff) or np.isnan(thrust) or thrust <= 0):
                segment_features[f"{prefix}_sfc_acropole"] = ff / thrust
        
        # Fuel flow differences and ratios
        if f"{prefix}_fuel_flow_acropole_mean" in segment_features and f"{prefix}_fuel_flow_openap_mean" in segment_features:
            ff_acro = segment_features[f"{prefix}_fuel_flow_acropole_mean"]
            ff_openap = segment_features[f"{prefix}_fuel_flow_openap_mean"]
            if not (np.isnan(ff_acro) or np.isnan(ff_openap)):
                segment_features[f"{prefix}_ff_diff"] = ff_acro - ff_openap
                if ff_openap > 0:
                    segment_features[f"{prefix}_ff_ratio"] = ff_acro / ff_openap
    
    # Interpolate NaN values for fpa and aoa from adjacent segments
    for seg_num in range(num_segments):
        prefix = f"seg{seg_num:02d}"
        for feature in ['fpa_mean_deg', 'aoa_proxy_deg', 'vertical_rate_mean']:
            key = f"{prefix}_{feature}"
            if pd.isna(segment_features.get(key)):
                # Find adjacent non-NaN
                left = seg_num - 1
                right = seg_num + 1
                left_val = segment_features.get(f"seg{left:02d}_{feature}") if left >= 0 else np.nan
                right_val = segment_features.get(f"seg{right:02d}_{feature}") if right < num_segments else np.nan
                if not pd.isna(left_val) and not pd.isna(right_val):
                    segment_features[key] = (left_val + right_val) / 2
                elif not pd.isna(left_val):
                    segment_features[key] = left_val
                elif not pd.isna(right_val):
                    segment_features[key] = right_val
    
    return segment_features

def process_flight(
    flight_id: str,
    dataset: str,
    ref_data: Dict[str, Any],
    skip_existing: bool = True
) -> Optional[pd.DataFrame]:
    """
    Process one flight: extract all interval features
    
    Returns:
        DataFrame with one row per interval, all features as columns
    """
    try:
        # Add memory check at start
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            if memory_usage > 2000:  # If already using > 2GB, skip
                logger.warning(f"High memory usage ({memory_usage:.1f}MB) for {flight_id}, skipping")
                return None
        except ImportError:
            pass  # psutil not available, continue
        output_dir = DATASETS[dataset]['output_dir']
        output_file = output_dir / f"{flight_id}.parquet"
        
        # Skip if already processed
        if skip_existing and output_file.exists():
            return None
        
        # Get fuel submission intervals for this flight FIRST
        fuel_df = ref_data['fuel']
        flight_intervals = fuel_df[fuel_df['flight_id'] == flight_id].copy()
        
        if flight_intervals.empty:
            logger.warning(f"{flight_id}: No fuel intervals found in submission")
            return None
        
        # Load trajectory data (may be None or minimal)
        # Check file size first to avoid memory issues
        traj_file = DATASETS[dataset]['traj_dir'] / f"{flight_id}.parquet"
        if traj_file.exists():
            file_size_mb = traj_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 50:  # Skip files > 50MB
                logger.warning(f"{flight_id}: Trajectory file too large ({file_size_mb:.1f}MB), skipping")
                return None
        
        df = load_flight_data(flight_id, dataset)
        
        # If no trajectory, create minimal dataframe for metadata extraction
        has_trajectory = bool(df is not None and not df.empty)
        if df is None:
            logger.info(f"{flight_id}: No trajectory data - extracting minimal features")
            traj_df: pd.DataFrame = pd.DataFrame()
        else:
            traj_df = df
        
        if not has_trajectory and traj_df.empty:
            logger.info(f"{flight_id}: trajectory empty - proceeding with placeholders")
        
        # Sort intervals
        flight_intervals = flight_intervals.sort_values('idx').reset_index(drop=True)
        
        # traj_df already initialized (empty if trajectory missing)
        
        # Extract flight-level features (handles empty df)
        flight_features = extract_flight_level_features(traj_df, flight_id, ref_data)
        
        # Process each interval
        interval_rows = []
        
        for idx, interval_row in flight_intervals.iterrows():
            # Get interval timestamps
            # Handle both column formats: 'start'/'end' (datetime strings or ms) or 'interval_start'/'interval_end' (ms)
            if 'start' in interval_row.index:
                start_val = interval_row['start']
                end_val = interval_row['end']
                if isinstance(start_val, (int, float)):
                    # Milliseconds since epoch
                    interval_start_ts = pd.to_datetime(start_val, unit='ms', utc=True)
                    interval_end_ts = pd.to_datetime(end_val, unit='ms', utc=True)
                else:
                    # Assume datetime string
                    interval_start_ts = pd.to_datetime(start_val)
                    interval_end_ts = pd.to_datetime(end_val)
                    if interval_start_ts.tz is None:
                        interval_start_ts = interval_start_ts.tz_localize('UTC')
                        interval_end_ts = interval_end_ts.tz_localize('UTC')
            else:
                interval_start_ms = interval_row['interval_start']
                interval_end_ms = interval_row['interval_end']
                interval_start_ts = pd.Timestamp(interval_start_ms, unit='ms', tz='UTC')
                interval_end_ts = pd.Timestamp(interval_end_ms, unit='ms', tz='UTC')
            
            # Extract interval features
            interval_features = extract_interval_features(
                traj_df, 
                interval_start_ts, 
                interval_end_ts,
                interval_row['idx'],
                flight_features
            )
            
            # Extract segment profile features
            segment_features = extract_segment_features(
                traj_df,
                interval_start_ts,
                interval_end_ts,
                flight_features,
                num_segments=NUM_SEGMENTS
            )
            
            # Combine all features
            row_features = {
                **flight_features,
                'idx': interval_row['idx'],
                'interval_start_ts': interval_start_ts,
                'interval_end_ts': interval_end_ts,
                **interval_features,
                **segment_features
            }
            
            # Add cross-segment transition features
            cross_segment_features = compute_cross_segment_features(row_features)
            row_features.update(cross_segment_features)
            
            # ========================================================================
            # F. Multiplier-specific interval features
            # ========================================================================
            
            # Multiplier statistics (already computed in interval features)
            # Add coupling with kinematics
            if 'int_acropole_multiplier_mean' in row_features and 'int_tas_mean' in row_features:
                row_features['int_acro_mult_tas_mean'] = row_features['int_acropole_multiplier_mean'] * row_features['int_tas_mean']
            
            if 'int_openap_multiplier_mean' in row_features and 'int_headwind_mean' in row_features:
                row_features['int_openap_mult_headwind_mean'] = row_features['int_openap_multiplier_mean'] * row_features['int_headwind_mean']
            
            if 'int_acropole_multiplier_mean' in row_features and 'int_vertical_rate_mean' in row_features:
                row_features['int_acro_mult_vr_mean'] = row_features['int_acropole_multiplier_mean'] * row_features['int_vertical_rate_mean']
            
            # Disagreement indicators
            if 'int_acropole_multiplier_mean' in row_features and 'int_openap_multiplier_mean' in row_features:
                acro_mult = row_features['int_acropole_multiplier_mean']
                openap_mult = row_features['int_openap_multiplier_mean']
                if not (np.isnan(acro_mult) or np.isnan(openap_mult)):
                    row_features['int_multiplier_diff'] = acro_mult - openap_mult
                    if openap_mult > 0:
                        row_features['int_multiplier_ratio'] = acro_mult / openap_mult
            
            # Fractions where multiplier exceeds thresholds
            if 'int_acropole_multiplier_mean' in row_features:
                mult = row_features['int_acropole_multiplier_mean']
                row_features['int_acro_mult_gt_1_1'] = 1 if mult > 1.1 else 0
                row_features['int_acro_mult_gt_1_2'] = 1 if mult > 1.2 else 0
                row_features['int_acro_mult_lt_0_9'] = 1 if mult < 0.9 else 0
            
            # ========================================================================
            # G. Non-linear transforms & interactions
            # ========================================================================
            
            # Selected important features for transforms
            transform_features = [
                'int_altitude_mean', 'int_tas_mean', 'int_headwind_mean',
                'int_fuel_consumed_acropole_kg', 'int_vertical_rate_mean',
                'int_mass_mean', 'int_density_mean'
            ]
            
            # Square transforms
            for feat in transform_features:
                if feat in row_features and not np.isnan(row_features[feat]):
                    row_features[f'{feat}_sq'] = row_features[feat] ** 2
            
            # Log transforms (for positive features)
            for feat in ['int_altitude_mean', 'int_tas_mean', 'int_fuel_consumed_acropole_kg', 'int_mass_mean', 'int_density_mean']:
                if feat in row_features and row_features[feat] > 0:
                    row_features[f'log_{feat}'] = np.log(row_features[feat])
            
            # Key pairwise interactions
            if 'int_altitude_mean' in row_features and 'int_tas_mean' in row_features:
                alt = row_features['int_altitude_mean']
                tas = row_features['int_tas_mean']
                if not (np.isnan(alt) or np.isnan(tas)):
                    row_features['int_alt_tas_interaction'] = alt * tas
            
            if 'int_headwind_mean' in row_features and 'int_fuel_consumed_acropole_kg' in row_features:
                hw = row_features['int_headwind_mean']
                fuel = row_features['int_fuel_consumed_acropole_kg']
                if not (np.isnan(hw) or np.isnan(fuel) or fuel == 0):
                    row_features['int_headwind_fuel_interaction'] = hw * fuel
            
            if 'int_vertical_rate_mean' in row_features and 'int_mass_mean' in row_features:
                vr = row_features['int_vertical_rate_mean']
                mass = row_features['int_mass_mean']
                if not (np.isnan(vr) or np.isnan(mass) or mass == 0):
                    row_features['int_vr_mass_interaction'] = vr * mass
            
            # Add actual fuel if available (train dataset)
            if 'fuel_kg' in interval_row:
                row_features['fuel_kg_actual'] = interval_row['fuel_kg']
            
            interval_rows.append(row_features)
        
        # Create DataFrame
        result_df = pd.DataFrame(interval_rows)
        
        # ========================================================================
        # Compute pseudo-prev/next intervals for boundary intervals
        # ========================================================================
        
        pseudo_prev_features = {}
        pseudo_next_features = {}
        
        if not result_df.empty and len(result_df) > 1 and traj_df is not None and not traj_df.empty:
            # Sort result_df by interval_idx to ensure order
            result_df = result_df.sort_values('interval_idx').reset_index(drop=True)
            
            # Key features for lag/lead analysis
            lag_features = [
                'int_tas_mean', 'int_altitude_mean', 'int_fuel_consumed_acropole_kg',
                'int_headwind_mean', 'int_acropole_multiplier_mean', 'int_mass_mean',
                'int_vertical_rate_mean', 'int_ground_distance_nm'
            ]
            
            # Pseudo-previous for first interval
            first_interval = result_df.iloc[0]
            interval_duration = first_interval['interval_duration_sec']
            pseudo_prev_start = first_interval['interval_start_ts'] - pd.Timedelta(seconds=interval_duration)
            pseudo_prev_end = first_interval['interval_start_ts']
            
            # Extract features for pseudo-previous interval (use dummy idx -1)
            pseudo_prev_features = extract_interval_features(
                traj_df, pseudo_prev_start, pseudo_prev_end, -1, flight_features
            )
            
            # Pseudo-next for last interval
            last_interval = result_df.iloc[-1]
            interval_duration = last_interval['interval_duration_sec']
            pseudo_next_start = last_interval['interval_end_ts']
            pseudo_next_end = last_interval['interval_end_ts'] + pd.Timedelta(seconds=interval_duration)
            
            # Extract features for pseudo-next interval (use dummy idx len(result_df))
            pseudo_next_features = extract_interval_features(
                traj_df, pseudo_next_start, pseudo_next_end, len(result_df), flight_features
            )
        
        # ========================================================================
        # Add lag/lead features across intervals
        # ========================================================================
        
        if not result_df.empty and len(result_df) > 1:
            # Sort by interval index to ensure proper ordering
            result_df = result_df.sort_values('interval_idx').reset_index(drop=True)
            
            # Key features for lag/lead analysis
            lag_features = [
                'int_tas_mean', 'int_altitude_mean', 'int_fuel_consumed_acropole_kg',
                'int_headwind_mean', 'int_acropole_multiplier_mean', 'int_mass_mean',
                'int_vertical_rate_mean', 'int_ground_distance_nm'
            ]
            
            # Add lag features (previous interval)
            for feat in lag_features:
                if feat in result_df.columns:
                    result_df[f'{feat}_prev'] = result_df[feat].shift(1)
                    # Fill NaN for first row with pseudo-prev
                    if pseudo_prev_features.get(feat) is not None and not pd.isna(pseudo_prev_features[feat]):
                        result_df[f'{feat}_prev'].fillna(pseudo_prev_features[feat], inplace=True)
                    # Difference and ratio with previous
                    result_df[f'{feat}_prev_diff'] = result_df[feat] - result_df[f'{feat}_prev']
                    valid_mask = (result_df[f'{feat}_prev'].notna()) & (result_df[feat].notna())
                    # Handle ratio with epsilon to avoid inf/-inf
                    denom = result_df.loc[valid_mask, f'{feat}_prev'] + 1e-6
                    result_df.loc[valid_mask, f'{feat}_prev_ratio'] = result_df.loc[valid_mask, feat] / denom
                    # Handle 0/0 case: set ratio to 1 when both are 0
                    zero_mask = (result_df[f'{feat}_prev'] == 0) & (result_df[feat] == 0)
                    result_df.loc[zero_mask, f'{feat}_prev_ratio'] = 1.0
            
            # Add lead features (next interval)
            for feat in lag_features:
                if feat in result_df.columns:
                    result_df[f'{feat}_next'] = result_df[feat].shift(-1)
                    # Fill NaN for last row with pseudo-next
                    if pseudo_next_features.get(feat) is not None and not pd.isna(pseudo_next_features[feat]):
                        result_df[f'{feat}_next'].fillna(pseudo_next_features[feat], inplace=True)
                    # Difference and ratio with next
                    result_df[f'{feat}_next_diff'] = result_df[f'{feat}_next'] - result_df[feat]
                    valid_mask = (result_df[f'{feat}_next'].notna()) & (result_df[feat].notna())
                    # Handle ratio with epsilon to avoid inf/-inf
                    denom = result_df.loc[valid_mask, feat] + 1e-6
                    result_df.loc[valid_mask, f'{feat}_next_ratio'] = result_df.loc[valid_mask, f'{feat}_next'] / denom
                    # Handle 0/0 case: set ratio to 1 when both are 0
                    zero_mask = (result_df[f'{feat}_next'] == 0) & (result_df[feat] == 0)
                    result_df.loc[zero_mask, f'{feat}_next_ratio'] = 1.0
            
            # Rolling window features (3-interval window)
            for feat in lag_features:
                if feat in result_df.columns:
                    # Rolling mean and std
                    result_df[f'{feat}_rolling_mean_3'] = result_df[feat].rolling(window=3, center=True, min_periods=1).mean()
                    result_df[f'{feat}_rolling_std_3'] = result_df[feat].rolling(window=3, center=True, min_periods=1).std()
                    
                    # Deviation from rolling mean
                    valid_mask = result_df[f'{feat}_rolling_mean_3'].notna()
                    result_df.loc[valid_mask, f'{feat}_rolling_dev_3'] = result_df.loc[valid_mask, feat] - result_df.loc[valid_mask, f'{feat}_rolling_mean_3']
        
        # Add cumulative fuel burn and total track change
        if not result_df.empty:
            # Cumulative fuel burn across intervals
            if 'int_fuel_consumed_acropole_kg' in result_df.columns:
                result_df = result_df.sort_values('interval_idx').reset_index(drop=True)
                result_df['int_cum_fuel_acropole_kg'] = result_df['int_fuel_consumed_acropole_kg'].cumsum()
            
            # Total track change for flight (sum across intervals)
            if 'int_total_track_change_deg' in result_df.columns:
                total_track_change = result_df['int_total_track_change_deg'].sum()
                result_df['flt_total_track_change_deg'] = total_track_change
        
        # Save per-flight features
        output_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_file, index=False)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing {flight_id}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


# ============================================================================
# Main Processing Function
# ============================================================================

def process_dataset(
    dataset: str, 
    max_flights: Optional[int] = None, 
    num_workers: int = 14,
    skip_existing: bool = True,
    flight_id_filter: Optional[str] = None
):
    """Process all flights in a dataset with multiprocessing"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing dataset: {dataset.upper()}")
    logger.info(f"{'='*80}\n")
    
    # Load reference data
    ref_data = load_reference_data(dataset)
    
    # Get unique flight IDs from fuel submission file
    flight_ids = ref_data['fuel']['flight_id'].unique().tolist()
    
    logger.info(f"Found {len(flight_ids)} flights in fuel submission file")
    
    # Filter to specific flight if requested
    if flight_id_filter:
        if flight_id_filter in flight_ids:
            flight_ids = [flight_id_filter]
            logger.info(f"Filtering to single flight: {flight_id_filter}")
        else:
            logger.error(f"Flight {flight_id_filter} not found in {dataset} dataset")
            return
    
    # For final dataset, exclude flights already in rank
    if dataset == 'final':
        rank_fuel = pd.read_parquet(DATASETS['rank']['fuel_submission'])
        rank_flight_ids = set(rank_fuel['flight_id'].unique())
        original_count = len(flight_ids)
        flight_ids = [fid for fid in flight_ids if fid not in rank_flight_ids]
        logger.info(f"  Excluded {original_count - len(flight_ids)} flights already in rank dataset")
        logger.info(f"  Final-only flights to process: {len(flight_ids)}")
    
    # Check which ones have trajectory files (for logging only)
    traj_dir = DATASETS[dataset]['traj_dir']
    existing_trajs = set([f.stem for f in traj_dir.glob('*.parquet')])
    
    # For final dataset, check for duplicates and also count rank trajectories
    if dataset == 'final':
        rank_traj_dir = DATASETS['rank']['traj_dir']
        rank_trajs = set([f.stem for f in rank_traj_dir.glob('*.parquet')])
        
        # Check for duplicate trajectory files
        duplicates = existing_trajs.intersection(rank_trajs)
        if duplicates:
            logger.warning(f"  Found {len(duplicates)} duplicate trajectory files in both rank and final folders")
            logger.warning(f"  These should be processed from rank folder only")
            # Remove duplicates from final existing_trajs set
            existing_trajs = existing_trajs - duplicates
        
        # Union for checking which flights have trajectory data
        existing_trajs = existing_trajs.union(rank_trajs)
    
    flights_with_traj = [fid for fid in flight_ids if fid in existing_trajs]
    flights_without_traj = [fid for fid in flight_ids if fid not in existing_trajs]
    
    logger.info(f"  Flights with trajectory data: {len(flights_with_traj)}")
    logger.info(f"  Flights without trajectory data: {len(flights_without_traj)}")
    logger.info(f"  Will process ALL flights (extract minimal features for flights without trajectories)")
    
    if max_flights:
        flight_ids = flight_ids[:max_flights]
        logger.info(f"Processing limited to {max_flights} flights")
    
    # Check for existing files if skip_existing is enabled
    output_dir = DATASETS[dataset]['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if skip_existing:
        existing_files = set([f.stem for f in output_dir.glob('*.parquet')])
        flight_ids = [fid for fid in flight_ids if fid not in existing_files]
        logger.info(f"Skipping {len(existing_files)} already processed flights")
        logger.info(f"Remaining flights to process: {len(flight_ids)}")
    
    if len(flight_ids) == 0:
        logger.info("No flights to process!")
        return
    
    # Process flights
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for flight_id in flight_ids:
            future = executor.submit(process_flight, flight_id, dataset, ref_data, skip_existing)
            futures[future] = flight_id
        
        with tqdm(total=len(futures), desc=f"Processing {dataset}", 
                  unit="flight", miniters=1, smoothing=0) as pbar:
            for future in as_completed(futures, timeout=1800):  # 30 minute timeout per task
                flight_id = futures[future]
                try:
                    result = future.result(timeout=60)  # 60 second result timeout
                    if result is not None:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    logger.error(f"Failed to process {flight_id}: {e}")
                    fail_count += 1
                
                pbar.update(1)
                pbar.refresh()  # Force refresh
    
    # Summary
    total_files = len(list(output_dir.glob('*.parquet')))
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset {dataset} complete:")
    logger.info(f"  Newly processed: {success_count} flights")
    logger.info(f"  Failed: {fail_count} flights")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"{'='*80}\n")


# ============================================================================
# Consolidate Features
# ============================================================================

def consolidate_dataset_features(dataset: str):
    """Consolidate all per-flight feature files into a single parquet file"""
    output_dir = DATASETS[dataset]['output_dir']

    if not output_dir.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return

    logger.info(f"Consolidating {dataset} features...")

    # Get all flight feature files
    flight_files = list(output_dir.glob('*.parquet'))

    if not flight_files:
        logger.warning(f"No feature files found in {output_dir}")
        return

    logger.info(f"Found {len(flight_files)} flight feature files")

    # Consolidated output file
    consolidated_file = PROCESSED_ROOT / "Stage_8_Consolidated_Features" / f"consolidated_features_{dataset}_multiplier.parquet"

    # Create consolidated directory if needed
    consolidated_file.parent.mkdir(parents=True, exist_ok=True)

    # Read and combine all files
    all_features = []
    total_rows = 0

    for flight_file in tqdm(flight_files, desc=f"Reading {dataset} files"):
        try:
            df = pd.read_parquet(flight_file)
            all_features.append(df)
            total_rows += len(df)
        except Exception as e:
            logger.error(f"Error reading {flight_file}: {e}")

    if not all_features:
        logger.error(f"No valid feature files found for {dataset}")
        return

    # Combine all dataframes
    logger.info(f"Combining {len(all_features)} dataframes with {total_rows} total rows...")
    consolidated_df = pd.concat(all_features, ignore_index=True)

    # Sort by flight_id and interval_idx for consistency
    if 'flight_id' in consolidated_df.columns and 'idx' in consolidated_df.columns:
        consolidated_df = consolidated_df.sort_values(['flight_id', 'idx']).reset_index(drop=True)

    # Drop constant features
    constant_cols = []
    for col in consolidated_df.columns:
        if consolidated_df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)

    if constant_cols:
        logger.info(f"Dropping {len(constant_cols)} constant features: {constant_cols}")
        consolidated_df = consolidated_df.drop(columns=constant_cols)

    # Save consolidated file
    logger.info(f"Saving consolidated file: {consolidated_file}")
    consolidated_df.to_parquet(consolidated_file, index=False)

    logger.info(f" Consolidated {dataset}: {len(consolidated_df)} rows, {len(consolidated_df.columns)} columns")
    logger.info(f"  Saved to: {consolidated_file}")

    # Clean up individual files if desired (optional)
    # logger.info("Cleaning up individual flight files...")
    # for flight_file in flight_files:
    #     flight_file.unlink()

    return consolidated_df


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract features from Acropole trajectories (Stage 7)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['train', 'rank', 'final', 'all'],
        default='all',
        help='Which dataset to process'
    )
    parser.add_argument(
        '--max-flights',
        type=int,
        default=None,
        help='Maximum number of flights to process (for testing)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--reprocess',
        action='store_true',
        help='Force reprocessing of already-processed flights'
    )
    parser.add_argument(
        '--flight_id',
        type=str,
        default=None,
        help='Process only this flight ID'
    )
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*80}")
    logger.info("STAGE 7: FEATURE ENGINEERING FROM ACROPOLE TRAJECTORIES")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"{'='*80}\n")
    
    skip_existing = not args.reprocess
    
    if args.dataset == 'all':
        for dataset in ['train', 'rank', 'final']:
            process_dataset(dataset, args.max_flights, args.workers, skip_existing, args.flight_id)
            consolidate_dataset_features(dataset)
    else:
        process_dataset(args.dataset, args.max_flights, args.workers, skip_existing, args.flight_id)
        consolidate_dataset_features(args.dataset)
    
    logger.info(f"\n{'='*80}")
    logger.info("ALL PROCESSING COMPLETE")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
