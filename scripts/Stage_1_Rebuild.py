"""
Unified Trajectory Processing Pipeline

Combines cleaning, resampling, gap reconstruction, and smoothing in a single pass:
1. Load raw trajectory
2. Initial filtering (remove spikes, outliers)
3. Resample to 1-second intervals with limited interpolation
4. Detect and reconstruct long gaps using OpenAP WRAP
5. Apply boundary smoothing at gap edges
6. Final Kalman smoothing for overall consistency
7. Validate and save

Usage:
    python scripts/trajectory_unified_pipeline.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================================
# Path Setup - Must be at module level for multiprocessing
# ============================================================================

REPO_ROOT = Path(r"F:\PRC_2025\Likable-ant_v1")
TRAFFIC_SRC = REPO_ROOT / "traffic" / "src"
OPENAP_ROOT = REPO_ROOT / "openap"

if str(TRAFFIC_SRC) not in sys.path:
    sys.path.insert(0, str(TRAFFIC_SRC))
if str(OPENAP_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENAP_ROOT))

# ============================================================================
# Imports
# ============================================================================

from traffic.core import Flight
from traffic.algorithms.filters import FilterMedian, FilterMean, FilterAboveSigmaMedian
from traffic.algorithms.filters.aggressive import FilterDerivative
from traffic.algorithms.filters.kalman import KalmanSmoother6D
from cartes.crs import EuroPP
from openap.kinematic import WRAP
from pyproj import Geod
from scipy.ndimage import gaussian_filter1d

# ============================================================================
# Configuration
# ============================================================================

RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

PARTITIONS = {
    #"train": RAW_ROOT / "flights_train",
    "rank": RAW_ROOT / "flights_rank",
    "final": RAW_ROOT / "flights_final",
}

OUT_ROOT = PROCESSED_ROOT / "trajectories_unified"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Processing configuration
RESAMPLE_FREQ = "1s"
MAX_INTERPOLATE_SEC = 30  # Short gaps filled by interpolation
LONG_GAP_SEC = 30.0       # Gaps > 30s need reconstruction
MAX_RECON_GAP_SEC = 28800.0  # Max gap to reconstruct (8 hours)
BOUNDARY_SMOOTH_WINDOW = 10  # Points to smooth at gap boundaries

# Multiprocessing configuration
MAX_WORKERS = 14  # Increase for faster processing (adjust based on CPU cores)

# Physical limits for validation and clamping
LIMITS = {
    'groundspeed': (50, 600),   # knots
    'mach': (0.15, 0.90),
    'TAS': (50, 600),           # knots
    'CAS': (50, 400),           # knots
    'vertical_rate': (-6000, 6000),  # ft/min
    'altitude': (0, 50000),     # feet
}

# Output columns
OUTPUT_COLUMNS = [
    "timestamp",
    "flight_id",
    "latitude",
    "longitude",
    "altitude",
    "groundspeed",
    "track",
    "vertical_rate",
    "mach",
    "TAS",
    "CAS",
    "is_reconstructed",
]

# Aircraft type mapping
TYPECODE_TO_OPENAP = {
    "A306": "a306", "A319": "a319", "A320": "a320", "A321": "a321",
    "A332": "a332", "A333": "a333", "A343": "a343", "A359": "a359",
    "A388": "a388",
    "B737": "b737", "B738": "b738", "B739": "b739", "B744": "b744",
    "B752": "b752", "B763": "b763", "B77L": "b77l", "B77W": "b77w",
    "B788": "b788", "B789": "b789",
    "E190": "e190",
    "A124": "b744", "A19N": "a320", "A20N": "a320", "A21N": "a321",
    "A310": "a319", "A318": "a319",
    "AT72": "e190", "AT75": "e190", "AT76": "e190",
    "B37M": "b737", "B38M": "b738", "B39M": "b739", "B3XM": "b737",
    "B733": "b737", "B734": "b737", "B735": "b737",
    "B748": "b744", "B762": "b752", "B772": "b77w", "B773": "b77w",
    "C25A": "e190", "C525": "e190", "C550": "e190", "C56X": "e190",
    "CRJ2": "e190", "CRJ9": "e190",
    "E145": "e190", "E170": "e190", "E75L": "e190", "E195": "e190",
    "E290": "e190",
    "GLF5": "e190", "GL5T": "e190", "GLF6": "e190",
    "LJ45": "e190", "PC24": "e190", "SU95": "e190",
}

# Constants
FT_TO_M = 0.3048
M_TO_FT = 3.28084
MS_TO_KT = 1.94384
KT_TO_MS = 0.514444
CRUISE_ALT_FT = 26000.0

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Geodesic calculator
geod = Geod(ellps='WGS84')

# ============================================================================
# ISA Atmosphere Functions
# ============================================================================

def isa_temperature(altitude_m):
    """ISA temperature at altitude (K)."""
    T0 = 288.15
    lapse_rate = -0.0065
    if altitude_m <= 11000:
        return T0 + lapse_rate * altitude_m
    else:
        return 216.65

def isa_pressure(altitude_m):
    """ISA pressure at altitude (Pa)."""
    p0 = 101325.0
    T0 = 288.15
    lapse_rate = -0.0065
    g = 9.80665
    R = 287.05
    
    # Clamp altitude to prevent numerical issues
    altitude_m = np.clip(altitude_m, 0, 25000)
    
    if altitude_m <= 11000:
        return p0 * (1 + lapse_rate * altitude_m / T0) ** (-g / (lapse_rate * R))
    else:
        p11 = isa_pressure(11000)
        T11 = 216.65
        return p11 * np.exp(-g * (altitude_m - 11000) / (R * T11))

def mach_to_tas(mach, altitude_m):
    """Convert Mach to TAS (m/s)."""
    gamma = 1.4
    R = 287.05
    T = isa_temperature(altitude_m)
    a = np.sqrt(gamma * R * T)
    return mach * a

def tas_to_mach(tas_ms, altitude_m):
    """Convert TAS to Mach."""
    gamma = 1.4
    R = 287.05
    T = isa_temperature(altitude_m)
    a = np.sqrt(gamma * R * T)
    if a > 0:
        return tas_ms / a
    return 0.0

def tas_to_cas(tas_ms, altitude_m):
    """Convert TAS to CAS (m/s)."""
    p = isa_pressure(altitude_m)
    p0 = 101325.0
    sigma = max(p / p0, 0.1)  # Prevent division by zero
    cas_ms = tas_ms * np.sqrt(sigma)
    return cas_ms

# ============================================================================
# Physical Value Validation and Clamping
# ============================================================================

def clamp_value(value, limits):
    """Clamp value to physical limits."""
    return np.clip(value, limits[0], limits[1])

def validate_and_clamp_kinematics(df):
    """
    Validate and clamp kinematic values to physical limits.
    
    Args:
        df: DataFrame with kinematic columns
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Clamp each parameter to physical limits
    if 'groundspeed' in df.columns:
        df['groundspeed'] = df['groundspeed'].clip(*LIMITS['groundspeed'])
    
    if 'mach' in df.columns:
        df['mach'] = df['mach'].clip(*LIMITS['mach'])
    
    if 'TAS' in df.columns:
        df['TAS'] = df['TAS'].clip(*LIMITS['TAS'])
    
    if 'CAS' in df.columns:
        df['CAS'] = df['CAS'].clip(*LIMITS['CAS'])
    
    if 'vertical_rate' in df.columns:
        df['vertical_rate'] = df['vertical_rate'].clip(*LIMITS['vertical_rate'])
    
    if 'altitude' in df.columns:
        df['altitude'] = df['altitude'].clip(*LIMITS['altitude'])
    
    return df

# ============================================================================
# Boundary Smoothing
# ============================================================================

def smooth_gap_boundaries(df, gap_indices, window=BOUNDARY_SMOOTH_WINDOW):
    """
    Apply Gaussian smoothing at gap boundaries to prevent discontinuities.
    
    Args:
        df: DataFrame with trajectory
        gap_indices: List of gap start indices (from original gap detection)
        window: Number of points to smooth on each side
        
    Returns:
        Smoothed DataFrame
    """
    if not gap_indices:
        return df
    
    df = df.copy()
    sigma = window / 3.0  # Gaussian sigma
    
    smooth_cols = ['groundspeed', 'vertical_rate', 'track']
    
    # Map old gap indices to new combined DataFrame indices
    # Find corresponding timestamps in combined DataFrame
    for gap_idx in gap_indices:
        # Skip invalid indices
        if gap_idx == 0 or gap_idx >= len(df):
            continue
        
        try:
            # Define boundary region around the gap index
            start_idx = max(0, gap_idx - window)
            end_idx = min(len(df), gap_idx + window)
            
            if end_idx - start_idx < 5:  # Too small to smooth
                continue
            
            # Apply Gaussian smoothing to boundary region
            for col in smooth_cols:
                if col in df.columns:
                    segment = df.loc[start_idx:end_idx-1, col].values
                    if len(segment) > 5 and not np.all(np.isnan(segment)):
                        # Handle NaN values
                        mask = ~np.isnan(segment)
                        if mask.sum() > 3:
                            try:
                                smoothed = gaussian_filter1d(segment[mask], sigma=sigma/2)
                                # Create a copy of the slice to avoid SettingWithCopyWarning
                                df_slice = df.loc[start_idx:end_idx-1, col].copy()
                                df_slice.iloc[mask] = smoothed
                                df.loc[start_idx:end_idx-1, col] = df_slice.values
                            except Exception as smooth_err:
                                # Skip smoothing if it fails
                                continue
        except Exception as e:
            # Skip this boundary if any error occurs
            continue
    
    return df

# ============================================================================
# Phase Classification
# ============================================================================

def classify_phase(alt1_ft, alt2_ft, vs_ftmin, flight_fraction):
    """Classify flight phase for gap segment."""
    alt_diff = alt2_ft - alt1_ft
    avg_alt = (alt1_ft + alt2_ft) / 2.0
    
    if avg_alt >= CRUISE_ALT_FT and abs(alt_diff) < 1000:
        return 'cruise'
    
    if alt_diff < -300:
        return 'descent'
    
    if alt_diff > 300:
        return 'climb'
    
    if abs(alt_diff) < 300:
        if flight_fraction < 0.3:
            return 'climb'
        elif flight_fraction > 0.7:
            return 'descent'
        else:
            return 'cruise'
    
    return 'cruise'

# ============================================================================
# Gap Synthesis with Boundary Matching
# ============================================================================

def interpolate_latlon(lat1, lon1, lat2, lon2, fractions):
    """Interpolate positions along great circle.
    
    Returns empty arrays if start and end positions are identical (frozen position).
    """
    # CRITICAL FIX: Detect frozen positions (identical start/end coordinates)
    lat_diff = abs(lat2 - lat1)
    lon_diff = abs(lon2 - lon1)
    
    if lat_diff < 0.0001 and lon_diff < 0.0001:  # Less than ~11m movement
        # Positions are identical - return empty to skip interpolation
        return np.array([]), np.array([])
    
    lats = []
    lons = []
    
    for frac in fractions:
        g = geod.inv(lon1, lat1, lon2, lat2)
        distance = g[2]
        azimuth = g[0]
        
        interp = geod.fwd(lon1, lat1, azimuth, distance * frac)
        lons.append(interp[0])
        lats.append(interp[1])
    
    return np.array(lats), np.array(lons)

def blend_boundary_values(val_before, val_synth, val_after, n_points, blend_width=5):
    """
    Blend synthetic values with boundaries using sigmoid transitions.
    
    Args:
        val_before: Value before gap
        val_synth: Array of synthetic values
        val_after: Value after gap
        n_points: Number of synthetic points
        blend_width: Number of points for blending transition
        
    Returns:
        Blended array
    """
    if n_points < 2 * blend_width:
        return val_synth
    
    result = val_synth.copy()
    
    # Blend start
    for i in range(min(blend_width, n_points)):
        weight = i / blend_width
        result[i] = val_before * (1 - weight) + val_synth[i] * weight
    
    # Blend end
    for i in range(min(blend_width, n_points)):
        idx = n_points - blend_width + i
        if idx >= 0:
            weight = i / blend_width
            result[idx] = val_synth[idx] * (1 - weight) + val_after * weight
    
    return result

def synthesize_cruise_gap(row1, row2, times, wrap):
    """Synthesize cruise segment with boundary blending."""
    n_points = len(times)
    if n_points == 0:
        return []
    
    # Get cruise parameters from WRAP
    try:
        mach_data = wrap.cruise_mach()
        mach = mach_data['default']
    except:
        mach = (row1.get('mach', 0.78) + row2.get('mach', 0.78)) / 2
        mach = clamp_value(mach, LIMITS['mach'])
    
    # Altitude progression (slight variation for realism)
    alt_ft = np.linspace(row1['altitude'], row2['altitude'], n_points)
    
    # Convert to TAS with blending
    tas_vals = []
    for i, alt in enumerate(alt_ft):
        alt_m = alt * FT_TO_M
        tas_ms = mach_to_tas(mach, alt_m)
        tas_kt = tas_ms * MS_TO_KT
        tas_vals.append(tas_kt)
    
    tas_vals = np.array(tas_vals)
    
    # Blend with boundary values
    tas_before = row1.get('TAS', tas_vals[0])
    tas_after = row2.get('TAS', tas_vals[-1])
    tas_vals = blend_boundary_values(tas_before, tas_vals, tas_after, n_points)
    
    # Clamp to limits
    tas_vals = np.clip(tas_vals, *LIMITS['TAS'])
    
    # Groundspeed (approximate as TAS)
    gs_vals = tas_vals.copy()
    gs_before = row1.get('groundspeed', gs_vals[0])
    gs_after = row2.get('groundspeed', gs_vals[-1])
    gs_vals = blend_boundary_values(gs_before, gs_vals, gs_after, n_points)
    gs_vals = np.clip(gs_vals, *LIMITS['groundspeed'])
    
    # Interpolate lat/lon
    fractions = np.linspace(0, 1, n_points + 2)[1:-1]
    lats, lons = interpolate_latlon(
        row1['latitude'], row1['longitude'],
        row2['latitude'], row2['longitude'],
        fractions
    )
    
    # Skip if frozen position detected
    if len(lats) == 0:
        return []
    
    # Track
    track1 = row1['track']
    track2 = row2['track']
    if abs(track2 - track1) > 180:
        if track2 > track1:
            track1 += 360
        else:
            track2 += 360
    tracks = np.interp(fractions, [0, 1], [track1, track2]) % 360
    
    # Build rows
    rows = []
    for i, t in enumerate(times):
        alt_m = alt_ft[i] * FT_TO_M
        mach_val = tas_to_mach(tas_vals[i] * KT_TO_MS, alt_m)
        mach_val = clamp_value(mach_val, LIMITS['mach'])
        
        cas_ms = tas_to_cas(tas_vals[i] * KT_TO_MS, alt_m)
        cas_kt = cas_ms * MS_TO_KT
        cas_kt = clamp_value(cas_kt, LIMITS['CAS'])
        
        rows.append({
            'timestamp': t,
            'flight_id': row1['flight_id'],
            'latitude': lats[i],
            'longitude': lons[i],
            'altitude': alt_ft[i],
            'groundspeed': gs_vals[i],
            'track': tracks[i],
            'vertical_rate': 0.0,
            'mach': mach_val,
            'TAS': tas_vals[i],
            'CAS': cas_kt,
            'is_reconstructed': True,
        })
    
    return rows

def synthesize_climb_gap(row1, row2, times, wrap):
    """Synthesize climb segment with boundary blending."""
    n_points = len(times)
    if n_points == 0:
        return []
    
    # Get climb parameters
    try:
        vs_data = wrap.climb_vs_conmach()
        vs_ms = vs_data['default']
    except:
        gap_sec = (row2['timestamp'] - row1['timestamp']).total_seconds()
        alt_diff_m = (row2['altitude'] - row1['altitude']) * FT_TO_M
        vs_ms = alt_diff_m / gap_sec
        vs_ms = max(2.0, min(vs_ms, 12.0))
    
    try:
        cas_data = wrap.climb_const_vcas()
        cas_kt = cas_data['default']
        cas_ms = cas_kt * KT_TO_MS
    except:
        cas_ms = 150 * KT_TO_MS
    
    # Altitude progression
    fractions = np.linspace(0, 1, n_points)
    target_alt_diff_ft = row2['altitude'] - row1['altitude']
    alt_ft = row1['altitude'] + (target_alt_diff_ft * fractions)
    
    # Calculate speeds with altitude dependence
    tas_vals = []
    mach_vals = []
    cas_vals = []
    
    for alt in alt_ft:
        alt_m = alt * FT_TO_M
        tas_ms = cas_ms / np.sqrt(max(isa_pressure(alt_m) / 101325.0, 0.1))
        tas_kt = tas_ms * MS_TO_KT
        tas_kt = clamp_value(tas_kt, LIMITS['TAS'])
        
        mach = tas_to_mach(tas_ms, alt_m)
        mach = clamp_value(mach, LIMITS['mach'])
        
        cas_out = tas_to_cas(tas_ms, alt_m) * MS_TO_KT
        cas_out = clamp_value(cas_out, LIMITS['CAS'])
        
        tas_vals.append(tas_kt)
        mach_vals.append(mach)
        cas_vals.append(cas_out)
    
    tas_vals = np.array(tas_vals)
    
    # Blend boundaries
    tas_before = row1.get('TAS', tas_vals[0])
    tas_after = row2.get('TAS', tas_vals[-1])
    tas_vals = blend_boundary_values(tas_before, tas_vals, tas_after, n_points)
    
    # Groundspeed
    gs_vals = tas_vals.copy()
    gs_before = row1.get('groundspeed', gs_vals[0])
    gs_after = row2.get('groundspeed', gs_vals[-1])
    gs_vals = blend_boundary_values(gs_before, gs_vals, gs_after, n_points)
    gs_vals = np.clip(gs_vals, *LIMITS['groundspeed'])
    
    # Position interpolation
    lats, lons = interpolate_latlon(
        row1['latitude'], row1['longitude'],
        row2['latitude'], row2['longitude'],
        fractions
    )
    
    # Skip if frozen position detected
    if len(lats) == 0:
        return []
    
    # Track
    track1 = row1['track']
    track2 = row2['track']
    if abs(track2 - track1) > 180:
        if track2 > track1:
            track1 += 360
        else:
            track2 += 360
    tracks = np.interp(fractions, [0, 1], [track1, track2]) % 360
    
    # Vertical rate
    vr_ftmin = vs_ms * M_TO_FT * 60
    vr_ftmin = clamp_value(vr_ftmin, LIMITS['vertical_rate'])
    
    # Build rows
    rows = []
    for i, t in enumerate(times):
        rows.append({
            'timestamp': t,
            'flight_id': row1['flight_id'],
            'latitude': lats[i],
            'longitude': lons[i],
            'altitude': alt_ft[i],
            'groundspeed': gs_vals[i],
            'track': tracks[i],
            'vertical_rate': vr_ftmin,
            'mach': mach_vals[i],
            'TAS': tas_vals[i],
            'CAS': cas_vals[i],
            'is_reconstructed': True,
        })
    
    return rows

def synthesize_descent_gap(row1, row2, times, wrap):
    """Synthesize descent segment with boundary blending."""
    n_points = len(times)
    if n_points == 0:
        return []
    
    # Get descent parameters
    try:
        vs_data = wrap.descent_vs_conmach()
        vs_ms = -abs(vs_data['default'])
    except:
        gap_sec = (row2['timestamp'] - row1['timestamp']).total_seconds()
        alt_diff_m = (row2['altitude'] - row1['altitude']) * FT_TO_M
        vs_ms = alt_diff_m / gap_sec
        vs_ms = min(-2.0, max(vs_ms, -12.0))
    
    try:
        cas_data = wrap.descent_const_vcas()
        cas_kt = cas_data['default']
        cas_ms = cas_kt * KT_TO_MS
    except:
        cas_ms = 140 * KT_TO_MS
    
    # Altitude progression
    fractions = np.linspace(0, 1, n_points)
    target_alt_diff_ft = row2['altitude'] - row1['altitude']
    alt_ft = row1['altitude'] + (target_alt_diff_ft * fractions)
    
    # Calculate speeds
    tas_vals = []
    mach_vals = []
    cas_vals = []
    
    for alt in alt_ft:
        alt_m = alt * FT_TO_M
        tas_ms = cas_ms / np.sqrt(max(isa_pressure(alt_m) / 101325.0, 0.1))
        tas_kt = tas_ms * MS_TO_KT
        tas_kt = clamp_value(tas_kt, LIMITS['TAS'])
        
        mach = tas_to_mach(tas_ms, alt_m)
        mach = clamp_value(mach, LIMITS['mach'])
        
        cas_out = tas_to_cas(tas_ms, alt_m) * MS_TO_KT
        cas_out = clamp_value(cas_out, LIMITS['CAS'])
        
        tas_vals.append(tas_kt)
        mach_vals.append(mach)
        cas_vals.append(cas_out)
    
    tas_vals = np.array(tas_vals)
    
    # Blend boundaries
    tas_before = row1.get('TAS', tas_vals[0])
    tas_after = row2.get('TAS', tas_vals[-1])
    tas_vals = blend_boundary_values(tas_before, tas_vals, tas_after, n_points)
    
    # Groundspeed
    gs_vals = tas_vals.copy()
    gs_before = row1.get('groundspeed', gs_vals[0])
    gs_after = row2.get('groundspeed', gs_vals[-1])
    gs_vals = blend_boundary_values(gs_before, gs_vals, gs_after, n_points)
    gs_vals = np.clip(gs_vals, *LIMITS['groundspeed'])
    
    # Position
    lats, lons = interpolate_latlon(
        row1['latitude'], row1['longitude'],
        row2['latitude'], row2['longitude'],
        fractions
    )
    
    # Skip if frozen position detected
    if len(lats) == 0:
        return []
    
    # Track
    track1 = row1['track']
    track2 = row2['track']
    if abs(track2 - track1) > 180:
        if track2 > track1:
            track1 += 360
        else:
            track2 += 360
    tracks = np.interp(fractions, [0, 1], [track1, track2]) % 360
    
    # Vertical rate
    vr_ftmin = vs_ms * M_TO_FT * 60
    vr_ftmin = clamp_value(vr_ftmin, LIMITS['vertical_rate'])
    
    # Build rows
    rows = []
    for i, t in enumerate(times):
        rows.append({
            'timestamp': t,
            'flight_id': row1['flight_id'],
            'latitude': lats[i],
            'longitude': lons[i],
            'altitude': alt_ft[i],
            'groundspeed': gs_vals[i],
            'track': tracks[i],
            'vertical_rate': vr_ftmin,
            'mach': mach_vals[i],
            'TAS': tas_vals[i],
            'CAS': cas_vals[i],
            'is_reconstructed': True,
        })
    
    return rows

# ============================================================================
# Main Processing Function
# ============================================================================

def process_flight(args):
    """
    Unified processing pipeline for a single flight.
    
    Steps:
    1. Load raw trajectory
    2. Initial filtering (derivative, sigma-median)
    3. Resample to 1s with short interpolation
    4. Detect long gaps
    5. Reconstruct gaps with OpenAP WRAP
    6. Smooth gap boundaries
    7. Apply Kalman smoothing
    8. Validate and save
    """
    partition_name, flight_path = args
    flight_id = flight_path.stem
    
    try:
        # Step 1: Load raw trajectory
        df = pd.read_parquet(flight_path)
        
        # Parse timestamp
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.dropna(subset=['latitude', 'longitude'])
        
        if len(df) < 10:
            return (partition_name, flight_id, False, "Too few points")
        
        # Get aircraft type
        typecode = df['typecode'].dropna().iloc[0] if 'typecode' in df.columns else None
        if typecode:
            typecode = str(typecode).upper()
        
        # Step 2: Create Flight object and apply initial filters
        flight = Flight(df)
        
        # 2a. Default filter
        f = flight.filter()
        if f is None or len(f.data) < 10:
            return (partition_name, flight_id, False, "Failed default filter")
        
        # 2b. Remove spikes
        f = f.filter(
            FilterDerivative(
                time_column="timestamp",
                altitude={"first": 3000, "second": 10000, "kernel": 60},
                groundspeed={"first": 150, "second": 500, "kernel": 60},
                track={"first": 20, "second": 60, "kernel": 60},
            )
        )
        if f is None or len(f.data) < 10:
            return (partition_name, flight_id, False, "Failed derivative filter")
        
        # 2c. Remove outliers
        f = f.filter(
            FilterAboveSigmaMedian(
                altitude=(17, 53),
                groundspeed=(11,),
                vertical_rate=(11,),
                track=(11,),
            )
        )
        if f is None or len(f.data) < 10:
            return (partition_name, flight_id, False, "Failed sigma filter")
        
        # 2d. Median smoothing
        f = f.filter(FilterMedian())
        if f is None or len(f.data) < 10:
            return (partition_name, flight_id, False, "Failed median filter")
        
        # Step 3: Resample to 1-second intervals
        freq_sec = pd.Timedelta(RESAMPLE_FREQ).total_seconds()
        max_interp_points = int(MAX_INTERPOLATE_SEC / freq_sec)
        
        f_resampled = f.resample(
            rule=RESAMPLE_FREQ,
            how="interpolate",
            interpolate_kw={
                "method": "linear",
                "limit": max_interp_points,
                "limit_area": "inside",
            },
            projection="lcc",
        )
        
        if f_resampled is None or len(f_resampled.data) < 10:
            return (partition_name, flight_id, False, "Resample failed")
        
        df_resampled = f_resampled.data.reset_index(drop=True)
        df_resampled = df_resampled.sort_values('timestamp')
        df_resampled = df_resampled.dropna(subset=['latitude', 'longitude', 'altitude'])
        
        if len(df_resampled) < 10:
            return (partition_name, flight_id, False, "Too few after resample")
        
        # Step 4: Detect long gaps
        df_resampled['time_diff'] = df_resampled['timestamp'].diff().dt.total_seconds()
        
        # Reset index to ensure iloc works correctly
        df_resampled = df_resampled.reset_index(drop=True)
        
        long_gap_indices = df_resampled.index[df_resampled['time_diff'] > LONG_GAP_SEC].tolist()
        
        gaps_filled = 0
        
        # Step 5: Reconstruct gaps if needed
        if long_gap_indices and typecode and typecode in TYPECODE_TO_OPENAP:
            ac_code = TYPECODE_TO_OPENAP[typecode]
            
            try:
                wrap = WRAP(ac=ac_code)
                
                synthetic_rows = []
                total_duration = (df_resampled['timestamp'].iloc[-1] - 
                                df_resampled['timestamp'].iloc[0]).total_seconds()
                
                for gap_idx in long_gap_indices:
                    # Validate indices before accessing
                    if gap_idx == 0 or gap_idx >= len(df_resampled):
                        continue
                    
                    try:
                        row_before = df_resampled.iloc[gap_idx - 1]
                        row_after = df_resampled.iloc[gap_idx]
                    except IndexError:
                        logger.warning(f"{flight_id}: Index out of bounds for gap_idx={gap_idx}, skipping gap")
                        continue
                    
                    gap_sec = (row_after['timestamp'] - row_before['timestamp']).total_seconds()
                    
                    if gap_sec > MAX_RECON_GAP_SEC:
                        continue
                    
                    times = pd.date_range(
                        start=row_before['timestamp'] + pd.Timedelta(seconds=1),
                        end=row_after['timestamp'] - pd.Timedelta(seconds=1),
                        freq=RESAMPLE_FREQ
                    )
                    
                    if len(times) == 0:
                        continue
                    
                    time_into_flight = (row_before['timestamp'] - 
                                      df_resampled['timestamp'].iloc[0]).total_seconds()
                    flight_fraction = time_into_flight / total_duration if total_duration > 0 else 0.5
                    
                    phase = classify_phase(
                        row_before['altitude'],
                        row_after['altitude'],
                        row_before.get('vertical_rate', 0),
                        flight_fraction
                    )
                    
                    try:
                        if phase == 'cruise':
                            synth = synthesize_cruise_gap(row_before, row_after, times, wrap)
                        elif phase == 'climb':
                            synth = synthesize_climb_gap(row_before, row_after, times, wrap)
                        else:
                            synth = synthesize_descent_gap(row_before, row_after, times, wrap)
                        
                        if synth:
                            synthetic_rows.extend(synth)
                            gaps_filled += 1
                    except Exception as synth_err:
                        logger.warning(f"{flight_id}: Gap synthesis failed for {phase} phase: {synth_err}")
                        continue
                
                # Combine original and synthetic
                df_resampled['is_reconstructed'] = False
                
                if synthetic_rows:
                    df_synth = pd.DataFrame(synthetic_rows)
                    df_combined = pd.concat([df_resampled, df_synth], ignore_index=True)
                    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
                    
                    # Step 6: Smooth gap boundaries
                    # Find new gap positions in combined DataFrame based on timestamps
                    df_combined['time_diff_new'] = df_combined['timestamp'].diff().dt.total_seconds()
                    new_gap_indices = df_combined.index[df_combined['time_diff_new'] > 2.0].tolist()
                    
                    if new_gap_indices:
                        df_combined = smooth_gap_boundaries(df_combined, new_gap_indices)
                    
                    # Drop temporary column
                    df_combined = df_combined.drop(columns=['time_diff_new'], errors='ignore')
                else:
                    df_combined = df_resampled.copy()
                    
            except Exception as e:
                logger.warning(f"{flight_id}: WRAP reconstruction failed: {e}")
                df_combined = df_resampled.copy()
                df_combined['is_reconstructed'] = False
        else:
            df_combined = df_resampled.copy()
            df_combined['is_reconstructed'] = False
        
        # Step 7: Validate and clamp kinematics
        df_combined = validate_and_clamp_kinematics(df_combined)
        
        # Step 8: Final Kalman smoothing
        if 'flight_id' not in df_combined.columns:
            df_combined['flight_id'] = flight_id
        
        try:
            flight_final = Flight(df_combined)
            f_xy = flight_final.compute_xy(EuroPP())
            
            if f_xy is not None and len(f_xy.data) >= 10:
                # Validate before Kalman
                required_cols = ['x', 'y', 'altitude', 'groundspeed', 'track', 'vertical_rate']
                if all(col in f_xy.data.columns for col in required_cols):
                    # Fill small gaps in vertical_rate
                    if f_xy.data['vertical_rate'].isna().sum() > 0:
                        f_xy.data['vertical_rate'] = f_xy.data['vertical_rate'].interpolate(
                            method='linear', limit=3, limit_area='inside'
                        ).fillna(0)
                    
                    try:
                        f_kalman = f_xy.filter(KalmanSmoother6D(reject_sigma=3))
                        if f_kalman is not None and len(f_kalman.data) >= 10:
                            df_final = f_kalman.data.drop(columns=['x', 'y'], errors='ignore')
                        else:
                            df_final = f_xy.data.drop(columns=['x', 'y'], errors='ignore')
                    except:
                        df_final = f_xy.data.drop(columns=['x', 'y'], errors='ignore')
                else:
                    df_final = df_combined
            else:
                df_final = df_combined
        except Exception as e:
            logger.warning(f"{flight_id}: Final smoothing failed: {e}")
            df_final = df_combined
        
        # Step 9: Final validation and save
        df_final = validate_and_clamp_kinematics(df_final)
        df_final = df_final.dropna(subset=['latitude', 'longitude', 'altitude'])
        
        if len(df_final) < 10:
            return (partition_name, flight_id, False, "Too few after final processing")
        
        # Ensure flight_id
        if 'flight_id' not in df_final.columns:
            df_final['flight_id'] = flight_id
        
        # Ensure is_reconstructed flag
        if 'is_reconstructed' not in df_final.columns:
            df_final['is_reconstructed'] = False
        
        # Select output columns
        df_out = df_final[OUTPUT_COLUMNS]
        
        # Save
        out_dir = OUT_ROOT / partition_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / flight_path.name
        df_out.to_parquet(out_path, index=False)
        
        return (partition_name, flight_id, True, f"Success ({gaps_filled} gaps filled)")
        
    except Exception as e:
        logger.error(f"{flight_id}: {e}")
        return (partition_name, flight_id, False, str(e))

# ============================================================================
# Main
# ============================================================================

def get_all_flights():
    """Collect all flight parquet files."""
    jobs = []
    for part_name, part_dir in PARTITIONS.items():
        if not part_dir.exists():
            continue
        
        files = list(part_dir.glob("*.parquet"))
        logger.info(f"Found {len(files)} flights in '{part_name}'")
        
        for fp in files:
            jobs.append((part_name, fp))
    
    return jobs

def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("UNIFIED TRAJECTORY PROCESSING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Resample: {RESAMPLE_FREQ}")
    logger.info(f"  Short gap interpolation: <{MAX_INTERPOLATE_SEC}s")
    logger.info(f"  Long gap reconstruction: {LONG_GAP_SEC}s - {MAX_RECON_GAP_SEC}s")
    logger.info(f"  Boundary smoothing: {BOUNDARY_SMOOTH_WINDOW} points")
    logger.info("")
    
    jobs = get_all_flights()
    
    if not jobs:
        logger.error("No flights found!")
        return
    
    logger.info(f"Total flights to process: {len(jobs)}")
    logger.info(f"Using {MAX_WORKERS} parallel workers")
    logger.info("")
    
    results = {part: {"success": 0, "failed": 0} for part in PARTITIONS.keys()}
    failed_flights = []
    
    start_time = time.time()
    logger.info(f"Starting processing at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_flight, job): job for job in jobs}
        
        completed_count = 0
        with tqdm(total=len(jobs), desc="Processing", unit="flight") as pbar:
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    part_name, flight_id, ok, msg = fut.result()
                    
                    if ok:
                        results[part_name]["success"] += 1
                    else:
                        results[part_name]["failed"] += 1
                        failed_flights.append((part_name, flight_id, msg))
                    
                    completed_count += 1
                    # Update speed every 100 flights
                    if completed_count % 100 == 0:
                        elapsed = time.time() - start_time
                        speed = completed_count / elapsed
                        remaining = len(jobs) - completed_count
                        eta_sec = remaining / speed if speed > 0 else 0
                        pbar.set_postfix({"speed": f"{speed:.1f} flights/s", "ETA": f"{eta_sec/60:.1f}min"})
                        
                except Exception as e:
                    logger.error(f"Future exception for {job[1].stem}: {e}")
                    results[job[0]]["failed"] += 1
                    failed_flights.append((job[0], job[1].stem, str(e)))
                    completed_count += 1
                
                pbar.update(1)
    
    # Report results
    end_time = time.time()
    total_time = end_time - start_time
    avg_speed = len(jobs) / total_time if total_time > 0 else 0
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    logger.info(f"Average speed: {avg_speed:.2f} flights/second")
    logger.info(f"Total flights processed: {len(jobs)}")
    
    for part_name in PARTITIONS.keys():
        total = results[part_name]["success"] + results[part_name]["failed"]
        success = results[part_name]["success"]
        failed = results[part_name]["failed"]
        
        if total > 0:
            logger.info(f"{part_name.upper()}:")
            logger.info(f"  Success: {success}/{total} ({success/total*100:.1f}%)")
            logger.info(f"  Failed: {failed}/{total} ({failed/total*100:.1f}%)")
    
    if failed_flights:
        logger.warning(f"\nFailed flights: {len(failed_flights)}")
        for part, fid, msg in failed_flights[:10]:
            logger.warning(f"  {part}/{fid}: {msg}")
        if len(failed_flights) > 10:
            logger.warning(f"  ... and {len(failed_flights) - 10} more")
    
    logger.info(f"\nOutput saved to: {OUT_ROOT}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
