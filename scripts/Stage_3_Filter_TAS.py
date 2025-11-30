"""
Stage 4: TAS/CAS/GS Filling with Physics-Based Validation
===========================================================

Correct Processing Sequence:
1. Filter input data outliers (phase-based + global caps on raw data)
2. Fill missing TAS using priority order:
   a) ACARS TAS (ground truth)
   b) Mach + Temperature → TAS (with validation, cruise only)
   c) CAS + Density → TAS (climb/descent preferred)
   d) GS + Wind → TAS (vector method, low altitude preferred)
   e) Interpolation (short gaps)
   f) Taxi clamp
   g) GS from lat/lon movement (calculate missing GS, filter with global caps)
   h) GS fallback (for remaining TAS gaps, assume minimal wind)
3. NO final filtering - preserve all filled values

Physics-Based Filters:
- Phase-based thresholds (ground/climb/descent/cruise) - ONLY on input data
- Global hard caps (GS, TAS, CAS, Mach, VR) - ONLY on input data

Author: PRC 2025 Fuel Challenge
Date: November 12, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Pipeline directories
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"
STAGE_2_OUTPUT = PROCESSED_DIR / "trajectories_era5"      # Stage 2 output (input for Stage 3)
STAGE_3_OUTPUT = PROCESSED_DIR / "trajectories_tas_filled"  # Stage 3 output

# Physical constants
GAMMA = 1.4
R = 287.05287  # J/(kg*K)
KTS_PER_MPS = 1.9438444924406048
RHO0 = 1.225  # kg/m³

# Global caps (converted to m/s for internal calculations)
CAPS = {
    'gs_max_kts': 750,  # flag 700-750, reject >750
    'gs_flag_kts': 700,
    'tas_min_kts': 60,  # only if airborne
    'tas_max_kts': 600,
    'cas_max_kts': 420,
    'mach_min': 0.40,
    'mach_max': 0.90,
    'vr_max_fpm': 8000,
}

# Phase-based caps
PHASE_LIMITS = {
    'ground': {
        'altitude_max_ft': 1500,
        'gs_max_kts': 200,
    },
    'climb_descent': {
        'altitude_min_ft': 1500,
        'altitude_max_ft': 20000,
        'tas_max_kts': 500,
        'vr_climb_max_fpm': 5000,
        'vr_descent_min_fpm': -5000,
    },
    'cruise': {
        'altitude_min_ft': 20000,
        'gs_max_kts': 700,  # flag 700-750 by global cap
        'tas_min_kts': 350,
        'tas_max_kts': 550,
        'vr_max_fpm': 1000,
    },
}

# Consistency thresholds
CONSISTENCY = {
    'mach_tas_tolerance': 0.08,  # fraction of speed of sound
    'wind_gs_tolerance_kts': 200,  # GS must be within TAS ± 200kt
}


def validate_mach_for_altitude(mach: np.ndarray, altitude_ft: np.ndarray) -> np.ndarray:
    """
    Validate Mach numbers - reject physically impossible values.
    
    Rules:
    - Below 10,000 ft: Mach should be < 0.5 (typical < 0.4)
    - 10,000-20,000 ft: Mach should be < 0.7
    - Above 20,000 ft: Mach can be up to 0.95 (relaxed to 0.90 by global cap)
    
    Returns:
        Boolean mask of valid Mach values
    """
    valid = np.ones(len(mach), dtype=bool)
    
    # Reject impossible values (updated caps)
    valid &= (mach >= CAPS['mach_min']) & (mach <= CAPS['mach_max'])
    
    # Altitude-based validation
    low_alt = altitude_ft < 10000
    mid_alt = (altitude_ft >= 10000) & (altitude_ft < 20000)
    
    valid[low_alt] &= mach[low_alt] < 0.5
    valid[mid_alt] &= mach[mid_alt] < 0.7
    
    return valid


def apply_global_caps(df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
    """
    Apply global hard caps to speed and vertical rate data.
    
    Sets values outside limits to NaN so interpolation can handle them.
    """
    capped_stats = {'gs': 0, 'tas': 0, 'cas': 0, 'mach': 0, 'vr': 0}
    
    # Groundspeed: 0-750 kt (flag 700-750)
    if 'groundspeed' in df.columns:
        mask = (df['groundspeed'] > CAPS['gs_max_kts']) | (df['groundspeed'] < 0)
        if mask.any():
            capped_stats['gs'] = mask.sum()
            df.loc[mask, 'groundspeed'] = np.nan
    
    # TAS: 60-600 kt (lower bound only if airborne)
    if 'TAS' in df.columns and 'altitude' in df.columns:
        airborne = df['altitude'] > PHASE_LIMITS['ground']['altitude_max_ft']
        mask = (df['TAS'] > CAPS['tas_max_kts']) | ((df['TAS'] < CAPS['tas_min_kts']) & airborne)
        if mask.any():
            capped_stats['tas'] = mask.sum()
            df.loc[mask, 'TAS'] = np.nan
    
    # CAS: 0-420 kt
    if 'CAS' in df.columns:
        mask = (df['CAS'] > CAPS['cas_max_kts']) | (df['CAS'] < 0)
        if mask.any():
            capped_stats['cas'] = mask.sum()
            df.loc[mask, 'CAS'] = np.nan
    
    # Mach: 0.40-0.90
    if 'mach' in df.columns:
        mask = (df['mach'] > CAPS['mach_max']) | (df['mach'] < CAPS['mach_min'])
        if mask.any():
            capped_stats['mach'] = mask.sum()
            df.loc[mask, 'mach'] = np.nan
    
    # Vertical rate: ±8000 fpm
    if 'vertical_rate' in df.columns:
        mask = np.abs(df['vertical_rate']) > CAPS['vr_max_fpm']
        if mask.any():
            capped_stats['vr'] = mask.sum()
            df.loc[mask, 'vertical_rate'] = np.nan
    
    if debug_mode and any(capped_stats.values()):
        logger.info(f"  Global caps applied: GS={capped_stats['gs']}, TAS={capped_stats['tas']}, "
                   f"CAS={capped_stats['cas']}, Mach={capped_stats['mach']}, VR={capped_stats['vr']}")
    
    return df


def apply_phase_based_limits(df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
    """
    Apply phase-based speed and vertical rate limits.
    
    Phases:
    - Ground: altitude < 1500 ft
    - Climb/Descent: 1500-20000 ft
    - Cruise: >20000 ft
    """
    phase_stats = {'ground': 0, 'climb_descent': 0, 'cruise': 0}
    
    if 'altitude' not in df.columns:
        return df
    
    # Define phases
    ground = df['altitude'] < PHASE_LIMITS['ground']['altitude_max_ft']
    climb_descent = (df['altitude'] >= PHASE_LIMITS['climb_descent']['altitude_min_ft']) & \
                    (df['altitude'] < PHASE_LIMITS['climb_descent']['altitude_max_ft'])
    cruise = df['altitude'] >= PHASE_LIMITS['cruise']['altitude_min_ft']
    
    # Ground: GS ≤ 200 kt
    if 'groundspeed' in df.columns:
        mask = ground & (df['groundspeed'] > PHASE_LIMITS['ground']['gs_max_kts'])
        if mask.any():
            phase_stats['ground'] += mask.sum()
            df.loc[mask, 'groundspeed'] = np.nan
    
    # Climb/Descent: TAS ≤ 500 kt
    if 'TAS' in df.columns:
        mask = climb_descent & (df['TAS'] > PHASE_LIMITS['climb_descent']['tas_max_kts'])
        if mask.any():
            phase_stats['climb_descent'] += mask.sum()
            df.loc[mask, 'TAS'] = np.nan
    
    # Climb/Descent: VR limits
    if 'vertical_rate' in df.columns:
        # Climb: 0 to +5000 fpm
        mask_climb = climb_descent & (df['vertical_rate'] > 0) & \
                     (df['vertical_rate'] > PHASE_LIMITS['climb_descent']['vr_climb_max_fpm'])
        # Descent: -5000 to 0 fpm
        mask_descent = climb_descent & (df['vertical_rate'] < 0) & \
                       (df['vertical_rate'] < PHASE_LIMITS['climb_descent']['vr_descent_min_fpm'])
        
        if mask_climb.any() or mask_descent.any():
            phase_stats['climb_descent'] += mask_climb.sum() + mask_descent.sum()
            df.loc[mask_climb | mask_descent, 'vertical_rate'] = np.nan
    
    # Cruise: GS ≤ 700 kt (flagged 700-750 by global cap)
    if 'groundspeed' in df.columns:
        mask = cruise & (df['groundspeed'] > PHASE_LIMITS['cruise']['gs_max_kts'])
        if mask.any():
            phase_stats['cruise'] += mask.sum()
            # Don't set to NaN yet - global cap will handle 700-750 range
            # Only clip if between 700-750 to flag as suspect
            flag_mask = mask & (df['groundspeed'] <= CAPS['gs_max_kts'])
            if flag_mask.any() and debug_mode:
                logger.info(f"    Flagged {flag_mask.sum()} cruise GS points in 700-750 kt range")
    
    # Cruise: TAS 350-550 kt
    if 'TAS' in df.columns:
        mask = cruise & ((df['TAS'] < PHASE_LIMITS['cruise']['tas_min_kts']) | 
                        (df['TAS'] > PHASE_LIMITS['cruise']['tas_max_kts']))
        if mask.any():
            phase_stats['cruise'] += mask.sum()
            df.loc[mask, 'TAS'] = np.nan
    
    # Cruise: VR ±1000 fpm
    if 'vertical_rate' in df.columns:
        mask = cruise & (np.abs(df['vertical_rate']) > PHASE_LIMITS['cruise']['vr_max_fpm'])
        if mask.any():
            phase_stats['cruise'] += mask.sum()
            df.loc[mask, 'vertical_rate'] = np.nan
    
    if debug_mode and any(phase_stats.values()):
        logger.info(f"  Phase limits applied: Ground={phase_stats['ground']}, "
                   f"Climb/Descent={phase_stats['climb_descent']}, Cruise={phase_stats['cruise']}")
    
    return df


def apply_consistency_checks(df: pd.DataFrame, temp_k: pd.Series, debug_mode: bool = False) -> pd.DataFrame:
    """
    Apply consistency checks between related speeds.
    
    Checks:
    1. Mach-TAS consistency: |TAS_reported - TAS_from_Mach| < 0.08 * speed_of_sound
    2. Wind-bounded GS: GS must be within TAS ± 200 kt
    """
    consistency_stats = {'mach_tas': 0, 'wind_gs': 0}
    
    # Check 1: Mach-TAS consistency
    if all(col in df.columns for col in ['mach', 'TAS']) and temp_k is not None:
        has_both = pd.notna(df['mach']) & pd.notna(df['TAS']) & pd.notna(temp_k)
        
        if has_both.any():
            # Calculate TAS from Mach
            speed_of_sound = np.sqrt(GAMMA * R * temp_k[has_both].values)  # m/s
            tas_from_mach_mps = df.loc[has_both, 'mach'].values * speed_of_sound
            tas_reported_mps = df.loc[has_both, 'TAS'].values / KTS_PER_MPS
            
            # Check if difference exceeds tolerance
            tolerance_mps = CONSISTENCY['mach_tas_tolerance'] * speed_of_sound
            diff = np.abs(tas_reported_mps - tas_from_mach_mps)
            inconsistent = diff > tolerance_mps
            
            if inconsistent.any():
                consistency_stats['mach_tas'] = inconsistent.sum()
                # Set both Mach and TAS to NaN where inconsistent
                inconsistent_idx = df[has_both].index[inconsistent]
                df.loc[inconsistent_idx, ['mach', 'TAS']] = np.nan
    
    # Check 2: Wind-bounded GS
    if all(col in df.columns for col in ['groundspeed', 'TAS']):
        has_both = pd.notna(df['groundspeed']) & pd.notna(df['TAS'])
        
        if has_both.any():
            gs_tas_diff = np.abs(df.loc[has_both, 'groundspeed'] - df.loc[has_both, 'TAS'])
            invalid = gs_tas_diff > CONSISTENCY['wind_gs_tolerance_kts']
            
            if invalid.any():
                consistency_stats['wind_gs'] = invalid.sum()
                # Set GS to NaN where wind effect is impossible
                invalid_idx = df[has_both].index[invalid]
                df.loc[invalid_idx, 'groundspeed'] = np.nan
    
    if debug_mode and any(consistency_stats.values()):
        logger.info(f"  Consistency checks: Mach-TAS={consistency_stats['mach_tas']}, "
                   f"Wind-GS={consistency_stats['wind_gs']}")
    
    return df


def choose_speed_by_altitude_regime(tas_mach: np.ndarray, tas_cas: np.ndarray, tas_gswind: np.ndarray, 
                                     altitude_ft: np.ndarray, has_mach: np.ndarray, has_cas: np.ndarray, 
                                     has_gswind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose TAS source based on altitude regime.
    
    Regime rules:
    - Below FL100 (10,000 ft): Prefer GS+Wind, then CAS, then Mach
    - FL100-FL200: Prefer CAS, then Mach, then GS+Wind  
    - Above FL200 (cruise): Prefer Mach, then CAS, then GS+Wind
    
    Returns:
        (tas_best, source_code) where source_code: 1=Mach, 2=CAS, 3=GS+Wind
    """
    tas_best = np.full(len(altitude_ft), np.nan)
    source_code = np.zeros(len(altitude_ft), dtype=int)
    
    # Define altitude regimes
    low_alt = altitude_ft < 10000  # Below FL100
    mid_alt = (altitude_ft >= 10000) & (altitude_ft < 20000)  # FL100-FL200
    high_alt = altitude_ft >= 20000  # Above FL200 - cruise
    
    # tas_arrays indexed by code: 1=Mach, 2=CAS, 3=GS+Wind
    tas_arrays = [tas_mach, tas_cas, tas_gswind]
    
    for mask, sources, codes in [
        (low_alt, [has_gswind, has_cas, has_mach], [3, 2, 1]),
        (mid_alt, [has_cas, has_mach, has_gswind], [2, 1, 3]),
        (high_alt, [has_mach, has_cas, has_gswind], [1, 2, 3])
    ]:
        for src_idx, code in zip(sources, codes):
            available = mask & src_idx & np.isnan(tas_best)
            if available.any():
                arr_idx = code - 1
                tas_best[available] = tas_arrays[arr_idx][available]
                source_code[available] = code
    
    return tas_best, source_code


def calculate_track_from_position(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculate track (heading/bearing) from latitude/longitude position changes.
    
    Uses the forward azimuth formula to compute bearing between consecutive points.
    
    Args:
        lat: Latitude array (degrees)
        lon: Longitude array (degrees)
    
    Returns:
        Track array in degrees (0-360), NaN for last point
    """
    # Convert to radians
    lat1 = np.deg2rad(lat[:-1])
    lat2 = np.deg2rad(lat[1:])
    lon1 = np.deg2rad(lon[:-1])
    lon2 = np.deg2rad(lon[1:])
    
    dlon = lon2 - lon1
    
    # Calculate bearing using standard formula
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    # Calculate bearing in radians, then convert to degrees
    track_calculated = np.rad2deg(np.arctan2(x, y))
    
    # Normalize to 0-360
    track_calculated = (track_calculated + 360) % 360
    
    # Append NaN for last point
    track_calculated = np.append(track_calculated, np.nan)
    
    return track_calculated


def calculate_groundspeed_from_position(lat: np.ndarray, lon: np.ndarray, 
                                        timestamps: pd.Series) -> np.ndarray:
    """
    Calculate groundspeed from latitude/longitude position changes.
    
    Uses Haversine formula to compute distance between consecutive points,
    then divides by time elapsed.
    
    Args:
        lat: Latitude array (degrees)
        lon: Longitude array (degrees)
        timestamps: Timestamp series
    
    Returns:
        Groundspeed array in knots, NaN for last point
    """
    # Earth radius in meters
    R_EARTH = 6371000.0
    
    # Convert to radians
    lat1 = np.deg2rad(lat[:-1])
    lat2 = np.deg2rad(lat[1:])
    lon1 = np.deg2rad(lon[:-1])
    lon2 = np.deg2rad(lon[1:])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance_m = R_EARTH * c
    
    # Calculate time difference in seconds
    time_diff = timestamps.diff().dt.total_seconds().values[1:]
    
    # Avoid division by zero
    time_diff = np.where(time_diff > 0, time_diff, np.nan)
    
    # Speed in m/s
    speed_mps = distance_m / time_diff
    
    # Convert to knots
    speed_kts = speed_mps * KTS_PER_MPS
    
    # Append NaN for last point
    speed_kts = np.append(speed_kts, np.nan)
    
    return speed_kts


def calculate_track_from_position(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculate track (heading/bearing) from latitude/longitude position changes.
    
    Uses the forward azimuth formula to compute bearing between consecutive points.
    
    Args:
        lat: Latitude array (degrees)
        lon: Longitude array (degrees)
    
    Returns:
        Track array in degrees (0-360), NaN for last point
    """
    # Convert to radians
    lat1 = np.deg2rad(lat[:-1])
    lat2 = np.deg2rad(lat[1:])
    lon1 = np.deg2rad(lon[:-1])
    lon2 = np.deg2rad(lon[1:])
    
    dlon = lon2 - lon1
    
    # Calculate bearing using standard formula
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    # Calculate bearing in radians, then convert to degrees
    track_calculated = np.rad2deg(np.arctan2(x, y))
    
    # Normalize to 0-360
    track_calculated = (track_calculated + 360) % 360
    
    # Append NaN for last point
    track_calculated = np.append(track_calculated, np.nan)
    
    return track_calculated


def fill_missing_speeds_bidirectional(df: pd.DataFrame, temp_k: pd.Series, rho: np.ndarray, 
                                      debug_mode: bool = False) -> pd.DataFrame:
    """
    Fill missing CAS/Mach/GS from known TAS using altitude regime logic.
    
    After TAS is filled, calculate missing complementary speeds:
    - TAS → CAS (using density)
    - TAS → Mach (using temperature)
    - TAS → GS (using wind - reverse calculation)
    """
    filled_count = {'cas': 0, 'mach': 0, 'gs': 0}
    
    # TAS → CAS (where CAS missing)
    if 'CAS' in df.columns and 'TAS' in df.columns:
        cas_missing = pd.isna(df['CAS']) & pd.notna(df['TAS'])
        if cas_missing.any():
            tas_mps = df.loc[cas_missing, 'TAS'].values / KTS_PER_MPS
            cas_mps = tas_mps * np.sqrt(rho[cas_missing] / RHO0)
            df.loc[cas_missing, 'CAS'] = cas_mps * KTS_PER_MPS
            filled_count['cas'] = cas_missing.sum()
    
    # TAS → Mach (where Mach missing and in cruise regime)
    if 'mach' in df.columns and 'TAS' in df.columns and 'altitude' in df.columns:
        mach_missing = pd.isna(df['mach']) & pd.notna(df['TAS'])
        cruise_regime = df['altitude'] > 20000  # Only fill Mach in cruise
        fill_mach = mach_missing & cruise_regime & pd.notna(temp_k)
        
        if fill_mach.any():
            tas_mps = df.loc[fill_mach, 'TAS'].values / KTS_PER_MPS
            speed_of_sound = np.sqrt(GAMMA * R * temp_k[fill_mach].values)
            calculated_mach = tas_mps / speed_of_sound
            
            # Apply Mach caps to calculated values (0.40-0.90)
            calculated_mach = np.clip(calculated_mach, CAPS['mach_min'], CAPS['mach_max'])
            
            df.loc[fill_mach, 'mach'] = calculated_mach
            filled_count['mach'] = fill_mach.sum()
    
    # TAS → GS (reverse wind calculation - where GS missing)
    if all(c in df.columns for c in ['groundspeed', 'TAS', 'track', 'u_component_of_wind_pl', 'v_component_of_wind_pl']):
        gs_missing = pd.isna(df['groundspeed']) & pd.notna(df['TAS']) & pd.notna(df['track'])
        wind_available = pd.notna(df['u_component_of_wind_pl']) & pd.notna(df['v_component_of_wind_pl'])
        fill_gs = gs_missing & wind_available
        
        if fill_gs.any():
            tas_mps = df.loc[fill_gs, 'TAS'].values / KTS_PER_MPS
            track_rad = np.deg2rad(df.loc[fill_gs, 'track'].values)
            
            # Air velocity components
            vE_air = tas_mps * np.sin(track_rad)
            vN_air = tas_mps * np.cos(track_rad)
            
            # Ground velocity = Air velocity + Wind velocity
            vE_ground = vE_air + df.loc[fill_gs, 'u_component_of_wind_pl'].values
            vN_ground = vN_air + df.loc[fill_gs, 'v_component_of_wind_pl'].values
            
            # GS is magnitude
            gs_mps = np.hypot(vE_ground, vN_ground)
            df.loc[fill_gs, 'groundspeed'] = gs_mps * KTS_PER_MPS
            filled_count['gs'] = fill_gs.sum()
    
    if debug_mode and any(filled_count.values()):
        logger.info(f"  Bidirectional fill: CAS={filled_count['cas']}, Mach={filled_count['mach']}, GS={filled_count['gs']}")
    
    return df


def process_single_flight(traj_file: Path, output_file: Path, debug_mode: bool = False) -> Tuple[str, str, str]:
    """
    Process a single flight: fill TAS/CAS with physics-based methods and filtering.
    
    Returns:
        Tuple of (status, flight_id, error_message)
    """
    flight_id = traj_file.stem
    
    try:
        # Load trajectory with ERA5 data
        df = pd.read_parquet(traj_file)
        
        if debug_mode:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {flight_id}")
            logger.info(f"Rows: {len(df)}")
        
        # ===== STEP 1: Filter INPUT data outliers (phase-based + global on raw data) =====
        # Remove bad input data BEFORE using it for calculations
        df = apply_phase_based_limits(df, debug_mode)  # Phase-based first
        df = apply_global_caps(df, debug_mode)         # Then global caps on raw inputs
        
        # Initialize TAS from existing data (ACARS truth)
        if "TAS" in df.columns:
            tas_mps = df['TAS'].values / KTS_PER_MPS
        elif "tas" in df.columns:
            tas_mps = df['tas'].values / KTS_PER_MPS
        else:
            tas_mps = np.full(len(df), np.nan)
        
        # Calculate track from position if missing
        if 'track' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
            track_missing = df['track'].isna()
            if track_missing.any():
                track_calc = calculate_track_from_position(df['latitude'].values, df['longitude'].values)
                df.loc[track_missing, 'track'] = track_calc[track_missing]
        
        # Prepare ERA5 temperature
        if "temperature_pl" in df.columns:
            temp_k = df['temperature_pl'].astype(float)
        else:
            # Fallback to ISA
            from fill_tas_complete import isa_tp_rho_from_alt
            temp_k_arr, _, _ = isa_tp_rho_from_alt(df['altitude'].values * 0.3048)
            temp_k = pd.Series(temp_k_arr)
        
        # Prepare pressure
        if "pressure_hpa" in df.columns:
            pres_pa = df['pressure_hpa'].values * 100.0
        else:
            from fill_tas_complete import isa_tp_rho_from_alt
            _, pres_pa, _ = isa_tp_rho_from_alt(df['altitude'].values * 0.3048)
        
        # Calculate air density
        rho = pres_pa / (R * temp_k.values)
        
        # ===== STEP 2: Calculate TAS from 3 methods with validation =====
        
        # Method 1: Mach + Temperature → TAS
        tas_mach = np.full(len(df), np.nan)
        has_mach = np.zeros(len(df), dtype=bool)
        
        if "mach" in df.columns:
            mach_candidate = pd.notna(df['mach']) & pd.notna(temp_k) & (df['mach'] > 0) & pd.isna(tas_mps)
            
            if mach_candidate.any():
                valid_mach = validate_mach_for_altitude(
                    df.loc[mach_candidate, 'mach'].values,
                    df.loc[mach_candidate, 'altitude'].values
                )
                
                mach_valid_idx = df[mach_candidate].index[valid_mach]
                
                if len(mach_valid_idx) > 0:
                    speed_of_sound = np.sqrt(GAMMA * R * temp_k.loc[mach_valid_idx].values)
                    tas_mach[mach_valid_idx] = df.loc[mach_valid_idx, 'mach'].values * speed_of_sound
                    has_mach[mach_valid_idx] = True
        
        # Method 2: CAS + Density → TAS
        tas_cas = np.full(len(df), np.nan)
        has_cas = np.zeros(len(df), dtype=bool)
        
        if "CAS" in df.columns or "cas" in df.columns:
            cas_col = "CAS" if "CAS" in df.columns else "cas"
            cas_candidate = pd.notna(df[cas_col]) & np.isfinite(rho) & pd.isna(tas_mps)
            
            if cas_candidate.any():
                cas_mps = df.loc[cas_candidate, cas_col].values / KTS_PER_MPS
                tas_cas[cas_candidate] = cas_mps / np.sqrt(rho[cas_candidate] / RHO0)
                has_cas[cas_candidate] = True
        
        # Method 3: GS + Wind → TAS
        tas_gswind = np.full(len(df), np.nan)
        has_gswind = np.zeros(len(df), dtype=bool)
        
        wind_cols = ["groundspeed", "track", "u_component_of_wind_pl", "v_component_of_wind_pl"]
        if all(col in df.columns for col in wind_cols):
            wind_candidate = (
                pd.notna(df['groundspeed']) & 
                pd.notna(df['track']) & 
                pd.notna(df['u_component_of_wind_pl']) & 
                pd.notna(df['v_component_of_wind_pl']) &
                pd.isna(tas_mps)
            )
            
            if wind_candidate.any():
                gs_mps = df.loc[wind_candidate, 'groundspeed'].values / KTS_PER_MPS
                track_rad = np.deg2rad(df.loc[wind_candidate, 'track'].values)
                
                vE_ground = gs_mps * np.sin(track_rad)
                vN_ground = gs_mps * np.cos(track_rad)
                
                vE_air = vE_ground - df.loc[wind_candidate, 'u_component_of_wind_pl'].values
                vN_air = vN_ground - df.loc[wind_candidate, 'v_component_of_wind_pl'].values
                
                tas_gswind[wind_candidate] = np.hypot(vE_air, vN_air)
                has_gswind[wind_candidate] = True
        
        # ===== STEP 3: Choose best TAS using altitude-regime logic =====
        altitude_ft = df['altitude'].values
        tas_chosen, source_code = choose_speed_by_altitude_regime(
            tas_mach, tas_cas, tas_gswind, altitude_ft,
            has_mach, has_cas, has_gswind
        )
        
        # Apply chosen TAS
        filled_mask = pd.notna(tas_chosen) & pd.isna(tas_mps)
        tas_mps[filled_mask] = tas_chosen[filled_mask]
        
        # ===== STEP 4: Interpolation (240 samples = 120s at 0.5s) =====
        tas_series = pd.Series(tas_mps, index=df.index, dtype=float)
        tas_series = tas_series.interpolate(method='linear', limit=240, limit_direction='both')
        tas_mps = tas_series.values
        
        # ===== STEP 5: Taxi clamp =====
        if "groundspeed" in df.columns:
            taxi_mask = (
                ((df['altitude'] < 500) & (df['groundspeed'] < 50)) |
                (df['groundspeed'] < 5)
            )
            taxi_mask = taxi_mask & pd.isna(tas_mps)
            
            if taxi_mask.any():
                gs_mps = df.loc[taxi_mask, 'groundspeed'].values / KTS_PER_MPS
                tas_mps[taxi_mask.values] = np.clip(gs_mps, 0.0, 1.5)
        
        # ===== STEP 6: Calculate missing GS from lat/lon movement =====
        # Fill missing groundspeed using position changes (Haversine formula)
        if "groundspeed" in df.columns and "latitude" in df.columns and "longitude" in df.columns and "timestamp" in df.columns:
            gs_missing = pd.isna(df['groundspeed'])
            
            if gs_missing.any():
                # Calculate GS from position changes for all points
                gs_calculated = calculate_groundspeed_from_position(
                    df['latitude'].values,
                    df['longitude'].values,
                    df['timestamp']
                )
                
                # Apply global GS caps (0-750 kt) to calculated values ONLY
                gs_calculated = np.where(
                    (gs_calculated >= 0) & (gs_calculated <= CAPS['gs_max_kts']),
                    gs_calculated,
                    np.nan
                )
                
                # Fill missing GS with calculated values
                df.loc[gs_missing, 'groundspeed'] = gs_calculated[gs_missing.values]
                
                filled_gs_count = pd.notna(gs_calculated[gs_missing.values]).sum()
                if debug_mode and filled_gs_count > 0:
                    logger.info(f"  GS from position: Filled {filled_gs_count} points (filtered with global caps)")
        
        # ===== STEP 7: GS Fallback (for remaining TAS gaps - assume minimal wind) =====
        # After all other methods, if TAS is still missing but GS exists, use GS as approximation
        # This assumes wind effect is minimal, useful for low altitude or when no wind data available
        if "groundspeed" in df.columns:
            gs_fallback_mask = pd.isna(tas_mps) & pd.notna(df['groundspeed'])
            
            if gs_fallback_mask.any():
                gs_mps_fallback = df.loc[gs_fallback_mask, 'groundspeed'].values / KTS_PER_MPS
                tas_mps[gs_fallback_mask.values] = gs_mps_fallback
                
                if debug_mode:
                    logger.info(f"  GS Fallback: Filled {gs_fallback_mask.sum()} TAS points using GS (assuming minimal wind)")
        
        # ===== STEP 8: Final sanity check and update =====
        tas_mps = np.clip(tas_mps, 0.0, 400.0)
        df['TAS'] = tas_mps * KTS_PER_MPS
        
        # Calculate CAS from TAS + ERA5 density
        cas_mps = tas_mps * np.sqrt(rho / RHO0)
        cas_mps = np.clip(cas_mps, 0.0, 400.0)
        df['CAS'] = cas_mps * KTS_PER_MPS
        
        # Bidirectional fill
        df = fill_missing_speeds_bidirectional(df, temp_k, rho, debug_mode)
        
        # Create provenance column
        provenance = np.full(len(df), 'unknown', dtype=object)
        provenance[pd.notna(df['TAS'])] = 'filled'
        df['tas_source'] = provenance
        
        # Save
        df.to_parquet(output_file, index=False)
        
        if debug_mode:
            tas_coverage = pd.notna(df['TAS']).sum() / len(df) * 100
            logger.info(f"  Final TAS Coverage: {tas_coverage:.1f}%")
        
        return ('success', flight_id, None)
        
    except Exception as e:
        logger.error(f"Error processing {flight_id}: {e}")
        if debug_mode:
            import traceback
            logger.error(traceback.format_exc())
        return ('failed', flight_id, str(e))


def process_flight_wrapper(args):
    """Wrapper function for multiprocessing."""
    traj_file, output_file, debug_mode = args
    return process_single_flight(traj_file, output_file, debug_mode)


def stage4_fill_tas(dataset: str, dirs: Dict, debug_mode: bool = False, max_flights: int = None, 
                   skip_existing: bool = True, workers: int = 1) -> Tuple[int, list]:
    """
    Main entry point for Stage 4: Fill TAS/CAS with physics-based methods and filtering.
    
    CORRECTED Processing Sequence:
    1. Filter INPUT data outliers (ONLY ONCE at start):
       - Phase-based limits (remove bad GS/TAS/VR by flight phase)
       - Global caps (remove impossible raw values)
    
    2. Fill missing TAS using priority order:
       a) ACARS TAS (ground truth)
       b) Mach + Temperature → TAS (with validation, cruise only)
       c) CAS + Density → TAS (climb/descent preferred)
       d) GS + Wind → TAS (vector method, low altitude preferred)
       e) Interpolation (short gaps)
       f) Taxi clamp
       g) GS from lat/lon movement (calculate missing GS, filter with global caps)
       h) GS fallback (for remaining TAS gaps, assume minimal wind)
    
    3. NO final filtering - preserve all filled values
    
    Args:
        dataset: Dataset name ('train', 'rank', 'final')
        dirs: Dictionary of output directories
        debug_mode: Enable detailed logging
        max_flights: Maximum flights to process (for testing)
        skip_existing: Skip already-processed flights
        workers: Number of parallel workers (default: 1)
    
    Returns:
        (success_count, failed_flights_list)
    """
    from tqdm import tqdm
    from multiprocessing import Pool
    
    logger.info("\n" + "="*80)
    logger.info("STAGE 3: Fill TAS/CAS with Physics-Based Methods + Filtering")
    logger.info("="*80)
    
    # Input from Stage 2, Output to Stage 3
    input_dir = STAGE_2_OUTPUT / dataset
    output_dir = STAGE_3_OUTPUT / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trajectory_files = list(input_dir.glob("*.parquet"))
    if debug_mode and max_flights:
        trajectory_files = trajectory_files[:max_flights]
    
    logger.info(f"Processing {len(trajectory_files)} flights")
    logger.info(f"Priority: ACARS → Mach+Temp → CAS+Density → GS+Wind → Interpolation → Taxi → GS Calc → GS Fallback")
    logger.info(f"Filtering: Global caps + Phase limits + Consistency checks")
    logger.info(f"Workers: {workers}")
    
    success = 0
    failed = []
    skipped = 0
    
    # Prepare tasks
    tasks = []
    for traj_file in trajectory_files:
        output_file = output_dir / f"{traj_file.stem}.parquet"
        
        if skip_existing and output_file.exists():
            skipped += 1
            success += 1
            continue
        
        tasks.append((traj_file, output_file, debug_mode))
    
    # Process flights
    if workers > 1 and len(tasks) > 0:
        # Parallel processing
        logger.info(f"Processing {len(tasks)} flights with {workers} workers...")
        with Pool(workers) as pool:
            results = list(tqdm(
                pool.imap(process_flight_wrapper, tasks),
                total=len(tasks),
                desc="TAS/CAS Filling"
            ))
        
        # Collect results
        for status, flight_id, error in results:
            if status == 'success':
                success += 1
            else:
                failed.append(flight_id)
                if error:
                    logger.warning(f"Failed {flight_id}: {error}")
    else:
        # Sequential processing
        for task in tqdm(tasks, desc="TAS/CAS Filling"):
            traj_file, output_file, debug = task
            status, flight_id, error = process_single_flight(traj_file, output_file, debug)
            
            if status == 'success':
                success += 1
            else:
                failed.append(flight_id)
                if error:
                    logger.warning(f"Failed {flight_id}: {error}")
    
    logger.info(f"\nStage 4 Complete:")
    logger.info(f"  Success: {success}/{len(trajectory_files)}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Failed: {len(failed)}")
    
    return success, failed


if __name__ == '__main__':
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Stage 3: Fill TAS/CAS with Physics-Based Methods')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['train', 'rank', 'final', 'all'],
                       help='Dataset to process (train, rank, final, or all)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed logging')
    parser.add_argument('--max-flights', type=int, default=None,
                       help='Maximum number of flights to process (for testing)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Reprocess all flights (don\'t skip existing)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    
    args = parser.parse_args()
    
    # Determine which datasets to process
    if args.dataset == 'all':
        datasets = ['train', 'rank', 'final']
    else:
        datasets = [args.dataset]
    
    # Process each dataset
    for dataset in datasets:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# Processing dataset: {dataset.upper()}")
        logger.info(f"{'#'*80}\n")
        
        success, failed = stage4_fill_tas(
            dataset=dataset,
            dirs={},  # Not used in current implementation
            debug_mode=args.debug,
            max_flights=args.max_flights,
            skip_existing=not args.no_skip,
            workers=args.workers
        )
        
        if failed:
            logger.warning(f"\nFailed flights for {dataset}:")
            for flight_id in failed:
                logger.warning(f"  - {flight_id}")
    
    logger.info("\n" + "="*80)
    logger.info("ALL DATASETS COMPLETE")
    logger.info("="*80)
