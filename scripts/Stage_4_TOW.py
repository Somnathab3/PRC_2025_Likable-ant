"""
Enhanced TOW Prediction Script - Version 2
Uses Updated 2024 Dataset + Slice-Based Feature Engineering

Key Improvements:
1. Uses Updated_2024_dataset_with_features.csv (610 new features!)
2. Extracts slice-based features from trajectories (matching 2024 dataset)
3. Saves extracted features for reuse (no reprocessing needed)
4. Excludes mass/energy/thrust features (require TOW as input)
5. Focuses on kinematics: altitude, speed, time, performance metrics

Data Flow:
  2024 Training Data (with slice features)
         ↓
  Train Model
         ↓
  Extract Slice Features from New Trajectories → Save to Disk
         ↓
  Predict TOW for New Flights

Output:
- data/processed/tow_features/train/*.parquet (cached features)
- data/processed/tow_features/rank/*.parquet (cached features)
- data/processed/tow_features/final/*.parquet (cached features)
- data/processed/tow_predictions/tow_predictions_train_v2.csv
- data/processed/tow_predictions/tow_predictions_rank_v2.csv
- data/processed/tow_predictions/tow_predictions_final_v2.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
from math import radians, sin, cos, sqrt, atan2
warnings.filterwarnings('ignore')

# Add the Tow_calculation directory to path for aircraft database loaders
sys.path.insert(0, str(Path(__file__).parent / "Backup" / "Tow_calculation"))

# Paths
WORKSPACE_ROOT = Path(__file__).parent.parent
OLD_DATA_PATH = WORKSPACE_ROOT / "data" / "raw" / "tow_prc2024" / "Updated_2024_dataset_with_features.csv"

# Pipeline stage directories
STAGE_3_OUTPUT = WORKSPACE_ROOT / "data" / "processed" / "trajectories_tas_filled"
FLIGHTS_TRAIN_DIR = STAGE_3_OUTPUT / "train"
FLIGHTS_RANK_DIR = STAGE_3_OUTPUT / "rank"
FLIGHTS_FINAL_DIR = STAGE_3_OUTPUT / "final"
FLIGHTLIST_TRAIN = WORKSPACE_ROOT / "data" / "raw" / "flightlist_train.parquet"
FLIGHTLIST_RANK = WORKSPACE_ROOT / "data" / "raw" / "flightlist_rank.parquet"
FLIGHTLIST_FINAL = WORKSPACE_ROOT / "data" / "raw" / "flightlist_final.parquet"
APT_DATA = WORKSPACE_ROOT / "data" / "raw" / "apt.parquet"
OPENAP_DATA_DIR = WORKSPACE_ROOT / "openap" / "openap" / "data"

# Output directories
FEATURES_OUTPUT_DIR = WORKSPACE_ROOT / "data" / "processed" / "tow_features"
PREDICTIONS_OUTPUT_DIR = WORKSPACE_ROOT / "data" / "processed" / "tow_predictions"

# Altitude slices (in feet) - matching 2024 dataset
ALTITUDE_SLICES = [
    # Slice features removed as requested
]

# Percentage slices
PERCENTAGE_SLICES = [
    # Slice features removed as requested
]


def load_openap_aircraft_data():
    """Load comprehensive aircraft database from OpenAP + FAA"""
    print("\nLoading comprehensive aircraft database...")
    from load_aircraft_database import load_comprehensive_aircraft_data
    aircraft_df = load_comprehensive_aircraft_data(workspace_root=WORKSPACE_ROOT)
    print(f"✓ Loaded {len(aircraft_df)} aircraft types")
    print(f"✓ Features: {len([c for c in aircraft_df.columns if c.startswith('ADB2_')])} ADB2 features")
    print(f"✓ Features: {len([c for c in aircraft_df.columns if c.startswith('openap_')])} OpenAP features")
    return aircraft_df


def calculate_great_circle_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in nautical miles"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance_km = 6371 * c
    distance_nm = distance_km * 0.539957
    return distance_nm


def extract_slice_based_features(df, flight_info):
    """
    Extract comprehensive slice-based features from trajectory data
    Matches features from Updated_2024_dataset_with_features.csv
    Excludes: mass, energy_rate, delta_T (require TOW)
    """
    features = {
        'flight_id': flight_info['flight_id'],
        'aircraft_type': flight_info['aircraft_type'],
        'origin_icao': flight_info['origin_icao'],
        'destination_icao': flight_info['destination_icao'],
    }
    
    if len(df) == 0:
        return features
    
    df = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate time metrics
    if 'timestamp' in df.columns and len(df) > 1:
        df['time_since_departure'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        df['time_until_arrival'] = (df['timestamp'].iloc[-1] - df['timestamp']).dt.total_seconds()
        features['total_flight_duration_sec'] = df['time_since_departure'].iloc[-1]
    
    # ========================================================================
    # ALTITUDE SLICE FEATURES - DISABLED
    # ========================================================================
    for alt_min, alt_max in ALTITUDE_SLICES:
        slice_name = f"slice_{alt_min}_{alt_max}"
        
        # Filter data in altitude slice
        if 'altitude' in df.columns:
            slice_df = df[(df['altitude'] >= alt_min) & (df['altitude'] < alt_max)]
            
            if len(slice_df) > 0:
                # Number of points
                features[f'{slice_name}_num_points'] = len(slice_df)
                
                # Time metrics
                if 'time_since_departure' in slice_df.columns:
                    features[f'{slice_name}_min_time_since_departure'] = slice_df['time_since_departure'].min()
                if 'time_until_arrival' in slice_df.columns:
                    features[f'{slice_name}_max_time_until_arrival'] = slice_df['time_until_arrival'].max()
                
                # ROCD (Rate of Climb/Descent) - kinematics only
                if 'vertical_rate' in slice_df.columns:
                    vr = slice_df['vertical_rate'].dropna()
                    if len(vr) > 0:
                        features[f'{slice_name}_median_rocd'] = vr.median()
                        features[f'{slice_name}_rocd_range'] = vr.max() - vr.min()
                        features[f'{slice_name}_mean_rocd'] = vr.mean()
                        features[f'{slice_name}_std_rocd'] = vr.std()
                
                # Speed metrics
                if 'groundspeed' in slice_df.columns:
                    gs = slice_df['groundspeed'].dropna()
                    if len(gs) > 0:
                        features[f'{slice_name}_median_groundspeed'] = gs.median()
                        features[f'{slice_name}_mean_groundspeed'] = gs.mean()
                        features[f'{slice_name}_max_groundspeed'] = gs.max()
                        features[f'{slice_name}_min_groundspeed'] = gs.min()
                
                if 'TAS' in slice_df.columns:
                    tas = slice_df['TAS'].dropna()
                    if len(tas) > 0:
                        features[f'{slice_name}_median_tas'] = tas.median()
                        features[f'{slice_name}_mean_tas'] = tas.mean()
                
                if 'CAS' in slice_df.columns:
                    cas = slice_df['CAS'].dropna()
                    if len(cas) > 0:
                        features[f'{slice_name}_median_cas'] = cas.median()
                
                if 'mach' in slice_df.columns:
                    mach = slice_df['mach'].dropna()
                    if len(mach) > 0:
                        features[f'{slice_name}_median_mach'] = mach.median()
                
                # Altitude stats
                if 'altitude' in slice_df.columns:
                    alt = slice_df['altitude'].dropna()
                    if len(alt) > 0:
                        features[f'{slice_name}_median_altitude'] = alt.median()
                        features[f'{slice_name}_mean_altitude'] = alt.mean()
                        features[f'{slice_name}_std_altitude'] = alt.std()
                
                # Wind components (ERA5)
                if 'u_component_of_wind_pl' in slice_df.columns and 'v_component_of_wind_pl' in slice_df.columns:
                    u_wind = slice_df['u_component_of_wind_pl'].dropna()
                    v_wind = slice_df['v_component_of_wind_pl'].dropna()
                    if len(u_wind) > 0 and len(v_wind) > 0:
                        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
                        features[f'{slice_name}_median_wind_speed'] = wind_speed.median()
                        features[f'{slice_name}_mean_u_wind'] = u_wind.mean()
                        features[f'{slice_name}_mean_v_wind'] = v_wind.mean()
                
                # Temperature (ERA5)
                if 'temperature_pl' in slice_df.columns:
                    temp = slice_df['temperature_pl'].dropna()
                    if len(temp) > 0:
                        features[f'{slice_name}_median_temperature'] = temp.median()
                
                # Pressure (ERA5)
                if 'pressure_hpa' in slice_df.columns:
                    pressure = slice_df['pressure_hpa'].dropna()
                    if len(pressure) > 0:
                        features[f'{slice_name}_median_pressure'] = pressure.median()
    
    # ========================================================================
    # PERCENTAGE SLICE FEATURES
    # ========================================================================
    if len(df) > 0:
        for pct_min, pct_max in PERCENTAGE_SLICES:
            slice_name = f"slice_{pct_min}_{pct_max}"
            
            # Calculate indices for percentage slice
            idx_min = int(len(df) * pct_min / 100)
            idx_max = int(len(df) * pct_max / 100)
            
            if idx_max > idx_min:
                slice_df = df.iloc[idx_min:idx_max]
                
                # Number of points
                features[f'{slice_name}_num_points'] = len(slice_df)
                
                # Median altitude
                if 'altitude' in slice_df.columns:
                    alt = slice_df['altitude'].dropna()
                    if len(alt) > 0:
                        features[f'{slice_name}_median_altitude'] = alt.median()
    
    # ========================================================================
    # PHASE-BASED FEATURES (Enhanced)
    # ========================================================================
    if 'vertical_rate' in df.columns and 'altitude' in df.columns:
        df['phase'] = 'cruise'
        
        # Climb phase
        climb_mask = (df['vertical_rate'] > 300)
        df.loc[climb_mask, 'phase'] = 'climb'
        
        # Descent phase
        descent_mask = (df['vertical_rate'] < -300)
        df.loc[descent_mask, 'phase'] = 'descent'
        
        # Takeoff phase
        if 'timestamp' in df.columns and len(df) > 0:
            time_from_start = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
            takeoff_mask = (time_from_start < 120) | ((df['altitude'] < 1500) & (df['vertical_rate'] > 500))
            df.loc[takeoff_mask, 'phase'] = 'takeoff'
        
        # Landing phase
        if 'timestamp' in df.columns and len(df) > 0:
            time_to_end = (df['timestamp'].max() - df['timestamp']).dt.total_seconds()
            landing_mask = (time_to_end < 120) | ((df['altitude'] < 1500) & (df['vertical_rate'] < -300))
            df.loc[landing_mask, 'phase'] = 'landing'
        
        # Extract phase statistics
        for phase in ['climb', 'cruise', 'descent', 'takeoff', 'landing']:
            phase_df = df[df['phase'] == phase]
            
            if len(phase_df) > 0:
                # Time in phase
                features[f'{phase}_time_sec'] = len(phase_df) * 0.5  # Assuming 0.5s intervals
                
                # Distance in phase
                if 'latitude' in phase_df.columns and 'longitude' in phase_df.columns:
                    coords = phase_df[['latitude', 'longitude']].dropna()
                    if len(coords) > 1:
                        dist = 0
                        for i in range(len(coords) - 1):
                            lat1, lon1 = radians(coords.iloc[i]['latitude']), radians(coords.iloc[i]['longitude'])
                            lat2, lon2 = radians(coords.iloc[i+1]['latitude']), radians(coords.iloc[i+1]['longitude'])
                            dlat, dlon = lat2 - lat1, lon2 - lon1
                            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                            c = 2 * atan2(sqrt(a), sqrt(1-a))
                            dist += 6371 * c
                        features[f'{phase}_distance_nm'] = dist * 0.539957
                
                # Speed statistics
                if 'groundspeed' in phase_df.columns:
                    gs = phase_df['groundspeed'].dropna()
                    if len(gs) > 0:
                        features[f'{phase}_avg_speed'] = gs.mean()
                        features[f'{phase}_max_speed'] = gs.max()
                        features[f'{phase}_median_speed'] = gs.median()
                
                if 'TAS' in phase_df.columns:
                    tas = phase_df['TAS'].dropna()
                    if len(tas) > 0:
                        features[f'{phase}_avg_tas'] = tas.mean()
                        features[f'{phase}_median_tas'] = tas.median()
                
                # Altitude statistics
                if 'altitude' in phase_df.columns:
                    alt = phase_df['altitude'].dropna()
                    if len(alt) > 0:
                        features[f'{phase}_avg_altitude'] = alt.mean()
                        features[f'{phase}_max_altitude'] = alt.max()
                        features[f'{phase}_median_altitude'] = alt.median()
                        features[f'{phase}_std_altitude'] = alt.std()
                
                # Vertical rate statistics
                if 'vertical_rate' in phase_df.columns:
                    vr = phase_df['vertical_rate'].dropna()
                    if len(vr) > 0:
                        features[f'{phase}_avg_vr'] = vr.mean()
                        features[f'{phase}_median_vr'] = vr.median()
                
                # Acceleration (kinematics)
                if 'groundspeed' in phase_df.columns and len(phase_df) > 1:
                    gs_diff = phase_df['groundspeed'].diff().dropna()
                    if len(gs_diff) > 0:
                        features[f'{phase}_accel'] = gs_diff.mean() * 2  # Convert to per second
        
        # Special takeoff/landing speeds
        takeoff_df = df[df['phase'] == 'takeoff']
        if len(takeoff_df) > 0 and 'groundspeed' in takeoff_df.columns:
            gs = takeoff_df['groundspeed'].dropna()
            if len(gs) > 0:
                features['v2_speed'] = gs.iloc[-1] if len(gs) > 0 else gs.max()
        
        landing_df = df[df['phase'] == 'landing']
        if len(landing_df) > 0 and 'groundspeed' in landing_df.columns:
            gs = landing_df['groundspeed'].dropna()
            if len(gs) > 0:
                features['landing_speed'] = gs.iloc[0] if len(gs) > 0 else gs.mean()
    
    # ========================================================================
    # OVERALL TRAJECTORY FEATURES
    # ========================================================================
    # Altitude features
    if 'altitude' in df.columns:
        alt = df['altitude'].dropna()
        if len(alt) > 0:
            features['avg_altitude'] = alt.mean()
            features['max_altitude'] = alt.max()
            features['min_altitude'] = alt.min()
            features['altitude_std'] = alt.std()
            features['median_altitude'] = alt.median()
    
    # Speed features
    if 'groundspeed' in df.columns:
        gs = df['groundspeed'].dropna()
        if len(gs) > 0:
            features['avg_groundspeed'] = gs.mean()
            features['max_groundspeed'] = gs.max()
            features['groundspeed_std'] = gs.std()
            features['median_groundspeed'] = gs.median()
    
    if 'TAS' in df.columns:
        tas = df['TAS'].dropna()
        if len(tas) > 0:
            features['avg_tas'] = tas.mean()
            features['max_tas'] = tas.max()
            features['tas_std'] = tas.std()
            features['median_tas'] = tas.median()
    
    if 'CAS' in df.columns:
        cas = df['CAS'].dropna()
        if len(cas) > 0:
            features['avg_cas'] = cas.mean()
            features['max_cas'] = cas.max()
            features['median_cas'] = cas.median()
    
    if 'mach' in df.columns:
        mach = df['mach'].dropna()
        if len(mach) > 0:
            features['avg_mach'] = mach.mean()
            features['max_mach'] = mach.max()
            features['median_mach'] = mach.median()
    
    # Vertical rate
    if 'vertical_rate' in df.columns:
        vr = df['vertical_rate'].dropna()
        if len(vr) > 0:
            features['avg_vertical_rate'] = vr.mean()
            features['max_vertical_rate'] = vr.max()
            features['vertical_rate_std'] = vr.std()
            features['median_vertical_rate'] = vr.median()
    
    # Wind (ERA5)
    if 'u_component_of_wind_pl' in df.columns and 'v_component_of_wind_pl' in df.columns:
        u_wind = df['u_component_of_wind_pl'].dropna()
        v_wind = df['v_component_of_wind_pl'].dropna()
        if len(u_wind) > 0 and len(v_wind) > 0:
            wind_speed = np.sqrt(u_wind**2 + v_wind**2)
            features['avg_wind_speed'] = wind_speed.mean()
            features['max_wind_speed'] = wind_speed.max()
            features['avg_u_wind'] = u_wind.mean()
            features['avg_v_wind'] = v_wind.mean()
    
    # Temperature (ERA5)
    if 'temperature_pl' in df.columns:
        temp = df['temperature_pl'].dropna()
        if len(temp) > 0:
            features['avg_temperature_k'] = temp.mean()
            features['median_temperature_k'] = temp.median()
    
    # Pressure (ERA5)
    if 'pressure_hpa' in df.columns:
        pressure = df['pressure_hpa'].dropna()
        if len(pressure) > 0:
            features['avg_pressure'] = pressure.mean()
            features['median_pressure'] = pressure.median()
    
    # Position count
    features['num_positions'] = len(df)
    
    # Distance estimation
    if 'latitude' in df.columns and 'longitude' in df.columns:
        coords = df[['latitude', 'longitude']].dropna()
        if len(coords) > 1:
            total_dist = 0
            for i in range(len(coords) - 1):
                lat1, lon1 = radians(coords.iloc[i]['latitude']), radians(coords.iloc[i]['longitude'])
                lat2, lon2 = radians(coords.iloc[i+1]['latitude']), radians(coords.iloc[i+1]['longitude'])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                total_dist += 6371 * c
            features['total_distance_km'] = total_dist
            features['total_distance_nm'] = total_dist * 0.539957
    
    # Track change
    if 'track' in df.columns:
        track = df['track'].dropna()
        if len(track) > 1:
            track_diff = np.abs(np.diff(track))
            track_diff = np.minimum(track_diff, 360 - track_diff)
            features['avg_track_change'] = track_diff.mean()
    
    return features


def process_and_save_flight_features(row_data, flights_dir, output_dir):
    """Process a single flight and save features to disk"""
    flight_id, aircraft_type, origin_icao, destination_icao, takeoff, landed = row_data
    trajectory_file = flights_dir / f"{flight_id}.parquet"
    feature_file = output_dir / f"{flight_id}.parquet"
    
    # Check if features already exist
    if feature_file.exists():
        try:
            features_df = pd.read_parquet(feature_file)
            return features_df.to_dict('records')[0]
        except:
            pass  # Recompute if file is corrupted
    
    if not trajectory_file.exists():
        return None
    
    try:
        traj_df = pd.read_parquet(trajectory_file)
        flight_info = {
            'flight_id': flight_id,
            'aircraft_type': aircraft_type,
            'origin_icao': origin_icao,
            'destination_icao': destination_icao,
        }
        
        # Add flight duration
        if pd.notna(takeoff) and pd.notna(landed):
            flight_info['flight_duration_sec'] = (landed - takeoff).total_seconds()
        
        features = extract_slice_based_features(traj_df, flight_info)
        
        # Save features to disk
        output_dir.mkdir(parents=True, exist_ok=True)
        features_df = pd.DataFrame([features])
        features_df.to_parquet(feature_file, index=False)
        
        return features
    except Exception as e:
        print(f"Error processing flight {flight_id}: {e}")
        return None


def add_derived_features(df):
    """Add derived features from base features"""
    # Phase ratios
    if 'climb_time_sec' in df.columns and 'total_flight_duration_sec' in df.columns:
        df['climb_time_ratio'] = df['climb_time_sec'] / np.maximum(df['total_flight_duration_sec'], 1)
    
    if 'cruise_time_sec' in df.columns and 'total_flight_duration_sec' in df.columns:
        df['cruise_time_ratio'] = df['cruise_time_sec'] / np.maximum(df['total_flight_duration_sec'], 1)
    
    if 'descent_time_sec' in df.columns and 'total_flight_duration_sec' in df.columns:
        df['descent_time_ratio'] = df['descent_time_sec'] / np.maximum(df['total_flight_duration_sec'], 1)
    
    # Speed ratios
    if 'avg_tas' in df.columns and 'avg_cas' in df.columns:
        df['tas_cas_ratio'] = df['avg_tas'] / np.maximum(df['avg_cas'], 1)
    
    if 'avg_groundspeed' in df.columns and 'avg_tas' in df.columns:
        df['wind_effect'] = df['avg_groundspeed'] - df['avg_tas']
    
    # Climb performance (kinematics only)
    if 'climb_avg_vr' in df.columns and 'climb_max_altitude' in df.columns:
        df['climb_performance'] = df['climb_avg_vr'] * df['climb_max_altitude'] / 100000
    
    # Route efficiency
    if 'total_distance_nm' in df.columns and 'great_circle_distance_nm' in df.columns:
        df['route_inefficiency'] = (df['total_distance_nm'] / np.maximum(df['great_circle_distance_nm'], 1)) - 1
    
    # Altitude-based features
    if 'adep_elevation_ft' in df.columns and 'ades_elevation_ft' in df.columns:
        df['elevation_diff'] = df['ades_elevation_ft'] - df['adep_elevation_ft']
    
    return df


def main():
    print("="*80)
    print("TOW PREDICTION - VERSION 2 (SLICE-BASED FEATURES)")
    print("="*80)
    
    # Load external data
    print("\nLoading external data...")
    apt_df = pd.read_parquet(APT_DATA)
    print(f"✓ Loaded {len(apt_df)} airports")
    
    openap_aircraft_df = load_openap_aircraft_data()
    
    print("\n" + "="*80)
    print("STEP 1: Loading Updated 2024 Dataset")
    print("="*80)
    
    print(f"Reading: {OLD_DATA_PATH.name}")
    print("This may take a while (848 columns, 295K rows)...")
    
    try:
        # Load with low_memory=False for mixed types
        old_df = pd.read_csv(OLD_DATA_PATH, low_memory=False)
    except:
        old_df = pd.read_csv(OLD_DATA_PATH, engine='python', on_bad_lines='skip')
    
    print(f"✓ Loaded shape: {old_df.shape}")
    print(f"✓ Columns: {len(old_df.columns)}")
    
    # Check for TOW column
    if 'tow' not in old_df.columns:
        print("\n⚠️  WARNING: 'tow' column not found in dataset!")
        print(f"Available columns: {list(old_df.columns[:20])}...")
        return
    
    print(f"✓ TOW data: {old_df['tow'].notna().sum()} flights")
    
    # Select features for training (exclude mass/energy/thrust features)
    print("\nSelecting features (excluding mass/energy/thrust)...")
    
    # Get all slice-based features
    slice_features = []
    for col in old_df.columns:
        if col.startswith('slice_'):
            # Exclude mass, energy, and delta_T features
            if not any(x in col for x in ['_mass', '_energy', '_delta_T']):
                slice_features.append(col)
    
    print(f"✓ Found {len(slice_features)} slice-based features (kinematics only)")
    
    # Add other important features
    base_features = [
        'flight_id', 'aircraft_type', 'adep', 'ades',
        'total_flight_duration_sec'
    ]
    
    # ADB2 features (aircraft performance)
    adb2_features = [col for col in old_df.columns if col.startswith('ADB2_')]
    print(f"✓ Found {len(adb2_features)} ADB2 features")
    
    # Combine all features
    all_features = base_features + slice_features + adb2_features + ['tow']
    available_features = [f for f in all_features if f in old_df.columns]
    
    print(f"✓ Total available features: {len(available_features)}")
    
    # Create modeling dataset
    old_modeling = old_df[available_features].copy()
    
    # Add airport data
    old_modeling = old_modeling.merge(
        apt_df[['icao', 'elevation', 'latitude', 'longitude']],
        left_on='adep',
        right_on='icao',
        how='left'
    ).rename(columns={'elevation': 'adep_elevation_ft', 'latitude': 'adep_lat', 'longitude': 'adep_lon'}).drop(columns=['icao'])
    
    old_modeling = old_modeling.merge(
        apt_df[['icao', 'elevation', 'latitude', 'longitude']],
        left_on='ades',
        right_on='icao',
        how='left'
    ).rename(columns={'elevation': 'ades_elevation_ft', 'latitude': 'ades_lat', 'longitude': 'ades_lon'}).drop(columns=['icao'])
    
    # Calculate great circle distance
    if all(col in old_modeling.columns for col in ['adep_lat', 'adep_lon', 'ades_lat', 'ades_lon']):
        old_modeling['great_circle_distance_nm'] = old_modeling.apply(
            lambda row: calculate_great_circle_distance(row['adep_lat'], row['adep_lon'], row['ades_lat'], row['ades_lon'])
            if pd.notna(row['adep_lat']) and pd.notna(row['adep_lon']) and pd.notna(row['ades_lat']) and pd.notna(row['ades_lon'])
            else np.nan,
            axis=1
        )
    
    # Add OpenAP data
    old_modeling['aircraft_code'] = old_modeling['aircraft_type'].str.lower()
    old_modeling = old_modeling.merge(
        openap_aircraft_df,
        on='aircraft_code',
        how='left'
    ).drop(columns=['aircraft_code'])
    
    # Add derived features
    old_modeling = add_derived_features(old_modeling)
    
    print(f"✓ Final dataset shape: {old_modeling.shape}")
    
    # Drop rows with missing TOW
    old_modeling = old_modeling.dropna(subset=['tow'])
    print(f"✓ After dropping missing TOW: {old_modeling.shape}")
    
    print("\n" + "="*80)
    print("STEP 2: Extracting Features from New Trajectories")
    print("="*80)
    
    # Prepare output directories
    train_features_dir = FEATURES_OUTPUT_DIR / "train"
    rank_features_dir = FEATURES_OUTPUT_DIR / "rank"
    final_features_dir = FEATURES_OUTPUT_DIR / "final"
    
    train_features_dir.mkdir(parents=True, exist_ok=True)
    rank_features_dir.mkdir(parents=True, exist_ok=True)
    final_features_dir.mkdir(parents=True, exist_ok=True)
    
    num_cores = max(1, cpu_count() - 1)
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Process training flights (uses cache if features already exist)
    print("\n▶ Processing training flights...")
    flightlist_train = pd.read_parquet(FLIGHTLIST_TRAIN)
    print(f"Total flights: {len(flightlist_train)}")
    
    # Check cache
    cached_count = sum(1 for _, row in flightlist_train.iterrows() 
                      if (train_features_dir / f"{row['flight_id']}.parquet").exists())
    print(f"Found {cached_count} cached features, will process {len(flightlist_train) - cached_count} new flights")
    
    train_data = [
        (row['flight_id'], row['aircraft_type'], row['origin_icao'], 
         row['destination_icao'], row['takeoff'], row['landed'])
        for _, row in flightlist_train.iterrows()
    ]
    
    process_func = partial(process_and_save_flight_features, 
                          flights_dir=FLIGHTS_TRAIN_DIR, 
                          output_dir=train_features_dir)
    
    with Pool(num_cores) as pool:
        train_features = pool.map(process_func, train_data)
    
    train_features = [f for f in train_features if f is not None]
    train_features_df = pd.DataFrame(train_features)
    print(f"✓ Loaded features for {len(train_features_df)} flights")
    print(f"✓ Features cached in: {train_features_dir}")
    
    # Add airport and aircraft data
    train_features_df = train_features_df.merge(
        apt_df[['icao', 'elevation', 'latitude', 'longitude']],
        left_on='origin_icao',
        right_on='icao',
        how='left'
    ).rename(columns={'elevation': 'adep_elevation_ft', 'latitude': 'adep_lat', 'longitude': 'adep_lon'}).drop(columns=['icao'])
    
    train_features_df = train_features_df.merge(
        apt_df[['icao', 'elevation', 'latitude', 'longitude']],
        left_on='destination_icao',
        right_on='icao',
        how='left'
    ).rename(columns={'elevation': 'ades_elevation_ft', 'latitude': 'ades_lat', 'longitude': 'ades_lon'}).drop(columns=['icao'])
    
    if all(col in train_features_df.columns for col in ['adep_lat', 'adep_lon', 'ades_lat', 'ades_lon']):
        train_features_df['great_circle_distance_nm'] = train_features_df.apply(
            lambda row: calculate_great_circle_distance(row['adep_lat'], row['adep_lon'], row['ades_lat'], row['ades_lon'])
            if pd.notna(row['adep_lat']) and pd.notna(row['adep_lon']) and pd.notna(row['ades_lat']) and pd.notna(row['ades_lon'])
            else np.nan,
            axis=1
        )
    
    train_features_df['aircraft_code'] = train_features_df['aircraft_type'].str.lower()
    train_features_df = train_features_df.merge(
        openap_aircraft_df,
        on='aircraft_code',
        how='left'
    ).drop(columns=['aircraft_code'])
    
    train_features_df = add_derived_features(train_features_df)
    
    # Process ranking flights (uses cache if features already exist)
    print("\n▶ Processing ranking flights...")
    flightlist_rank = pd.read_parquet(FLIGHTLIST_RANK)
    print(f"Total flights: {len(flightlist_rank)}")
    
    # Check cache
    cached_count = sum(1 for _, row in flightlist_rank.iterrows() 
                      if (rank_features_dir / f"{row['flight_id']}.parquet").exists())
    print(f"Found {cached_count} cached features, will process {len(flightlist_rank) - cached_count} new flights")
    
    rank_data = [
        (row['flight_id'], row['aircraft_type'], row['origin_icao'], 
         row['destination_icao'], row['takeoff'], row['landed'])
        for _, row in flightlist_rank.iterrows()
    ]
    
    process_func = partial(process_and_save_flight_features, 
                          flights_dir=FLIGHTS_RANK_DIR, 
                          output_dir=rank_features_dir)
    
    with Pool(num_cores) as pool:
        rank_features = pool.map(process_func, rank_data)
    
    rank_features = [f for f in rank_features if f is not None]
    rank_features_df = pd.DataFrame(rank_features)
    print(f"✓ Loaded features for {len(rank_features_df)} flights")
    print(f"✓ Features cached in: {rank_features_dir}")
    
    # Add airport and aircraft data
    rank_features_df = rank_features_df.merge(
        apt_df[['icao', 'elevation', 'latitude', 'longitude']],
        left_on='origin_icao',
        right_on='icao',
        how='left'
    ).rename(columns={'elevation': 'adep_elevation_ft', 'latitude': 'adep_lat', 'longitude': 'adep_lon'}).drop(columns=['icao'])
    
    rank_features_df = rank_features_df.merge(
        apt_df[['icao', 'elevation', 'latitude', 'longitude']],
        left_on='destination_icao',
        right_on='icao',
        how='left'
    ).rename(columns={'elevation': 'ades_elevation_ft', 'latitude': 'ades_lat', 'longitude': 'ades_lon'}).drop(columns=['icao'])
    
    if all(col in rank_features_df.columns for col in ['adep_lat', 'adep_lon', 'ades_lat', 'ades_lon']):
        rank_features_df['great_circle_distance_nm'] = rank_features_df.apply(
            lambda row: calculate_great_circle_distance(row['adep_lat'], row['adep_lon'], row['ades_lat'], row['ades_lon'])
            if pd.notna(row['adep_lat']) and pd.notna(row['adep_lon']) and pd.notna(row['ades_lat']) and pd.notna(row['ades_lon'])
            else np.nan,
            axis=1
        )
    
    rank_features_df['aircraft_code'] = rank_features_df['aircraft_type'].str.lower()
    rank_features_df = rank_features_df.merge(
        openap_aircraft_df,
        on='aircraft_code',
        how='left'
    ).drop(columns=['aircraft_code'])
    
    rank_features_df = add_derived_features(rank_features_df)
    
    print("\n" + "="*80)
    print("STEP 3: Training TOW Prediction Model")
    print("="*80)
    
    # Get features that are common between 2024 dataset and extracted trajectory features
    # Use train_features_df as reference since it has the features we can actually extract
    print(f"\n2024 dataset columns: {len(old_modeling.columns)}")
    print(f"Train features columns: {len(train_features_df.columns)}")
    
    # Find common features (exclude ID columns)
    exclude_cols = ['flight_id', 'adep', 'ades', 'tow', 'origin_icao', 'destination_icao', 'aircraft_code']
    old_feature_cols = [col for col in old_modeling.columns if col not in exclude_cols]
    new_feature_cols = [col for col in train_features_df.columns if col not in exclude_cols]
    
    # Only use features that exist in BOTH datasets
    available_features = [f for f in old_feature_cols if f in new_feature_cols]
    
    print(f"Common features between datasets: {len(available_features)}")
    print(f"Sample features: {available_features[:10]}")
    
    # Prepare training data
    X_train_full = old_modeling[available_features].copy()
    y_train_full = old_modeling['tow'].values
    
    # Handle categorical features - keep only aircraft_type, drop other object columns
    cat_features = ['aircraft_type'] if 'aircraft_type' in X_train_full.columns else []
    
    # Drop all other object dtype columns (string columns that LightGBM can't handle)
    object_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == 'object' and col not in cat_features]
    if object_cols:
        print(f"Dropping {len(object_cols)} object columns that LightGBM can't handle: {object_cols[:5]}...")
        X_train_full = X_train_full.drop(columns=object_cols)
    
    # Convert categorical features
    for cat in cat_features:
        if cat in X_train_full.columns:
            X_train_full[cat] = X_train_full[cat].astype('category')
    
    # Fill missing values (only numeric columns now)
    for col in X_train_full.columns:
        if col not in cat_features:
            X_train_full[col].fillna(X_train_full[col].median(), inplace=True)
    
    # Update available features after dropping object columns
    available_features = list(X_train_full.columns)
    print(f"Final features after cleanup: {len(available_features)}")
    
    print(f"\n⚠️  Note: Training on {len(X_train_full)} samples may take a while...")
    print(f"Consider reducing sample size for faster testing if needed.\n")
    
    # Split for validation - use a smaller sample for faster testing
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Train LightGBM
    print("\nTraining LightGBM model...")
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    # Validation performance
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
    
    print(f"\n{'='*80}")
    print(f"VALIDATION PERFORMANCE")
    print(f"{'='*80}")
    print(f"  MAE:  {val_mae:.2f} kg")
    print(f"  R²:   {val_r2:.4f}")
    print(f"  MAPE: {val_mape:.2f}%")
    
    # Feature importance
    print(f"\n{'='*80}")
    print(f"TOP 20 MOST IMPORTANT FEATURES")
    print(f"{'='*80}")
    importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    print(importance.head(20).to_string(index=False))
    
    # Save feature importance
    importance_path = PREDICTIONS_OUTPUT_DIR / "feature_importance_tow_v2.csv"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(importance_path, index=False)
    print(f"\n✓ Saved feature importance to: {importance_path}")
    
    # Save trained model
    model_path = PREDICTIONS_OUTPUT_DIR / "tow_model_v2.txt"
    model.save_model(str(model_path))
    print(f"✓ Saved trained model to: {model_path}")
    
    # Save available features list for reference
    features_list_path = PREDICTIONS_OUTPUT_DIR / "model_features_v2.txt"
    with open(features_list_path, 'w') as f:
        f.write("\n".join(available_features))
    print(f"✓ Saved feature list to: {features_list_path}")
    
    print("\n" + "="*80)
    print("STEP 4: Generating TOW Predictions")
    print("="*80)
    
    # Predict for training set
    print("\n▶ Predicting TOW for training flights...")
    X_train_new = train_features_df[available_features].copy()
    
    # Drop object columns
    object_cols_train = [col for col in X_train_new.columns if X_train_new[col].dtype == 'object' and col not in cat_features]
    if object_cols_train:
        print(f"Dropping {len(object_cols_train)} object columns from train predictions")
        X_train_new = X_train_new.drop(columns=object_cols_train)
    
    for cat in cat_features:
        if cat in X_train_new.columns:
            X_train_new[cat] = X_train_new[cat].astype('category')
    
    for col in X_train_new.columns:
        if col not in cat_features:
            X_train_new[col].fillna(X_train_new[col].median() if X_train_new[col].notna().sum() > 0 else 0, inplace=True)
    
    tow_train_pred = model.predict(X_train_new, num_iteration=model.best_iteration)
    
    # Apply physical constraints (OEW ≤ TOW ≤ MTOW)
    if 'openap_oew_kg' in train_features_df.columns and 'openap_mtow_kg' in train_features_df.columns:
        for i, row in train_features_df.iterrows():
            if pd.notna(row['openap_oew_kg']) and pd.notna(row['openap_mtow_kg']):
                tow_train_pred[i] = np.clip(tow_train_pred[i], row['openap_oew_kg'], row['openap_mtow_kg'])
    
    train_results = pd.DataFrame({
        'flight_id': train_features_df['flight_id'],
        'tow': tow_train_pred
    })
    
    train_output_path = PREDICTIONS_OUTPUT_DIR / "tow_predictions_train_v2.csv"
    train_results.to_csv(train_output_path, index=False)
    print(f"✓ Saved: {train_output_path}")
    print(f"  Flights: {len(train_results)}")
    print(f"  TOW range: {tow_train_pred.min():.2f} - {tow_train_pred.max():.2f} kg")
    print(f"  TOW mean: {tow_train_pred.mean():.2f} kg")
    
    # Predict for ranking set
    print("\n▶ Predicting TOW for ranking flights...")
    X_rank_new = rank_features_df[available_features].copy()
    
    # Drop object columns
    object_cols_rank = [col for col in X_rank_new.columns if X_rank_new[col].dtype == 'object' and col not in cat_features]
    if object_cols_rank:
        print(f"Dropping {len(object_cols_rank)} object columns from rank predictions")
        X_rank_new = X_rank_new.drop(columns=object_cols_rank)
    
    for cat in cat_features:
        if cat in X_rank_new.columns:
            X_rank_new[cat] = X_rank_new[cat].astype('category')
    
    for col in X_rank_new.columns:
        if col not in cat_features:
            X_rank_new[col].fillna(X_rank_new[col].median() if X_rank_new[col].notna().sum() > 0 else 0, inplace=True)
    
    tow_rank_pred = model.predict(X_rank_new, num_iteration=model.best_iteration)
    
    # Apply physical constraints
    if 'openap_oew_kg' in rank_features_df.columns and 'openap_mtow_kg' in rank_features_df.columns:
        for i, row in rank_features_df.iterrows():
            if pd.notna(row['openap_oew_kg']) and pd.notna(row['openap_mtow_kg']):
                tow_rank_pred[i] = np.clip(tow_rank_pred[i], row['openap_oew_kg'], row['openap_mtow_kg'])
    
    rank_results = pd.DataFrame({
        'flight_id': rank_features_df['flight_id'],
        'tow': tow_rank_pred
    })
    
    rank_output_path = PREDICTIONS_OUTPUT_DIR / "tow_predictions_rank_v2.csv"
    rank_results.to_csv(rank_output_path, index=False)
    print(f"✓ Saved: {rank_output_path}")
    print(f"  Flights: {len(rank_results)}")
    print(f"  TOW range: {tow_rank_pred.min():.2f} - {tow_rank_pred.max():.2f} kg")
    print(f"  TOW mean: {tow_rank_pred.mean():.2f} kg")
    
    # Process final flights (always attempt)
    print(f"\nChecking for final dataset...")
    print(f"FLIGHTLIST_FINAL path: {FLIGHTLIST_FINAL}")
    print(f"FLIGHTLIST_FINAL exists: {FLIGHTLIST_FINAL.exists()}")
    print(f"FLIGHTS_FINAL_DIR path: {FLIGHTS_FINAL_DIR}")
    print(f"FLIGHTS_FINAL_DIR exists: {FLIGHTS_FINAL_DIR.exists()}")
    
    if FLIGHTLIST_FINAL.exists():
        print("\n" + "="*80)
        print("Processing FINAL flights")
        print("="*80)
        
        flightlist_final = pd.read_parquet(FLIGHTLIST_FINAL)
        print(f"Total flights: {len(flightlist_final)}")
        
        # Check cache
        cached_count = sum(1 for _, row in flightlist_final.iterrows() 
                          if (final_features_dir / f"{row['flight_id']}.parquet").exists())
        print(f"Found {cached_count} cached features, will process {len(flightlist_final) - cached_count} new flights")
        
        final_data = [
            (row['flight_id'], row['aircraft_type'], row['origin_icao'], 
             row['destination_icao'], row['takeoff'], row['landed'])
            for _, row in flightlist_final.iterrows()
        ]
        
        process_func = partial(process_and_save_flight_features, 
                              flights_dir=FLIGHTS_FINAL_DIR, 
                              output_dir=final_features_dir)
        
        with Pool(num_cores) as pool:
            final_features = pool.map(process_func, final_data)
        
        final_features = [f for f in final_features if f is not None]
        final_features_df = pd.DataFrame(final_features)
        print(f"✓ Extracted features for {len(final_features_df)} flights")
        
        # Add airport and aircraft data
        final_features_df = final_features_df.merge(
            apt_df[['icao', 'elevation', 'latitude', 'longitude']],
            left_on='origin_icao',
            right_on='icao',
            how='left'
        ).rename(columns={'elevation': 'adep_elevation_ft', 'latitude': 'adep_lat', 'longitude': 'adep_lon'}).drop(columns=['icao'])
        
        final_features_df = final_features_df.merge(
            apt_df[['icao', 'elevation', 'latitude', 'longitude']],
            left_on='destination_icao',
            right_on='icao',
            how='left'
        ).rename(columns={'elevation': 'ades_elevation_ft', 'latitude': 'ades_lat', 'longitude': 'ades_lon'}).drop(columns=['icao'])
        
        if all(col in final_features_df.columns for col in ['adep_lat', 'adep_lon', 'ades_lat', 'ades_lon']):
            final_features_df['great_circle_distance_nm'] = final_features_df.apply(
                lambda row: calculate_great_circle_distance(row['adep_lat'], row['adep_lon'], row['ades_lat'], row['ades_lon'])
                if pd.notna(row['adep_lat']) and pd.notna(row['adep_lon']) and pd.notna(row['ades_lat']) and pd.notna(row['ades_lon'])
                else np.nan,
                axis=1
            )
        
        final_features_df['aircraft_code'] = final_features_df['aircraft_type'].str.lower()
        final_features_df = final_features_df.merge(
            openap_aircraft_df,
            on='aircraft_code',
            how='left'
        ).drop(columns=['aircraft_code'])
        
        final_features_df = add_derived_features(final_features_df)
        
        # Predict - handle missing features
        print(f"\nPreparing features for prediction...")
        print(f"Required features: {len(available_features)}")
        print(f"Available in final_features_df: {len([f for f in available_features if f in final_features_df.columns])}")
        
        # Add missing features with NaN
        for feat in available_features:
            if feat not in final_features_df.columns:
                final_features_df[feat] = np.nan
                print(f"Added missing feature: {feat}")
        
        X_final_new = final_features_df[available_features].copy()
        
        # Drop any object columns
        object_cols_final = [col for col in X_final_new.columns if X_final_new[col].dtype == 'object' and col not in cat_features]
        if object_cols_final:
            print(f"Dropping {len(object_cols_final)} object columns from final dataset")
            X_final_new = X_final_new.drop(columns=object_cols_final)
        
        for cat in cat_features:
            if cat in X_final_new.columns:
                X_final_new[cat] = X_final_new[cat].astype('category')
        
        for col in X_final_new.columns:
            if col not in cat_features:
                X_final_new[col].fillna(X_final_new[col].median() if X_final_new[col].notna().sum() > 0 else 0, inplace=True)
        
        tow_final_pred = model.predict(X_final_new, num_iteration=model.best_iteration)
        
        # Apply physical constraints
        if 'openap_oew_kg' in final_features_df.columns and 'openap_mtow_kg' in final_features_df.columns:
            for i, row in final_features_df.iterrows():
                if pd.notna(row['openap_oew_kg']) and pd.notna(row['openap_mtow_kg']):
                    tow_final_pred[i] = np.clip(tow_final_pred[i], row['openap_oew_kg'], row['openap_mtow_kg'])
        
        final_results = pd.DataFrame({
            'flight_id': final_features_df['flight_id'],
            'tow': tow_final_pred
        })
        
        final_output_path = PREDICTIONS_OUTPUT_DIR / "tow_predictions_final_v2.csv"
        final_results.to_csv(final_output_path, index=False)
        print(f"✓ Saved: {final_output_path}")
        print(f"  Flights: {len(final_results)}")
    
    print("\n" + "="*80)
    print("TOW PREDICTION COMPLETE!")
    print("="*80)
    print(f"\n📁 Feature Cache Locations:")
    print(f"   Train:  {train_features_dir}")
    print(f"   Rank:   {rank_features_dir}")
    print(f"   Final:  {final_features_dir}")
    print(f"\n📊 Prediction Outputs:")
    print(f"   Train:  {train_output_path}")
    print(f"   Rank:   {rank_output_path}")
    if FLIGHTLIST_FINAL.exists():
        print(f"   Final:  {final_output_path}")
    print(f"\n🤖 Model & Metadata:")
    print(f"   Model:  {model_path}")
    print(f"   Features: {features_list_path}")
    print(f"   Importance: {importance_path}")
    print(f"\n✅ Key Improvements:")
    print(f"   • Using updated 2024 dataset (848 features)")
    print(f"   • Slice-based feature extraction from trajectories")
    print(f"   • Cached features for fast reprocessing (no duplicate work)")
    print(f"   • Kinematics-only features (no mass/energy/thrust)")
    print(f"   • {len(available_features)} features for training")
    print(f"   • Trained model saved for reuse")
    print(f"   • Predictions for ALL datasets (train, rank, final)")
    print("="*80)


if __name__ == '__main__':
    main()
