"""
Purpose: Generate realistic time-series sequences from static predictive maintenance data
Input: preprocessed_data\predictive_maintenance_subset_400.csv
Output: preprocessed_data/timestamped_predictive_maintenance_timeseries_NEW.csv

Pipeline Structure:
1. Data Loading & Configuration
2. Sequence Generation Functions
3. Sensor Signal Generation
4. Time-Series Assembly
5. Validation & Quality Checks
6. Summary Statistics & Export
"""

# Imports
import numpy as np
import pandas as pd

from typing import Tuple, Dict, List
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')

# Sensor bounds (from original predictive_maintenance.csv)
# Prevents drift/jitter from pushing values into unrealistic ranges
SENSOR_BOUNDS = {
    'air_temperature_K': (296.0, 304.0),
    'process_temperature_K': (305.0, 314.0),
    'rotational_speed_rpm': (1200, 2500),
    'torque_Nm': (10.0, 70.0),
    'tool_wear_min': (0.0, 250.0)
}

def clip_sensor_values(sequence: np.ndarray, sensor_name: str) -> np.ndarray:
    """Clip sensor values to realistic bounds after drift/jitter/anomalies."""
    if sensor_name in SENSOR_BOUNDS:
        min_val, max_val = SENSOR_BOUNDS[sensor_name]
        return np.clip(sequence, min_val, max_val)
    return sequence

# SECTION 1: DATA LOADING & CONFIGURATION

def load_subset_data(path: str) -> pd.DataFrame:
    """Load the 400-machine subset dataset."""
    df = pd.read_csv(path)
    print("[INFO] Loading Subset Data")
    print(f"Total Machines: {len(df)}")
    print(f"Failing Machines: {(df['Target'] == 1).sum()}")
    print(f"Healthy Machines: {(df['Target'] == 0).sum()}")
    print(f"Failure Rate: {(df['Target'] == 1).sum() / len(df):.4f}")
    print()
    return df


# SECTION 2: SEQUENCE GENERATION FUNCTIONS

def generate_sequence_lengths(n_machines: int, random_state: int = 42) -> np.ndarray:
    """Generate random sequence lengths between 60 and 120 timesteps."""
    np.random.seed(random_state)
    return np.random.randint(60, 121, size=n_machines)


def calculate_failure_window(seq_length: int) -> Tuple[int, int]:
    """Calculate failure window with weighted random positioning (50-95% of sequence)."""
    u = np.random.rand()
    if u < 0.2:
        start_frac = np.random.uniform(0.50, 0.70)
    elif u < 0.6:
        start_frac = np.random.uniform(0.70, 0.85)
    else:
        start_frac = np.random.uniform(0.85, 0.95)
    
    failure_start = int(start_frac * seq_length)
    failure_start = max(1, min(failure_start, seq_length - 2))
    return failure_start, seq_length - 1


def add_sensor_drift(sequence: np.ndarray, drift_pct: float = 0.015) -> np.ndarray:
    """Add cumulative random walk drift (0.5-2% total drift)."""
    drift_magnitude = np.abs(sequence.mean()) * drift_pct
    step_size = drift_magnitude / len(sequence)
    walk = np.cumsum(np.random.normal(0, step_size, len(sequence)))
    return sequence + walk


def add_measurement_jitter(sequence: np.ndarray, jitter_pct: float = 0.02) -> np.ndarray:
    """Add zero-mean Gaussian jitter (1-3% of reading)."""
    jitter = np.random.normal(0, np.abs(sequence) * jitter_pct)
    return sequence + jitter


def inject_anomalies(sequence: np.ndarray, anomaly_prob: float = 0.10, 
                     sensor_name: str = 'generic') -> np.ndarray:
    """Inject low-frequency anomalies (spikes, drops, plateaus)."""
    if np.random.rand() > anomaly_prob or len(sequence) < 12:
        return sequence
    
    n_anomalies = np.random.randint(1, min(4, len(sequence) // 10 + 1))
    for _ in range(n_anomalies):
        max_start = max(1, len(sequence) - 10)
        start_idx = np.random.randint(0, max_start)
        duration = np.random.randint(3, min(11, len(sequence) - start_idx))
        end_idx = min(start_idx + duration, len(sequence))
        
        anomaly_type = np.random.choice(['spike', 'drop', 'plateau'], p=[0.4, 0.4, 0.2])
        
        if sensor_name == 'tool_wear' and anomaly_type != 'plateau':
            anomaly_type = 'plateau'
        
        if anomaly_type == 'spike':
            magnitude = np.abs(sequence[start_idx]) * np.random.uniform(0.10, 0.25)
            sequence[start_idx:end_idx] += magnitude
        elif anomaly_type == 'drop':
            magnitude = np.abs(sequence[start_idx]) * np.random.uniform(0.10, 0.25)
            sequence[start_idx:end_idx] -= magnitude
        else:
            plateau_value = sequence[start_idx]
            sequence[start_idx:end_idx] = plateau_value
    
    return sequence


# SECTION 3: SENSOR SIGNAL GENERATION

def generate_tool_wear_sequence(
    final_value: float,
    seq_length: int,
    is_failing: bool,
    failure_type: str = None
) -> np.ndarray:
    """Generate tool wear with failure-type-specific non-linear degradation."""
    start_pct = np.random.uniform(0.20, 0.40)
    start_value = final_value * start_pct
    
    t = np.linspace(0, 1, seq_length)
    
    curve_type = np.random.choice(['exponential', 'sigmoid', 'piecewise'])
    
    if is_failing and failure_type == 'Tool Wear Failure':
        if curve_type == 'exponential':
            exponent = np.random.uniform(2.5, 4.0)
            base_curve = start_value + (final_value - start_value) * (t ** exponent)
        elif curve_type == 'sigmoid':
            steepness = np.random.uniform(8, 12)
            midpoint = np.random.uniform(0.6, 0.8)
            sigmoid = 1 / (1 + np.exp(-steepness * (t - midpoint)))
            base_curve = start_value + (final_value - start_value) * sigmoid
        else:
            breakpoint = np.random.randint(int(seq_length * 0.6), int(seq_length * 0.8))
            early = np.linspace(start_value, start_value + (final_value - start_value) * 0.4, breakpoint)
            late = np.linspace(early[-1], final_value, seq_length - breakpoint)
            base_curve = np.concatenate([early, late])
    else:
        exponent = np.random.uniform(1.5, 2.5)
        base_curve = start_value + (final_value - start_value) * (t ** exponent)
    
    # Add realistic noise BEFORE monotonicity enforcement
    sequence = add_measurement_jitter(base_curve, jitter_pct=0.015)
    
    # Enforce monotonic increasing with small jitter tolerance (1%)
    jitter_tolerance = final_value * 0.01
    for i in range(1, len(sequence)):
        if sequence[i] < sequence[i-1] - jitter_tolerance:
            # Allow small decreases (sensor noise), but enforce trend
            sequence[i] = sequence[i-1]
    
    # Inject anomalies AFTER monotonicity (only plateaus for tool wear)
    sequence = inject_anomalies(sequence, anomaly_prob=0.10, sensor_name='tool_wear')
    
    sequence[-1] = final_value
    sequence = clip_sensor_values(sequence, 'tool_wear_min')
    
    return sequence


def generate_rotational_speed_sequence(
    final_value: float,
    seq_length: int,
    is_failing: bool,
    failure_type: str = None
) -> np.ndarray:
    """Generate RPM with failure-type-specific degradation and realistic noise."""
    t = np.linspace(0, seq_length - 1, seq_length)
    
    base_value = final_value
    
    if is_failing:
        if failure_type == 'Tool Wear Failure':
            trend = np.linspace(0, -final_value * 0.06, seq_length)
        elif failure_type == 'Overstrain Failure':
            trend = np.linspace(0, -final_value * 0.08, seq_length)
        else:
            trend = np.linspace(0, -final_value * 0.04, seq_length)
    else:
        trend = np.linspace(0, np.random.uniform(-10, 10), seq_length)
    
    cycle_freq = np.random.uniform(0.1, 0.3)
    oscillation = final_value * 0.03 * np.sin(2 * np.pi * cycle_freq * t)
    
    sequence = base_value + trend + oscillation
    sequence = add_sensor_drift(sequence, drift_pct=0.015)
    sequence = add_measurement_jitter(sequence, jitter_pct=0.025)
    sequence = inject_anomalies(sequence, anomaly_prob=0.10, sensor_name='rpm')
    
    adjustment = final_value - sequence[-1]
    sequence = sequence + adjustment
    sequence = clip_sensor_values(sequence, 'rotational_speed_rpm')
    
    return sequence


def generate_process_temperature_sequence(
    final_value: float,
    seq_length: int,
    is_failing: bool,
    failure_start: int = None,
    failure_type: str = None
) -> np.ndarray:
    """Generate process temperature with failure-type-specific heat patterns."""
    if is_failing and failure_start is not None:
        pre_failure_length = failure_start
        failure_length = seq_length - failure_start
        
        if failure_type == 'Heat Dissipation Failure':
            start_temp = final_value - np.random.uniform(3.0, 5.0)
            pre_failure = np.linspace(start_temp, final_value - 2.0, pre_failure_length)
            t_fail = np.linspace(0, 1, failure_length)
            steep_rise = (final_value - 2.0) + 2.0 * (t_fail ** 2.5)
            failure_phase = steep_rise
        elif failure_type == 'Overstrain Failure':
            start_temp = final_value - np.random.uniform(2.0, 3.5)
            pre_failure = np.linspace(start_temp, final_value - 1.0, pre_failure_length)
            failure_phase = np.linspace(final_value - 1.0, final_value, failure_length)
        else:
            start_temp = final_value - np.random.uniform(2.0, 4.0)
            pre_failure = np.linspace(start_temp, final_value - 1.5, pre_failure_length)
            failure_phase = np.linspace(final_value - 1.5, final_value, failure_length)
        
        sequence = np.concatenate([pre_failure, failure_phase])
    else:
        start_temp = final_value - np.random.uniform(1.0, 2.5)
        sequence = np.linspace(start_temp, final_value, seq_length)
    
    sequence = add_sensor_drift(sequence, drift_pct=0.01)
    sequence = add_measurement_jitter(sequence, jitter_pct=0.015)
    sequence = inject_anomalies(sequence, anomaly_prob=0.10, sensor_name='process_temp')
    sequence = clip_sensor_values(sequence, 'process_temperature_K')
    
    return sequence


def generate_air_temperature_sequence(
    final_value: float,
    seq_length: int,
    failure_type: str = None
) -> np.ndarray:
    """Generate air temperature with mild fluctuations and realistic noise."""
    drift_range = np.random.uniform(-0.5, 0.5)
    drift = np.linspace(0, drift_range, seq_length)
    
    t = np.linspace(0, seq_length - 1, seq_length)
    oscillation = 0.2 * np.sin(2 * np.pi * 0.05 * t)
    
    sequence = final_value + drift + oscillation
    sequence = add_sensor_drift(sequence, drift_pct=0.008)
    sequence = add_measurement_jitter(sequence, jitter_pct=0.01)
    
    if failure_type == 'Heat Dissipation Failure':
        sequence = inject_anomalies(sequence, anomaly_prob=0.15, sensor_name='air_temp')
    
    sequence[-1] = final_value
    sequence = clip_sensor_values(sequence, 'air_temperature_K')
    
    return sequence


def generate_torque_sequence(
    final_value: float,
    seq_length: int,
    is_failing: bool,
    failure_start: int = None,
    failure_type: str = None
) -> np.ndarray:
    """Generate torque with failure-type-specific patterns and realistic noise."""
    t = np.linspace(0, seq_length - 1, seq_length)
    base_value = final_value
    
    cycle_freq = np.random.uniform(0.08, 0.15)
    oscillation = base_value * 0.10 * np.sin(2 * np.pi * cycle_freq * t)
    
    sequence = base_value + oscillation
    sequence = add_sensor_drift(sequence, drift_pct=0.012)
    sequence = add_measurement_jitter(sequence, jitter_pct=0.035)
    
    if is_failing and failure_start is not None:
        if failure_type == 'Overstrain Failure':
            spike_start = max(failure_start - 5, seq_length // 2)
            spike_magnitude = base_value * np.random.uniform(0.20, 0.40)
            spike_curve = np.linspace(0, spike_magnitude, seq_length - spike_start)
            sequence[spike_start:] += spike_curve
            sequence_segment = sequence[spike_start:].copy()
            sequence[spike_start:] = inject_anomalies(sequence_segment, anomaly_prob=0.30, sensor_name='torque')
        elif failure_type == 'Tool Wear Failure':
            grad_increase = base_value * 0.10 * (t / seq_length)
            sequence += grad_increase
        else:
            spike_start = max(failure_start, seq_length - 10)
            spike_magnitude = base_value * np.random.uniform(0.15, 0.25)
            spike_curve = np.linspace(0, spike_magnitude, seq_length - spike_start)
            sequence[spike_start:] += spike_curve
    else:
        # 5% of healthy machines get mild anomalies (increase class overlap)
        if np.random.rand() < 0.05:
            sequence = inject_anomalies(sequence, anomaly_prob=0.15, sensor_name='torque')
        else:
            sequence = inject_anomalies(sequence, anomaly_prob=0.08, sensor_name='torque')
    
    sequence[-1] = final_value
    sequence = clip_sensor_values(sequence, 'torque_Nm')
    
    return sequence


def generate_synthetic_RUL(
    seq_length: int,
    is_failing: bool,
    failure_position: int = None
) -> np.ndarray:
    """Generate RUL with non-linear stochastic decay."""
    if is_failing:
        max_rul = failure_position
        t = np.linspace(0, 1, seq_length)
        alpha = np.random.uniform(1.5, 3.0)
        base_rul = max_rul * (1 - t**alpha)
        
        noise_magnitude = max_rul * 0.03
        noise = np.random.normal(0, noise_magnitude, seq_length)
        rul = base_rul + noise
        rul = np.maximum(rul, 0)
        
        for i in range(len(rul) - 1, 0, -1):
            if rul[i] > rul[i-1]:
                rul[i] = rul[i-1]
        
        rul[failure_position:] = 0
        rul = np.maximum(rul, 0)
    else:
        max_rul = seq_length + np.random.randint(20, 60)
        t = np.linspace(0, 1, seq_length)
        alpha = np.random.uniform(1.0, 1.8)
        base_rul = max_rul * (1 - 0.5 * t**alpha)
        
        noise = np.random.normal(0, max_rul * 0.02, seq_length)
        rul = base_rul + noise
        rul = np.maximum(rul, seq_length * 0.2)
    
    return rul


# SECTION 4: TIME-SERIES ASSEMBLY

def generate_machine_sequence(
    machine_data: pd.Series,
    seq_length: int,
    machine_index: int,
    start_timestamp: datetime
) -> pd.DataFrame:
    """Generate complete time-series sequence for a single machine."""
    is_failing = machine_data['Target'] == 1
    failure_type = machine_data['Failure Type'] if is_failing else None
    
    if is_failing:
        failure_start, failure_position = calculate_failure_window(seq_length)
    else:
        failure_start = None
        failure_position = None
    
    timestamps = [start_timestamp + timedelta(hours=i) for i in range(seq_length)]
    
    tool_wear = generate_tool_wear_sequence(
        machine_data['Tool wear [min]'],
        seq_length,
        is_failing,
        failure_type
    )
    
    rotational_speed = generate_rotational_speed_sequence(
        machine_data['Rotational speed [rpm]'],
        seq_length,
        is_failing,
        failure_type
    )
    
    process_temp = generate_process_temperature_sequence(
        machine_data['Process temperature [K]'],
        seq_length,
        is_failing,
        failure_start,
        failure_type
    )
    
    air_temp = generate_air_temperature_sequence(
        machine_data['Air temperature [K]'],
        seq_length,
        failure_type
    )
    
    torque = generate_torque_sequence(
        machine_data['Torque [Nm]'],
        seq_length,
        is_failing,
        failure_start,
        failure_type
    )
    
    synthetic_rul = generate_synthetic_RUL(seq_length, is_failing, failure_position)
    
    is_failure_flags = np.zeros(seq_length, dtype=int)
    if is_failing:
        is_failure_flags[failure_start:] = 1
    
    failure_types = ['No Failure'] * seq_length
    if is_failing:
        for i in range(failure_start, seq_length):
            failure_types[i] = machine_data['Failure Type']
    
    # Create DataFrame
    sequence_df = pd.DataFrame({
        'product_id': machine_data['Product ID'],
        'unit_id': machine_data['UDI'],
        'timestamp': timestamps,
        'step_index': np.arange(seq_length),
        'engine_type': machine_data['Type'],
        'air_temperature_K': air_temp,
        'process_temperature_K': process_temp,
        'rotational_speed_rpm': rotational_speed,
        'torque_Nm': torque,
        'tool_wear_min': tool_wear,
        'is_failure': is_failure_flags,
        'failure_type': failure_types,
        'synthetic_RUL': synthetic_rul
    })
    
    return sequence_df


def generate_all_sequences(df_subset: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic time-series for all machines."""
    np.random.seed(random_state)
    
    print("[INFO] Generating Synthetic Time-Series")
    
    n_machines = len(df_subset)
    sequence_lengths = generate_sequence_lengths(n_machines, random_state)
    
    print(f"Total Machines to Process: {n_machines}")
    print(f"Sequence Length Range: {sequence_lengths.min()} - {sequence_lengths.max()}")
    print(f"Average Sequence Length: {sequence_lengths.mean():.1f}")
    print()
    
    # Define 2014-2024 timespan
    start_date = datetime(2014, 1, 1, 0, 0, 0)
    end_date = datetime(2024, 12, 31, 23, 59, 59)
    total_hours = int((end_date - start_date).total_seconds() / 3600)
    
    print(f"Distributing Machines Across: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total Timespan: {total_hours:,} Hours (~{total_hours/8760:.1f} Years)")
    print()
    
    all_sequences = []
    
    for idx, (_, machine_data) in enumerate(df_subset.iterrows()):
        if (idx + 1) % 50 == 0:
            print(f"Processing Machine {idx + 1}/{n_machines}...")
        
        seq_length = sequence_lengths[idx]
        
        # Calculate maximum start hour to ensure sequence fits within 2014-2024
        max_start_hour = total_hours - seq_length
        
        # Random start time within valid range
        random_start_hour = np.random.randint(0, max_start_hour + 1)
        start_timestamp = start_date + timedelta(hours=random_start_hour)
        
        sequence_df = generate_machine_sequence(
            machine_data,
            seq_length,
            idx,
            start_timestamp
        )
        
        all_sequences.append(sequence_df)
    
    # Combine all sequences
    df_timeseries = pd.concat(all_sequences, axis=0, ignore_index=True)
    
    print(f"\n[RESULT] Total Rows Generated: {len(df_timeseries)}")
    print()
    
    return df_timeseries


# SECTION 5: VALIDATION & QUALITY CHECKS

def validate_timeseries(df_timeseries: pd.DataFrame, df_subset: pd.DataFrame) -> None:
    """Comprehensive validation of generated time-series dataset."""
    print("[INFO] Validation Report")
    
    # 1. Structural checks
    print("\n[INFO] Structural Checks:")
    print(f"Total Rows: {len(df_timeseries)}")
    print(f"Total Unique Machines: {df_timeseries['product_id'].nunique()}")
    print(f"Expected Machines: {len(df_subset)}")
    
    seq_lengths = df_timeseries.groupby('product_id').size()
    print(f"\n[INFO] Sequence Length Distribution:")
    print(f"Min: {seq_lengths.min()}")
    print(f"Max: {seq_lengths.max()}")
    print(f"Mean: {seq_lengths.mean():.1f}")
    print(f"Median: {seq_lengths.median():.1f}")
    
    if df_timeseries['product_id'].nunique() == len(df_subset):
        print("[INFO] All Machines have Sequences")
    else:
        print("[ERROR] Mismatch - Some Machines Missing Sequences")
    
    # 2. Temporal checks
    print("\n[INFO] Temporal Checks:")
    
    # Check monotonic timestamps per machine
    monotonic_check = df_timeseries.groupby('product_id')['timestamp'].apply(
        lambda x: x.is_monotonic_increasing
    )
    if monotonic_check.all():
        print("[INFO] All Timestamps are Monotonically Increasing Per Machine")
    else:
        print(f"[ERROR] {(~monotonic_check).sum()} Machines Have Non-Monotonic Timestamps")
    
    # Check delta t = 1 hour
    def check_delta_t(group):
        diffs = group['timestamp'].diff().dropna()
        return (diffs == timedelta(hours=1)).all()
    
    delta_t_check = df_timeseries.groupby('product_id').apply(check_delta_t)
    if delta_t_check.all():
        print("[INFO] All Timestamp Deltas are Exactly 1 Hour")
    else:
        print(f"[ERROR] {(~delta_t_check).sum()} Machines Have Incorrect Timestamp Deltas")
    
    # Check no duplicate timestamps per machine
    dup_timestamps = df_timeseries.groupby('product_id')['timestamp'].apply(
        lambda x: x.duplicated().any()
    )
    if not dup_timestamps.any():
        print("[INFO] No Duplicate Timestamps Within Machines")
    else:
        print(f"[ERROR] {dup_timestamps.sum()} Machines Have Duplicate Timestamps")
    
    # 3. Sensor realism checks
    print("\n[INFO] Sensor Realism Checks:")
    
    # Tool wear monotonicity
    def check_tool_wear_monotonic(group):
        return (group['tool_wear_min'].diff().dropna() >= 0).all()
    
    tool_wear_check = df_timeseries.groupby('product_id').apply(check_tool_wear_monotonic)
    pct_monotonic = tool_wear_check.mean() * 100
    print(f"Tool Wear Monotonic: {pct_monotonic:.1f}% of machines")
    
    if pct_monotonic >= 99:
        print("[INFO] Tool Wear is Monotonically Increasing for Majority of Machines")
    else:
        print(f"[WARNING] {(~tool_wear_check).sum()} Machines Have Non-Monotonic Tool Wear")
    
    # Process temperature range
    process_temp_min = df_timeseries['process_temperature_K'].min()
    process_temp_max = df_timeseries['process_temperature_K'].max()
    print(f"\n[INFO] Process Temperature Range: [{process_temp_min:.2f}, {process_temp_max:.2f}] K")
    if 304 <= process_temp_min <= 307 and 312 <= process_temp_max <= 316:
        print("[INFO] Process Temperature Within Realistic Bounds")
    else:
        print("[WARN] Process Temperature May Be Outside Expected Range")
    
    # RPM range
    rpm_min = df_timeseries['rotational_speed_rpm'].min()
    rpm_max = df_timeseries['rotational_speed_rpm'].max()
    print(f"\n[INFO] Rotational Speed Range: [{rpm_min:.0f}, {rpm_max:.0f}] rpm")
    if 1100 <= rpm_min <= 1200 and 2800 <= rpm_max <= 3000:
        print("[INFO] Rotational Speed Within Realistic Bounds")
    else:
        print("[WARN] Rotational Speed May Be Outside Expected Range")
    
    # Torque range
    torque_min = df_timeseries['torque_Nm'].min()
    torque_max = df_timeseries['torque_Nm'].max()
    print(f"\n[INFO] Torque Range: [{torque_min:.2f}, {torque_max:.2f}] Nm")
    if 3 <= torque_min <= 5 and 75 <= torque_max <= 85:
        print("[INFO] Torque Within Realistic Bounds")
    else:
        print("[WARN] Torque May Be Outside Expected Range")
    
    # 4. Failure pattern checks
    print("\n[INFO] Failure Pattern Checks")
    
    # Count failing machines
    failing_machines = df_timeseries[df_timeseries['is_failure'] == 1]['product_id'].nunique()
    total_machines = df_timeseries['product_id'].nunique()
    failure_rate = failing_machines / total_machines
    
    print(f"[INFO] Failing Machines: {failing_machines}")
    print(f"[INFO] Healthy Machines: {total_machines - failing_machines}")
    print(f"[INFO] Failure Rate: {failure_rate:.4f} ({failure_rate * 100:.2f}%)")
    
    expected_failures = (df_subset['Target'] == 1).sum()
    if failing_machines == expected_failures:
        print(f"[INFO] Failure Count Matches Expected ({expected_failures})")
    else:
        print(f"[ERROR] Failure Count Mismatch - Expected {expected_failures}, Got {failing_machines}")
    
    # Check failures occur in second half
    def check_failure_position(group):
        if group['is_failure'].sum() == 0:
            return True
        first_failure_idx = group[group['is_failure'] == 1]['step_index'].iloc[0]
        seq_len = len(group)
        return first_failure_idx >= seq_len * 0.50
    
    failure_position_check = df_timeseries.groupby('product_id').apply(check_failure_position)
    pct_correct_position = failure_position_check.mean() * 100
    print(f"\n[INFO] Failures in Second Half: {pct_correct_position:.1f}% of sequences")
    
    if pct_correct_position >= 95:
        print("[INFO] Failures Occur in Second Half (50-95% range)")
    else:
        print(f"[WARN] Some Failures Too Early in Sequence")
    
    fail_positions = []
    for pid, group in df_timeseries[df_timeseries['is_failure'] == 1].groupby('product_id'):
        first_fail = group['step_index'].iloc[0]
        seq_len = df_timeseries[df_timeseries['product_id'] == pid]['step_index'].max() + 1
        fail_positions.append(first_fail / seq_len)
    
    if len(fail_positions) > 0:
        fail_pos_array = np.array(fail_positions)
        print(f"\n[INFO] Failure Position Distribution:")
        print(f"Mean: {fail_pos_array.mean()*100:.1f}% of Sequence")
        print(f"50-70% Range: {((fail_pos_array >= 0.5) & (fail_pos_array < 0.7)).sum()} Machines")
        print(f"70-85% Range: {((fail_pos_array >= 0.7) & (fail_pos_array < 0.85)).sum()} Machines")
        print(f"85-95% Range: {((fail_pos_array >= 0.85) & (fail_pos_array <= 0.95)).sum()} Machines")
    
    # Failure type distribution
    print("\n[INFO] Failure Type Distribution")
    
    original_failures = df_subset[df_subset['Target'] == 1]['Failure Type'].value_counts()
    synthetic_failures = df_timeseries[
        (df_timeseries['is_failure'] == 1) & 
        (df_timeseries['failure_type'] != 'No Failure')
    ].groupby('product_id')['failure_type'].first().value_counts()
    
    print("\n[INFO] Original (Subset):")
    print(original_failures.sort_index())
    print("\n[INFO] Synthetic (Time-Series):")
    print(synthetic_failures.sort_index())
    
    # 6. RUL checks
    print("\n[INFO] Remaining Useful Life (RUL) Checks")
    
    def check_rul_decreasing(group):
        diffs = group['synthetic_RUL'].diff().dropna()
        # FIX: RUL should decrease, so diff should be <= 0 (not <= 0.1)
        return (diffs <= 0).all()
    
    rul_check = df_timeseries.groupby('product_id').apply(check_rul_decreasing)
    pct_decreasing = rul_check.mean() * 100
    print(f"[INFO] RUL Decreasing: {pct_decreasing:.1f}% of machines")
    
    if pct_decreasing >= 95:  # Allow 5% tolerance for edge cases
        print("[INFO] RUL is Monotonically Decreasing")
    else:
        print(f"[WARNING] {(~rul_check).sum()} Machines Have Non-Decreasing RUL")
    
    machines_with_zero_rul = df_timeseries[df_timeseries['synthetic_RUL'] < 1.0]['product_id'].unique()
    machines_with_failures = df_timeseries[df_timeseries['is_failure'] == 1]['product_id'].unique()
    
    if len(set(machines_with_zero_rul) - set(machines_with_failures)) == 0:
        print("[INFO] Near-Zero RUL Only for Failing Machines")
    else:
        print(f"[WARNING] {len(set(machines_with_zero_rul) - set(machines_with_failures))} Healthy Machines With Low RUL")
    
    print("\n[INFO] RUL Distribution Analysis")
    
    all_rul = df_timeseries['synthetic_RUL'].values
    low_rul = (all_rul < 24).sum()
    mid_rul = ((all_rul >= 24) & (all_rul < 72)).sum()
    high_rul = (all_rul >= 72).sum()
    total = len(all_rul)
    
    print(f"[INFO] RUL Distribution:")
    print(f"< 24H: {low_rul:,} ({low_rul/total*100:.1f}%)")
    print(f"24-72H: {mid_rul:,} ({mid_rul/total*100:.1f}%)")
    print(f"> 72H: {high_rul:,} ({high_rul/total*100:.1f}%)")
    
    print(f"\n[INFO] RUL Statistics:")
    print(f"Mean: {all_rul.mean():.1f}h")
    print(f"Std: {all_rul.std():.1f}h")
    print(f"Min: {all_rul.min():.1f}h")
    print(f"Max: {all_rul.max():.1f}h")
    print(f"Unique Values: {len(np.unique(all_rul)):,}")
    
    if len(np.unique(all_rul)) < 100:
        print("[WARNING] RUL Distribution May be Too Concentrated")
    else:
        print("[INFO] RUL Shows Healthy Diversity")
    
    # 8. Missing values check
    print("\n[INFO] Data Quality - Missing Values Check")
    missing = df_timeseries.isnull().sum().sum()
    if missing == 0:
        print("[INFO] No Missing Values")
    else:
        print(f"[ERROR] {missing} Missing Values Found")
    
    # 9. Sensor clipping summary
    print("\n[INFO] Sensor Clipping Summary")
    
    sensors_to_check = {
        'air_temperature_K': 'air_temperature_K',
        'process_temperature_K': 'process_temperature_K',
        'rotational_speed_rpm': 'rotational_speed_rpm',
        'torque_Nm': 'torque_Nm',
        'tool_wear_min': 'tool_wear_min'
    }
    
    for col_name, bound_key in sensors_to_check.items():
        if bound_key in SENSOR_BOUNDS:
            min_bound, max_bound = SENSOR_BOUNDS[bound_key]
            actual_min = df_timeseries[col_name].min()
            actual_max = df_timeseries[col_name].max()
            
            clipped_low = (df_timeseries[col_name] == min_bound).sum()
            clipped_high = (df_timeseries[col_name] == max_bound).sum()
            
            print(f"\n[INFO] {col_name}:")
            print(f"Bounds: [{min_bound}, {max_bound}]")
            print(f"Actual: [{actual_min:.2f}, {actual_max:.2f}]")
            print(f"Clipped at Min: {clipped_low} Samples ({clipped_low/len(df_timeseries)*100:.2f}%)")
            print(f"Clipped at Max: {clipped_high} Samples ({clipped_high/len(df_timeseries)*100:.2f}%)")
            
            if clipped_low + clipped_high > len(df_timeseries) * 0.10:
                print(f"[WARN] > 10% of Samples Clipped - Consider Adjusting Drift/Jitter")
            else:
                print(f"[INFO] Minimal Clipping (<10%)")
    
    # 10. Tool wear monotonicity check
    print("\n[INFO] Tool Wear Monotonicity Check")
    
    def check_tool_wear_increasing(group):
        diffs = group['tool_wear_min'].diff().dropna()
        # Allow small decreases (1% tolerance for sensor jitter)
        tolerance = group['tool_wear_min'].iloc[-1] * 0.01
        return (diffs >= -tolerance).all()
    
    tool_wear_check = df_timeseries.groupby('product_id').apply(check_tool_wear_increasing)
    pct_increasing = tool_wear_check.mean() * 100
    print(f"Tool Wear Monotonic Increasing: {pct_increasing:.1f}% of machines")
    
    if pct_increasing >= 95:
        print("[INFO] Tool Wear Shows Monotonic Increasing Trend (With Minor Jitter Allowed)")
    else:
        print(f"[WARNING] {(~tool_wear_check).sum()} Machines Have Decreasing Tool Wear Beyond Tolerance")

# Statistical Summarization & Reporting

def create_sequence_summary(df_timeseries: pd.DataFrame) -> pd.DataFrame:
    """Create sequence-level summary (1 row per machine)."""
    summary_data = []
    
    for product_id, group in df_timeseries.groupby('product_id'):
        failure_rows = group[group['is_failure'] == 1]
        
        if len(failure_rows) > 0:
            failure_position = failure_rows['step_index'].iloc[0]
            failure_type = failure_rows['failure_type'].iloc[0]
        else:
            failure_position = None
            failure_type = 'No Failure'
        
        summary = {
            'machine_id': product_id,
            'sequence_length': len(group),
            'engine_type': group['engine_type'].iloc[0],
            'is_failing_machine': 1 if len(failure_rows) > 0 else 0,
            'failure_position': failure_position,
            'failure_type': failure_type,
            'min_air_temp': group['air_temperature_K'].min(),
            'max_air_temp': group['air_temperature_K'].max(),
            'min_process_temp': group['process_temperature_K'].min(),
            'max_process_temp': group['process_temperature_K'].max(),
            'min_rpm': group['rotational_speed_rpm'].min(),
            'max_rpm': group['rotational_speed_rpm'].max(),
            'min_torque': group['torque_Nm'].min(),
            'max_torque': group['torque_Nm'].max(),
            'min_tool_wear': group['tool_wear_min'].min(),
            'max_tool_wear': group['tool_wear_min'].max(),
            'initial_RUL': group['synthetic_RUL'].iloc[0],
            'final_RUL': group['synthetic_RUL'].iloc[-1]
        }
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)


def print_dataset_preview(df_timeseries: pd.DataFrame) -> None:
    """Print head and tail of generated dataset."""
    print("Dataset Preview")
    
    print("\n[RESULT] Head (First 20 Rows)")
    print(df_timeseries.head(20).to_string(index=False))
    
    print("\n[RESULT] Tail (Last 20 Rows)")
    print(df_timeseries.tail(20).to_string(index=False))
    
    print("\n[RESULT] Sample from Failing Machine")
    failing_machine = df_timeseries[df_timeseries['is_failure'] == 1]['product_id'].iloc[0]
    sample = df_timeseries[df_timeseries['product_id'] == failing_machine].tail(15)
    print(sample.to_string(index=False))
    


def print_summary_statistics(df_timeseries: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    """Print comprehensive summary statistics."""
    print("\n[RESULT] Summary Statistics")
    
    print("\n[INFO] Dataset Dimensions")
    print(f"Total Timesteps: {len(df_timeseries):,}")
    print(f"Total Machines: {df_timeseries['product_id'].nunique()}")
    print(f"Failing Machines: {df_summary['is_failing_machine'].sum()}")
    print(f"Healthy Machines: {(df_summary['is_failing_machine'] == 0).sum()}")
    
    print("\n[INFO] Sequence Statistics")
    print(f"Min Sequence Length: {df_summary['sequence_length'].min()}")
    print(f"Max Sequence Length: {df_summary['sequence_length'].max()}")
    print(f"Mean Sequence Length: {df_summary['sequence_length'].mean():.1f}")
    print(f"Median Sequence Length: {df_summary['sequence_length'].median():.1f}")
    
    print("\n[INFO] Sensor Statistics (Full Time-Series)")
    sensors = {
        'Air Temperature [K]': 'air_temperature_K',
        'Process Temperature [K]': 'process_temperature_K',
        'Rotational Speed [rpm]': 'rotational_speed_rpm',
        'Torque [Nm]': 'torque_Nm',
        'Tool Wear [min]': 'tool_wear_min'
    }
    
    for label, col in sensors.items():
        print(f"\n[INFO] {label}:")
        print(f"Mean: {df_timeseries[col].mean():.2f}")
        print(f"Std: {df_timeseries[col].std():.2f}")
        print(f"Min: {df_timeseries[col].min():.2f}")
        print(f"Max: {df_timeseries[col].max():.2f}")
    
    print("\n[INFO] Engine Type Distribution")
    print(df_timeseries.groupby('engine_type')['product_id'].nunique())
    
    print("\n[INFO] Failure Statistics")
    failure_timesteps = (df_timeseries['is_failure'] == 1).sum()
    total_timesteps = len(df_timeseries)
    print(f"Failure Timesteps: {failure_timesteps:,}")
    print(f"Normal Timesteps: {total_timesteps - failure_timesteps:,}")
    print(f"Failure Ratio (Timesteps): {failure_timesteps / total_timesteps:.4f}")
    
    print("\n[INFO] RUL Statistics")
    print(f"Mean RUL: {df_timeseries['synthetic_RUL'].mean():.1f}")
    print(f"Max RUL: {df_timeseries['synthetic_RUL'].max()}")
    print(f"Min RUL: {df_timeseries['synthetic_RUL'].min()}")
    


# MAIN EXECUTION PIPELINE

def main():
    """Main execution function."""
    print("\n")
    print("[RESULT] Synthetic Time-Series Generation for Predictive Maintenance Dataset")
    print()
    
    # Configuration
    input_path = 'preprocessed/predictive_maintenance_subset_400_machines.csv'
    output_timeseries_path = 'preprocessed/predictive_maintenance_timeseries.csv'
    output_summary_path = 'preprocessed/sequence_summary.csv'
    random_state = 42
    
    # Step 1: Load Static Data
    df_subset = load_subset_data(input_path)
    
    # Step 2: Generate Synthetic Time-Series
    df_timeseries = generate_all_sequences(df_subset, random_state)
    
    # Step 3: Validate Generated Time-Series
    validate_timeseries(df_timeseries, df_subset)
    
    # Step 4: Create Sequence Summary
    print("[INFO] Creating Sequence Summary")
    df_summary = create_sequence_summary(df_timeseries)
    print(f"[INFO] Summary Created With {len(df_summary)} Machines")
    print()
    
    # Step 5: Print Dataset Preview
    print_dataset_preview(df_timeseries)
    
    # Step 6: Print Summary Statistics
    print_summary_statistics(df_timeseries, df_summary)
    
    # Step 7: Save Outputs
    print("\n[INFO] Saving Outputs")
    
    df_timeseries.to_csv(output_timeseries_path, index=False)
    print(f"[INFO] Time-Series Saved to: {output_timeseries_path}")
    print(f"[INFO] Rows: {len(df_timeseries):,}")
    print(f"[INFO] Size: {len(df_timeseries) * df_timeseries.shape[1] / 1e6:.2f}M cells")
    
    df_summary.to_csv(output_summary_path, index=False)
    print(f"[INFO] Summary Saved to: {output_summary_path}")
    print(f"[INFO] Rows: {len(df_summary)}")

    print("\n[RESULT] Dataset Generation Complete")

if __name__ == "__main__":
    main()