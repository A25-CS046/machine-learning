import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesGenerator:
    """
    Generates synthetic time-series sequences from snapshot predictive maintenance data.
    
    Each machine in the original dataset is expanded into a temporal sequence
    showing realistic degradation patterns leading to its final state.
    """
    
    def __init__(self, 
                 min_steps=60, 
                 max_steps=80, 
                 time_delta_hours=1,
                 random_seed=42):
        """
        Parameters:
        -----------
        min_steps : int
            Minimum sequence length per machine
        max_steps : int
            Maximum sequence length per machine
        time_delta_hours : int
            Hours between consecutive time-steps
        random_seed : int
            For reproducibility
        """
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.time_delta_hours = time_delta_hours
        np.random.seed(random_seed)
        
    def generate_sequences(self, df_original):
        """
        Main function to generate time-series dataset.
        
        Parameters:
        -----------
        df_original : pd.DataFrame
            Original snapshot dataset
            
        Returns:
        --------
        pd.DataFrame : Synthetic time-series dataset
        """
        sequences = []
        
        print(f"Generating time-series for {len(df_original)} machines...")
        
        for idx, row in df_original.iterrows():
            if idx % 100 == 0:
                print(f"  Processing machine {idx}/{len(df_original)}")
            
            sequence = self._generate_machine_sequence(row)
            sequences.append(sequence)
        
        # Concatenate all sequences
        df_timeseries = pd.concat(sequences, ignore_index=True)
        
        print(f"\n✓ Generated {len(df_timeseries)} time-series observations")
        print(f"✓ Average sequence length: {len(df_timeseries) / len(df_original):.1f} steps")
        
        return df_timeseries
    
    def _generate_machine_sequence(self, final_state):
        """
        Generate degradation sequence for a single machine.
        
        Parameters:
        -----------
        final_state : pd.Series
            The original row representing the machine's final state
            
        Returns:
        --------
        pd.DataFrame : Time-series sequence for this machine
        """
        # 1. Determine sequence length
        seq_length = np.random.randint(self.min_steps, self.max_steps + 1)
        
        # 2. Check if machine fails
        will_fail = (final_state['target'] == 1)
        
        # 3. Determine failure point (last 5-12% of sequence)
        if will_fail:
            failure_window = int(seq_length * 0.08)  # 8% of sequence
            failure_step = seq_length - np.random.randint(3, max(4, failure_window))
        else:
            failure_step = None
        
        # 4. Initialize arrays
        sequence_data = {
            'product_id': [],
            'unit_id': [],
            'timestamp': [],
            'engine_type': [],
            'air_temperature_K': [],
            'process_temperature_K': [],
            'rotational_speed_rpm': [],
            'torque_Nm': [],
            'tool_wear_min': [],
            'failure_type': [],
            'is_failure': [],
            'synthetic_RUL': [],
            'step_index': []
        }
        
        # 5. Generate each time-step
        for t in range(seq_length):
            progress = t / (seq_length - 1)  # 0.0 to 1.0
            
            # Degradation curve (exponential acceleration)
            degradation_factor = self._degradation_curve(progress)
            
            # 5a. Tool wear (monotonic increase)
            tool_wear = self._generate_tool_wear(
                final_state['tool_wear'], 
                progress, 
                degradation_factor
            )
            
            # 5b. Rotational speed (drift + noise)
            rpm = self._generate_rpm(
                final_state['rpm'],
                progress,
                t,
                will_fail
            )
            
            # 5c. Air temperature
            air_temp = self._generate_air_temperature(
                final_state['air_temp'],
                progress
            )
            
            # 5d. Process temperature
            process_temp = self._generate_process_temperature(
                final_state['process_temp'],
                progress,
                will_fail
            )
            
            # 5e. Torque
            torque = self._generate_torque(
                final_state['torque_nm'],
                progress,
                t,
                will_fail,
                failure_step
            )
            
            # 5f. Failure labels
            if will_fail and failure_step is not None and t >= failure_step:
                is_failure = 1
                failure_type = final_state['failure_type']
            else:
                is_failure = 0
                failure_type = 'No Failure'
            
            # 5g. RUL (Remaining Useful Life)
            if will_fail and failure_step is not None:
                rul = max(0, failure_step - t)
            else:
                rul = seq_length - t - 1  # Healthy machine
            
            # 5h. Timestamp (backward from final state)
            hours_back = (seq_length - t - 1) * self.time_delta_hours
            timestamp = pd.to_datetime(final_state['timestamp']) - timedelta(hours=hours_back)
            
            # Append to sequence
            sequence_data['product_id'].append(final_state['product_id'])
            sequence_data['unit_id'].append(final_state['unit_id'])
            sequence_data['timestamp'].append(timestamp)
            sequence_data['engine_type'].append(final_state['engine_type'])
            sequence_data['air_temperature_K'].append(air_temp)
            sequence_data['process_temperature_K'].append(process_temp)
            sequence_data['rotational_speed_rpm'].append(rpm)
            sequence_data['torque_Nm'].append(torque)
            sequence_data['tool_wear_min'].append(tool_wear)
            sequence_data['failure_type'].append(failure_type)
            sequence_data['is_failure'].append(is_failure)
            sequence_data['synthetic_RUL'].append(rul)
            sequence_data['step_index'].append(t)
        
        return pd.DataFrame(sequence_data)
    
    def _degradation_curve(self, progress):
        """
        Exponential degradation curve (accelerates near end).
        
        Returns value between 0.0 (start) and 1.0 (end) with exponential growth.
        """
        return progress ** 1.5  # Quadratic-ish acceleration
    
    def _generate_tool_wear(self, final_value, progress, degradation_factor):
        """Generate tool wear (monotonic increase)."""
        # Start at 30% of final value
        start_value = final_value * 0.3
        
        # Use degradation_factor which incorporates progress
        base_wear = start_value + (final_value - start_value) * degradation_factor
        
        # Add small noise
        noise = np.random.normal(0, 0.5)
        
        return max(0, base_wear + noise)
    
    def _generate_rpm(self, final_value, progress, t, will_fail):
        """Generate rotational speed with drift and oscillation."""
        # Drift factor (slight decrease over time for failing machines)
        drift = 0.03 * progress if will_fail else 0.01 * progress
        
        # Periodic oscillation (simulates operational cycles)
        oscillation = 5 * np.sin(2 * np.pi * t / 10)
        
        # Random noise
        noise = np.random.normal(0, 2)
        
        rpm = final_value * (1 - drift) + oscillation + noise
        
        return max(0, rpm)
    
    def _generate_air_temperature(self, final_value, progress):
        """Generate air temperature (gradual increase with noise)."""
        # Start slightly cooler
        start_value = final_value - 3
        
        # Linear interpolation
        base_temp = start_value + (final_value - start_value) * progress
        
        # Add noise
        noise = np.random.normal(0, 0.3)
        
        return base_temp + noise
    
    def _generate_process_temperature(self, final_value, progress, will_fail):
        """Generate process temperature (increases more for failing machines)."""
        # Start cooler
        temp_delta = 5 if will_fail else 3
        start_value = final_value - temp_delta
        
        # Linear interpolation
        base_temp = start_value + (final_value - start_value) * progress
        
        # Add noise
        noise = np.random.normal(0, 0.5)
        
        return base_temp + noise
    
    def _generate_torque(self, final_value, progress, t, will_fail, failure_step):
        """Generate torque (spikes near failure)."""
        # Base torque with small oscillation and slight progression
        oscillation = np.random.normal(0, 0.02)
        progression_factor = 1 + (0.05 * progress) if will_fail else 1
        base_torque = final_value * (1 + oscillation) * progression_factor
        
        # Stress factor near failure
        if will_fail and failure_step is not None and t >= failure_step:
            stress_factor = 1.1 + 0.05 * ((t - failure_step) / 5)  # Gradual increase
            base_torque *= stress_factor
        
        return max(0, base_torque)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("SYNTHETIC TIME-SERIES DATASET GENERATOR FOR AC-02")
    print("=" * 70)
    
    # 1. Load original snapshot dataset
    input_file = 'preprocessed_data/timestamped_predictive_maintenance.csv'
    print(f"\n1. Loading original dataset: {input_file}")
    
    df_original = pd.read_csv(input_file)
    print(f"   ✓ Loaded {len(df_original)} machines")
    print(f"   ✓ Columns: {list(df_original.columns)}")
    
    # 2. Initialize generator
    print("\n2. Initializing Time-Series Generator")
    generator = TimeSeriesGenerator(
        min_steps=60,
        max_steps=80,
        time_delta_hours=1,
        random_seed=42
    )
    
    # 3. Generate synthetic time-series
    print("\n3. Generating synthetic sequences...")
    df_timeseries = generator.generate_sequences(df_original)
    
    # 4. Validate output
    print("\n4. Validation Results:")
    print(f"   ✓ Total observations: {len(df_timeseries)}")
    print(f"   ✓ Unique machines: {df_timeseries['product_id'].nunique()}")
    print(f"   ✓ Failure observations: {df_timeseries['is_failure'].sum()}")
    print(f"   ✓ Failure rate: {df_timeseries['is_failure'].mean():.2%}")
    
    # Check monotonicity
    monotonic_check = df_timeseries.groupby('product_id').apply(
        lambda x: x['timestamp'].is_monotonic_increasing
    ).all()
    print(f"   ✓ Timestamps monotonic: {monotonic_check}")
    
    # Check RUL
    rul_check = df_timeseries.groupby('product_id').apply(
        lambda x: x['synthetic_RUL'].is_monotonic_decreasing
    ).all()
    print(f"   ✓ RUL decreasing: {rul_check}")
    
    # 5. Save to CSV
    output_file = 'preprocessed_data/timestamped_predictive_maintenance_timeseries_synthetic.csv'
    print(f"\n5. Saving synthetic dataset: {output_file}")
    df_timeseries.to_csv(output_file, index=False)
    print(f"   ✓ Dataset saved successfully!")
    
    # 6. Display sample
    print("\n6. Sample Output (First Machine):")
    sample_machine = df_timeseries[df_timeseries['product_id'] == df_timeseries['product_id'].iloc[0]]
    print(sample_machine[['timestamp', 'tool_wear_min', 'synthetic_RUL', 'is_failure']].head(10))
    print("   ...")
    print(sample_machine[['timestamp', 'tool_wear_min', 'synthetic_RUL', 'is_failure']].tail(5))
    
    print("\n" + "=" * 70)
    print("✓ SYNTHETIC DATASET GENERATION COMPLETE")
    print("=" * 70)
    
    return df_timeseries


if __name__ == "__main__":
    df_synthetic = main()