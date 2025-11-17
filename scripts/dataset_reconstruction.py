"""
Purpose: Create balanced subset with stratified sampling preserving distributions
Input: data/predictive_maintenance.csv (full dataset)
Output: preprocessed_data/predictive_maintenance_subset_400.csv

Pipeline Structure:
1. Data Loading & Configuration
2. Feature Engineering (Operating Buckets)
3. Stratified Sampling (Failures & Non-Failures)
4. Subset Assembly & Validation
5. Quality Checks & Export
"""

# Imports
import os
import numpy as np
import pandas as pd

from typing import Tuple

os.makedirs('preprocessed', exist_ok=True)

# Data loading & configuration functions
def load_data(path: str) -> pd.DataFrame:
    """Load the predictive maintenance dataset and perform initial validation."""
    df = pd.read_csv(path)
    
    print("[INFO] Loading & Initial Validation")
    print(f"[INFO] Total Rows Loaded: {len(df)}")
    print(f"[INFO] Unique Product IDs: {df['Product ID'].nunique()}")
    
    # Check required columns
    required_cols = [
        'UDI', 'Product ID', 'Type', 
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'Target', 'Failure Type'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] Missing Required Columns: {missing_cols}")
    
    # Validate 1 row per machine
    assert df['Product ID'].nunique() == len(df), \
        "[ERROR] Dataset Should Have Exactly 1 Row Per Unique Product ID"
    
    # Count failures
    n_fail = (df['Target'] == 1).sum()
    n_non_fail = (df['Target'] == 0).sum()
    print(f"[INFO] Failures (Target=1): {n_fail}")
    print(f"[INFO] Non-Failures (Target=0): {n_non_fail}")
    print(f"[INFO] Failure Rate: {n_fail / len(df):.4f}")
    print()
    
    return df

# Feature engineering functions
def add_operating_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add torque_level and rpm_band buckets based on quantiles."""
    df = df.copy()
    
    # Torque level
    q33_torque = df['Torque [Nm]'].quantile(0.33)
    q66_torque = df['Torque [Nm]'].quantile(0.66)
    
    def torque_bucket(val):
        if val <= q33_torque:
            return 'low'
        elif val <= q66_torque:
            return 'mid'
        else:
            return 'high'
    
    df['torque_level'] = df['Torque [Nm]'].apply(torque_bucket)
    
    # RPM band
    q33_rpm = df['Rotational speed [rpm]'].quantile(0.33)
    q66_rpm = df['Rotational speed [rpm]'].quantile(0.66)
    
    def rpm_bucket(val):
        if val <= q33_rpm:
            return 'low'
        elif val <= q66_rpm:
            return 'mid'
        else:
            return 'high'
    
    df['rpm_band'] = df['Rotational speed [rpm]'].apply(rpm_bucket)
    
    print("[INFO] Operating Condition Buckets Created")
    print(f"[INFO] Torque Quantiles: q33={q33_torque:.2f}, q66={q66_torque:.2f}")
    print(f"[INFO] RPM Quantiles: q33={q33_rpm:.0f}, q66={q66_rpm:.0f}")
    print()
    
    return df

# Stratified sampling functions
def sample_failure_machines(df: pd.DataFrame, n_fail_target: int, random_state: int) -> pd.DataFrame:
    """Sample failure machines stratified by Failure Type."""
    df_fail = df[df['Target'] == 1].copy()
    
    # Exclude "No Failure" from failures
    df_fail = df_fail[df_fail['Failure Type'] != 'No Failure'].copy()
    
    print("[INFO] Sampling Failure Machines")
    print(f"[INFO] Original Failure Count: {len(df_fail)}")
    print("\n[INFO] Original Failure Type Distribution:")
    print(df_fail['Failure Type'].value_counts().sort_index())
    print("\n[INFO] Normalized:")
    print(df_fail['Failure Type'].value_counts(normalize=True).sort_index())
    
    # Compute stratified sample sizes
    failure_counts = df_fail['Failure Type'].value_counts()
    total_failures = len(df_fail)
    
    sampled_dfs = []
    allocated = 0
    
    for failure_type, count in failure_counts.items():
        proportion = count / total_failures
        target_n = int(round(proportion * n_fail_target))
        
        # Clip to available
        target_n = min(target_n, count)
        allocated += target_n
        
        group_df = df_fail[df_fail['Failure Type'] == failure_type]
        sampled = group_df.sample(n=target_n, random_state=random_state, replace=False)
        sampled_dfs.append(sampled)
    
    # Adjust if rounding caused mismatch
    if allocated < n_fail_target:
        diff = n_fail_target - allocated
        remaining = df_fail[~df_fail.index.isin(pd.concat(sampled_dfs).index)]
        if len(remaining) > 0:
            extra = remaining.sample(n=min(diff, len(remaining)), random_state=random_state, replace=False)
            sampled_dfs.append(extra)
    elif allocated > n_fail_target:
        # Remove excess from largest group
        combined = pd.concat(sampled_dfs)
        combined = combined.sample(n=n_fail_target, random_state=random_state, replace=False)
        sampled_dfs = [combined]
    
    df_fail_sampled = pd.concat(sampled_dfs, axis=0)
    
    print(f"\n[INFO] Target Failure Sample Size: {n_fail_target}")
    print(f"[INFO] Actual Sampled: {len(df_fail_sampled)}")
    print("\n[INFO] Sampled Failure Type Distribution:")
    print(df_fail_sampled['Failure Type'].value_counts().sort_index())
    print("\n[INFO] Normalized:")
    print(df_fail_sampled['Failure Type'].value_counts(normalize=True).sort_index())
    print()
    
    return df_fail_sampled


def sample_non_failure_machines(df: pd.DataFrame, n_non_fail_target: int, random_state: int) -> pd.DataFrame:
    """Sample non-failure machines stratified by Type, torque_level, rpm_band."""
    df_non_fail = df[df['Target'] == 0].copy()
    
    print("[INFO] Sampling Non-Failure Machines")
    print(f"[INFO] Original Non-Failure Count: {len(df_non_fail)}")
    
    # Group by stratification keys
    strat_cols = ['Type', 'torque_level', 'rpm_band']
    grouped = df_non_fail.groupby(strat_cols)
    
    print("\n[INFO] Original Distribution by (Type, torque_level, rpm_band):")
    print(grouped.size())
    
    # Compute proportions and target sizes
    group_counts = grouped.size()
    total_non_fail = len(df_non_fail)
    
    sampled_dfs = []
    allocated = 0
    
    for group_key, count in group_counts.items():
        proportion = count / total_non_fail
        target_n = int(round(proportion * n_non_fail_target))
        
        # Clip to available
        target_n = min(target_n, count)
        
        if target_n > 0:
            group_df = df_non_fail.loc[df_non_fail[strat_cols].apply(tuple, axis=1) == group_key]
            sampled = group_df.sample(n=target_n, random_state=random_state, replace=False)
            sampled_dfs.append(sampled)
            allocated += target_n
    
    # Top up if needed
    if allocated < n_non_fail_target:
        diff = n_non_fail_target - allocated
        already_sampled = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()
        remaining = df_non_fail[~df_non_fail.index.isin(already_sampled.index)]
        
        if len(remaining) > 0:
            extra = remaining.sample(n=min(diff, len(remaining)), random_state=random_state, replace=False)
            sampled_dfs.append(extra)
            allocated += len(extra)
    
    # If still over (unlikely), trim
    if allocated > n_non_fail_target:
        combined = pd.concat(sampled_dfs)
        combined = combined.sample(n=n_non_fail_target, random_state=random_state, replace=False)
        sampled_dfs = [combined]
    
    df_non_fail_sampled = pd.concat(sampled_dfs, axis=0)
    
    print(f"\n[INFO] Target Non-Failure Sample Size: {n_non_fail_target}")
    print(f"[INFO] Actual Sampled: {len(df_non_fail_sampled)}")
    print("\n[INFO] Sampled Distribution by (Type, torque_level, rpm_band):")
    print(df_non_fail_sampled.groupby(strat_cols).size())
    print()
    
    return df_non_fail_sampled

# Subset assembly & validation
def build_subset(df_fail: pd.DataFrame, df_non_fail: pd.DataFrame, total_target: int) -> pd.DataFrame:
    """Combine failure and non-failure samples into final subset."""
    df_subset = pd.concat([df_fail, df_non_fail], axis=0)
    
    # Drop duplicates on Product ID (safety check)
    df_subset = df_subset.drop_duplicates(subset='Product ID')
    
    # Validate
    assert len(df_subset) == total_target, \
        f"Expected {total_target} rows, got {len(df_subset)}"
    assert df_subset['Product ID'].nunique() == total_target, \
        f"Expected {total_target} unique Product IDs, got {df_subset['Product ID'].nunique()}"
    
    # Shuffle
    df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("[RESULT] Final Subset Summary")
    print(f"Total Rows: {len(df_subset)}")
    print(f"Unique Product IDs: {df_subset['Product ID'].nunique()}")
    print(f"Failures: {(df_subset['Target'] == 1).sum()}")
    print(f"Non-Failures: {(df_subset['Target'] == 0).sum()}")
    print()
    
    return df_subset


def validate_subset(df_full: pd.DataFrame, df_subset: pd.DataFrame) -> None:
    """Perform comprehensive validation EDA on the subset."""
    print("[INFO] Validation EDA: Subset Quality Checks")
    
    # 1. Class balance check
    print("\n[INFO] Class Balance Check")
    failure_rate_full = (df_full['Target'] == 1).sum() / len(df_full)
    failure_rate_subset = (df_subset['Target'] == 1).sum() / len(df_subset)
    
    print(f"[INFO] Failure Rate in Full Dataset: {failure_rate_full:.4f} ({failure_rate_full*100:.2f}%)")
    print(f"[INFO] Failure Rate in Subset: {failure_rate_subset:.4f} ({failure_rate_subset*100:.2f}%)")
    
    if failure_rate_subset < 0.10 or failure_rate_subset > 0.30:
        print("\n[WARN] Subset Failure Rate is Outside the 10%-30% Range")
        print("[WARN] This May Indicate Too Much Imbalance for Certain Use Cases")
    else:
        print("\n[RESULT] Subset Failure Rate is Within Acceptable Range (10%-30%)")
    
    # 2. Failure Type distribution
    print("\n[INFO] Failure Type Distribution")
    df_full_fail = df_full[df_full['Target'] == 1]
    df_subset_fail = df_subset[df_subset['Target'] == 1]
    
    print("\n[INFO] Full Dataset (Failures Only):")
    print(df_full_fail['Failure Type'].value_counts().sort_index())
    print("\n[INFO] Normalized:")
    print(df_full_fail['Failure Type'].value_counts(normalize=True).sort_index())
    
    print("\n[INFO] Subset (Failures Only):")
    print(df_subset_fail['Failure Type'].value_counts().sort_index())
    print("\n[INFO] Normalized:")
    print(df_subset_fail['Failure Type'].value_counts(normalize=True).sort_index())
    
    # 3. Operating condition distributions
    print("\n[INFO] Operating Condition Distributions")
    
    for col in ['Type', 'torque_level', 'rpm_band']:
        print(f"\n[FEATURES] {col}")
        full_dist = df_full[col].value_counts(normalize=True).sort_index()
        subset_dist = df_subset[col].value_counts(normalize=True).sort_index()
        
        print("[INFO] Full Dataset:")
        print(full_dist)
        print("\n[INFO] Subset:")
        print(subset_dist)
        
        print("\n[INFO] Absolute Percentage Difference:")
        for cat in full_dist.index:
            full_pct = full_dist.get(cat, 0) * 100
            subset_pct = subset_dist.get(cat, 0) * 100
            diff = abs(full_pct - subset_pct)
            print(f"{cat}: {diff:.2f}%")
    
    # 4. Sensor statistics
    print("\n[INFO] Sensor Statistics Comparison")
    
    sensors = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    for sensor in sensors:
        print(f"\n[FEATURES] {sensor}")
        full_mean = df_full[sensor].mean()
        full_std = df_full[sensor].std()
        full_min = df_full[sensor].min()
        full_max = df_full[sensor].max()
        
        subset_mean = df_subset[sensor].mean()
        subset_std = df_subset[sensor].std()
        subset_min = df_subset[sensor].min()
        subset_max = df_subset[sensor].max()
        
        print(f"[INFO] Full:   Mean: {full_mean:.2f}, Std: {full_std:.2f}, Min: {full_min:.2f}, Max: {full_max:.2f}")
        print(f"[INFO] Subset: Mean: {subset_mean:.2f}, Std: {subset_std:.2f}, Min: {subset_min:.2f}, Max: {subset_max:.2f}")
        
        mean_diff = abs(subset_mean - full_mean)
        if mean_diff > 2 * full_std:
            print(f"[WARN] Subset Mean Differs by {mean_diff:.2f} ( > {2*full_std:.2f}, 2 x Std)")
        else:
            print(f"[RESULT] Subset Mean Within Acceptable Range (diff={mean_diff:.2f})")
    
    # 5. Missing values and range checks
    print("\n[INFO] Data Quality Checks")
    
    missing_count = df_subset.isnull().sum().sum()
    print(f"Missing Values in Subset: {missing_count}")
    if missing_count > 0:
        print("[WARN] Subset Contains Missing Values")
    else:
        print("[RESULT] No Missing Values in Subset")
    
    print("\n[INFO] Sensor Range Validation:")
    for sensor in sensors:
        full_min = df_full[sensor].min()
        full_max = df_full[sensor].max()
        subset_min = df_subset[sensor].min()
        subset_max = df_subset[sensor].max()
        
        if subset_min < full_min or subset_max > full_max:
            print(f"[WARN] {sensor} Subset Range [{subset_min:.2f}, {subset_max:.2f}] Exceeds Full Range [{full_min:.2f}, {full_max:.2f}]")
        else:
            print(f"[RESULT] {sensor} Range Within Full Dataset Bounds")
    

# Quality checks & export
def save_subset(df: pd.DataFrame, path: str) -> None:
    """Save the subset to CSV and print final summary."""
    df.to_csv(path, index=False)
    
    print("\n[INFO] Subset Saved")
    print(f"Saved to: {path}")
    print(f"Total Rows: {len(df)}")
    print(f"Failures: {(df['Target'] == 1).sum()}")
    print(f"Non-Failures: {(df['Target'] == 0).sum()}")
    print(f"Failure Rate: {(df['Target'] == 1).sum() / len(df):.4f}")
    
    print("\n[RESULT] Type Distribution:")
    print(df['Type'].value_counts().sort_index())
    
    print("\n[RESULT] Torque Level Distribution:")
    print(df['torque_level'].value_counts().sort_index())
    
    print("\n[RESULT] RPM Band Distribution:")
    print(df['rpm_band'].value_counts().sort_index())
    
    print("\n[INFO] Script Completed Successfully")

# MAIN EXECUTION PIPELINE
def main():
    # Configuration
    input_path = 'data/predictive_maintenance.csv'
    output_path = 'preprocessed/predictive_maintenance_subset_400_machines.csv'
    n_total = 400
    target_failure_rate = 0.2
    random_state = 42
    
    # Step 1: Load Full Dataset
    df = load_data(input_path)
    
    # Step 2: Add Operating Buckets
    df = add_operating_buckets(df)
    
    # Step 3: Compute Target Sample Sizes
    n_fail_available = (df['Target'] == 1).sum()
    n_fail_target = min(int(target_failure_rate * n_total), n_fail_available)
    n_non_fail_target = n_total - n_fail_target
    
    print("[RESULT] Sampling Configuration:")
    print(f"Total Target Size: {n_total}")
    print(f"Target Failure Rate: {target_failure_rate:.2f}")
    print(f"Available Failures: {n_fail_available}")
    print(f"Target Failures to Sample: {n_fail_target}")
    print(f"Target Non-Failures to Sample: {n_non_fail_target}")
    print()
    
    # Step 4: Sample Failure Machines (Stratified by Failure Type)
    df_fail_sampled = sample_failure_machines(df, n_fail_target, random_state)
    
    # Step 5: Sample Non-Failure Machines (Stratified by Operating Conditions)
    df_non_fail_sampled = sample_non_failure_machines(df, n_non_fail_target, random_state)
    
    # Step 6: Build Final Subset
    df_subset = build_subset(df_fail_sampled, df_non_fail_sampled, n_total)
    
    # Step 7: Validate Subset Quality
    validate_subset(df, df_subset)
    
    # Step 8: Save Subset to CSV
    save_subset(df_subset, output_path)

if __name__ == "__main__":
    main()