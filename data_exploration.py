"""
Quick data exploration script for Drone Telemetry Tampering Dataset v2
"""

import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Dataset path
DATA_DIR = Path("data/drone_temparing_dataset_v2")

def analyze_dataset():
    """Analyze the entire dataset structure and statistics"""
    
    stats = {
        'profiles': {},
        'total_cases': 0,
        'total_rows': 0,
        'anomaly_counts': defaultdict(int),
        'normal_flights': []
    }
    
    profiles = ['balanced', 'strong', 'subtle']
    
    for profile in profiles:
        profile_stats = {
            'replicates': 4,
            'cases_per_replicate': 0,
            'total_rows': 0,
            'anomaly_types': defaultdict(int)
        }
        
        # Analyze rep_00 as representative
        rep_path = DATA_DIR / profile / 'rep_00' / 'cases'
        
        if not rep_path.exists():
            continue
            
        cases = list(rep_path.iterdir())
        profile_stats['cases_per_replicate'] = len(cases)
        
        for case_dir in cases[:5]:  # Sample first 5 cases
            if not case_dir.is_dir():
                continue
                
            # Read metadata
            meta_file = case_dir / 'case_meta.json'
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                    
                anomaly_type = meta['tamper']['type']
                profile_stats['anomaly_types'][anomaly_type] += 1
                stats['anomaly_counts'][anomaly_type] += 1
                
                # Count rows
                csv_file = case_dir / 'decoded_flightlog.csv'
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    profile_stats['total_rows'] += len(df)
                    stats['total_rows'] += len(df)
                    
                    # Track normal flights
                    if anomaly_type == 'normal':
                        stats['normal_flights'].append({
                            'profile': profile,
                            'case': case_dir.name,
                            'rows': len(df),
                            'chunks_20': len(df) // 20
                        })
        
        stats['profiles'][profile] = profile_stats
        stats['total_cases'] += profile_stats['cases_per_replicate'] * 4
    
    return stats

def inspect_sample_case():
    """Inspect a single case in detail"""
    
    sample_normal = DATA_DIR / 'balanced' / 'rep_00' / 'cases' / 'case_0001_normal'
    sample_injection = DATA_DIR / 'balanced' / 'rep_00' / 'cases' / 'case_0000_injection'
    
    print("=" * 60)
    print("SAMPLE NORMAL FLIGHT")
    print("=" * 60)
    
    # Load normal flight
    df_normal = pd.read_csv(sample_normal / 'decoded_flightlog.csv')
    labels_normal = pd.read_csv(sample_normal / 'labels.csv')
    
    print(f"\nShape: {df_normal.shape}")
    print(f"\nColumns: {list(df_normal.columns)}")
    print(f"\nFirst 3 rows:")
    print(df_normal.head(3))
    print(f"\nData types:")
    print(df_normal.dtypes)
    print(f"\nBasic stats:")
    print(df_normal.describe())
    print(f"\nLabel distribution: {labels_normal['label'].value_counts().to_dict()}")
    
    print("\n" + "=" * 60)
    print("SAMPLE INJECTION ATTACK (GPS Spoofing)")
    print("=" * 60)
    
    # Load injection case
    df_injection = pd.read_csv(sample_injection / 'decoded_flightlog.csv')
    labels_injection = pd.read_csv(sample_injection / 'labels.csv')
    
    with open(sample_injection / 'case_meta.json') as f:
        meta = json.load(f)
    
    print(f"\nShape: {df_injection.shape}")
    print(f"\nMetadata: {json.dumps(meta, indent=2)}")
    print(f"\nLabel distribution: {labels_injection['label'].value_counts().to_dict()}")
    print(f"\nAnomaly rate: {labels_injection['label'].sum() / len(labels_injection) * 100:.2f}%")
    
    # Find anomalous regions
    anomalous_indices = labels_injection[labels_injection['label'] == 1].index.tolist()
    if anomalous_indices:
        print(f"\nFirst anomalous waypoint: {anomalous_indices[0]}")
        print(f"Last anomalous waypoint: {anomalous_indices[-1]}")
        print(f"\nNormal data at index {anomalous_indices[0] - 1}:")
        print(df_injection.iloc[anomalous_indices[0] - 1])
        print(f"\nAnomalous data at index {anomalous_indices[0]}:")
        print(df_injection.iloc[anomalous_indices[0]])

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DRONE TELEMETRY TAMPERING DATASET - ANALYSIS")
    print("=" * 60 + "\n")
    
    # Overall statistics
    stats = analyze_dataset()
    
    print("Overall Dataset Statistics:")
    print(f"  Total Cases: {stats['total_cases']}")
    print(f"  Total Rows (sampled): {stats['total_rows']:,}")
    print(f"\nAnomalies by Type (from sample):")
    for anom_type, count in sorted(stats['anomaly_counts'].items(), key=lambda x: -x[1]):
        print(f"  {anom_type:25s}: {count}")
    
    print(f"\nNormal Flights Found: {len(stats['normal_flights'])}")
    if stats['normal_flights']:
        total_normal_chunks = sum(f['chunks_20'] for f in stats['normal_flights'])
        print(f"  Total chunks (20-waypoint): {total_normal_chunks:,}")
    
    print(f"\n{'Profile':<15} {'Cases/Rep':<12} {'Total Cases'}")
    print("-" * 45)
    for profile, data in stats['profiles'].items():
        print(f"{profile:<15} {data['cases_per_replicate']:<12} {data['cases_per_replicate'] * 4}")
    
    # Detailed inspection
    print("\n")
    inspect_sample_case()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
