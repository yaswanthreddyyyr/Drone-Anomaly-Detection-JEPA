#!/usr/bin/env python3
"""
JEPA-DRONE Data Preprocessing Script

Run this script to preprocess the raw drone telemetry data and create
train/validation/test splits for the JEPA model.

Usage:
    python scripts/preprocess_data.py [--config CONFIG_PATH]
    
Example:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import run_preprocessing_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess drone telemetry data for JEPA-DRONE"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    JEPA-DRONE                             ║
    ║     Self-Supervised Drone Anomaly Detection               ║
    ║                                                           ║
    ║              DATA PREPROCESSING PIPELINE                  ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"📁 Config: {config_path}")
    print(f"🚀 Starting preprocessing pipeline...\n")
    
    try:
        run_preprocessing_pipeline(str(config_path))
        print("\n✅ Preprocessing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
