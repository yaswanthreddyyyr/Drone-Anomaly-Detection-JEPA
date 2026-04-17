*📊 Dataset Summary*

- 720 total flight cases (240 per difficulty: balanced, strong, subtle)
- ~420,000 chunks total (at 20 waypoints/chunk)
- 10 anomaly types including GPS injection attacks
- Perfect ground truth labels for validation
- Real DJI flight logs with telemetry data

*✅ Why This Dataset is Perfect*

- Real-world data - Actual DJI drone telemetry, not synthetic
- GPS spoofing included - 54+ injection attack cases for zero-shot testing
- Multiple difficulty levels - Can show robustness (strong) and sensitivity (subtle)
- Sufficient normal data - ~50K clean chunks for JEPA self-supervised training
- Comprehensive validation - All 10 anomaly types with binary labels

*📁 Files Created*

- DATA_PLAN.md - Complete data split strategy for your 12-week timeline
- DATASET_SUMMARY.md - Dataset structure and statistics
- data_exploration.py - Analysis script (already run)

*🎯 Recommended Split*

- Training: Balanced profile (rep_00-01) - normal chunks only
- Validation: Balanced (rep_02) - 60 cases for hyperparameter tuning
- Testing: Balanced (rep_03) + All Strong + All Subtle - ~540 cases for final results
- This gives you everything you need to hit your targets: AUC-ROC >0.90, Recall >85%, FAR <5% 🎯