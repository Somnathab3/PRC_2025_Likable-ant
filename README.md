# PRC 2025 Likable-ant Fuel Prediction Pipeline

This repository contains the code for the fuel prediction pipeline developed for the PRC 2025 competition.

## Pipeline Stages

The pipeline consists of the following stages:

1. **Stage 1: Rebuild** (`Stage_1_Rebuild.py`) - Unified trajectory processing pipeline that combines cleaning, resampling, gap reconstruction, and smoothing.

2. **Stage 2: ERA5 Features** (`Stage_2_ERA5_Features.py`) - Adds ERA5 weather features to the trajectories.

3. **Stage 3: Filter TAS** (`Stage_3_Filter_TAS.py`) - Filters true airspeed (TAS) data.

4. **Stage 4: TOW** (`Stage_4_TOW.py`) - Calculates takeoff weight (TOW).

5. **Stage 5: Fuel Calculation** (`Stage_5_Fuel_Calculation.py`) - Performs fuel calculations.

6. **Stage 5.1: Acropole Only with KNN** (`Stage_5.1_Acropole_Only_with_KNN.py`) - Uses Acropole model with KNN imputation.

7. **Stage 5.2: Add OpenAP to Acropole** (`Stage_5.2_Add_OpenAP_to_Acropole.py`) - Integrates OpenAP with Acropole.

8. **Stage 6.1: Regenerate Training Data** (`Stage_6.1_Regenerate_training_data.py`) - Regenerates training data.

9. **Stage 6.2: Train Multiplier** (`Stage_6.2_Train_Multiplier.py`) - Trains the multiplier model.

10. **Stage 6.3: Create Multiplier Trajectory** (`Stage_6.3_Create_Multiplier_Trajectory.py`) - Creates multiplier trajectories.

11. **Stage 7: Feature Engineering Multiplier** (`Stage_7_Feature_Engineering_Multiplier.py`) - Performs feature engineering for multipliers.

12. **Stage 8: Imputation** (`Stage_8_Imputation.py`, `Stage_8_Tow_Merge_Imputation_LightGBM.py`, `Stage_8_Imputation_LightGBM.py`) - Handles data imputation using various methods.

13. **Stage 9: Fuel Prediction** (`Stage_9_Fuel_Prediction.py`, `Stage_9.1_Fuel_pred_Catboost.py`, `Stage_9_Fuel_predict_CV.py`) - Final fuel prediction using machine learning models like CatBoost.

## Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

Or using Poetry:

```bash
poetry install
```

## Usage

Run the stages in order. Each stage processes data and prepares it for the next stage.

## Folders

- `traffic/`: Traffic data processing library
- `Acropole/`: Acropole fuel prediction library
- `openap/`: OpenAP library for aircraft performance
- `scripts/`: Pipeline scripts