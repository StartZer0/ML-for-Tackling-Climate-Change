# Flash Drought Prediction README

## Overview
This repository contains code and documentation for predicting flash droughts using machine learning models, specifically tailored for a 14x14 grid over the Midwest U.S. from 2013 to 2023. The project leverages normalized climate data and implements a GraphCast model for forecasting meteorological variables, alongside a heuristic formula to detect flash droughts.

## Files and Structure
- **Data Preparation**: 
  - `drought_train_normalized_imp.nc`: Training dataset (2013-2019, 2556 time steps).
  - `drought_val_normalized_imp.nc`: Validation dataset (2020-2022, 1096 time steps).
  - `drought_test_normalized_imp.nc`: Test dataset (2023, 365 time steps).
  - Normalization parameters: Stored in `/climatology` (e.g., `era5_mean.csv`, `lra5_sigma.csv`).

- **Model Implementation**:
  - `GraphCast.ipynb`: Implements a Graph Neural Network (GNN) combining convolutional and graph layers for forecasting 10 variables over 14, 30, or 44 days from a 60-day input.
  - `FuXi-S2S`
  - `UNet`
  - `UNet+LSTM`
  - `FlashDroughtFormula.ipynb`: Defines a heuristic dryness index and flash drought detection logic.


## Prerequisites
- **Python**: 3.8+
- **Libraries**: 
  ```bash
  pip install torch torch-geometric xarray pandas numpy matplotlib tqdm netCDF4 scikit-learn
  ```
- **Hardware**: GPU recommended (e.g., NVIDIA L4 used in development).

## Usage
1. **Data Setup**:
   - Place `.nc` files in `/drought_data/processed/`.
   - Ensure climatology files are in `/climatology/`.

2. **Training the Model**:
   - Run `GraphCast.ipynb`:
     ```python
     train_file = '/path/to/drought_train_normalized_imp.nc'
     val_file = '/path/to/drought_val_normalized_imp.nc'
     train_dataset = ClimateDataset(train_file)
     val_dataset = ClimateDataset(val_file)
     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
     val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
     model = GraphCast().to(device)
     train_model(model, train_loader, val_loader, num_epochs=30)
     ```
   - Outputs `best_model.pth` with the lowest validation loss.

3. **Evaluating RMSE**:
   - Compute denormalized RMSE:
     ```python
     calculate_denormalized_rmse(model, val_loader, '/path/to/climatology')
     ```

4. **Flash Drought Detection**:
   - Use `FlashDroughtFormula.ipynb` to process predictions:
     ```python
     df_processed, flash_summary = detect_flash_droughts_all_locations(df)
     ```

## Key Features
- **GraphCast Model**: Combines spatial graph convolutions and temporal attention for forecasting.
- **Variables**: Predicts `z500`, `t850`, `e`, `evavt`, `lai_lv`, `pev`, `swvl2`, `swvl3`, `t2m`, `tp`.
- **Flash Drought Index**: Heuristic formula based on soil moisture, evaporation, precipitation, temperature, potential evaporation, geopotential height, and leaf area index.

## Contributing
- Fork the repository, make changes, and submit a pull request.
- Issues and suggestions welcome via GitHub Issues.

## License
- MIT License (see `LICENSE` file for details).