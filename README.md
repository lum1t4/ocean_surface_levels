# OSL - Ocean Surface Currents Forecasting

**Objective**: Develop a predictive model to forecast ocean surface currents with enough accuracy to capture day-to-day changes. This forecast should ideally capture both the spatial and temporal dynamics of ocean currents in the European Seas.


## Getting started

### Install
This project uses `uv` to handle dependencies so you can install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Official documentation can be found here: [UV docs](https://docs.astral.sh/uv/getting-started/installation/)

After installing uv, you can create a new environment with the following command `uv sync` in the root of the project. This will create a new environment with all the dependencies specified in the `pyproject.toml` file than you can activate the env as any python virtual env.

### Data
To fetch the dataset, this project uses dvc to handle data and its processing pipeline. In summary to get everything up and running you need just to `dvc repro`

### Project Organization
```bash
├── README.md          <- The top-level README for developers using this project.
├── LICENSE            <- The license for the project
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── pyproject.toml     <- Project configuration file with package metadata
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── osl   <- Source code for use in this project.
│
└── dvc.yaml       <- DVC pipeline file that defines the stages of the pipeline
│
└── dvc.lock       <- DVC lock file that stores the state of the pipelineì
```

## Background:
Data Source: Satellite altimetry products (e.g., SLA, ugosa, vgosa) that provide gridded fields of sea surface height and derived velocities.
Challenge: The data are spatially distributed and temporally evolving. Predicting the changes in ocean currents involves modeling complex physical processes such as geostrophic balance, mesoscale eddies, and possibly submesoscale features.


### Dataset description
[cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.0625deg_P1D](https://data.marine.copernicus.eu/product/SEALEVEL_EUR_PHY_L4_NRT_008_060/description)

- **Dimensions & Coordinates**
  - **time:** The time coordinate (with a daily or near-daily sampling as indicated by the “time_coverage_resolution” of P1D) over which the altimetry measurements are made.
  - **latitude & longitude:** These spatial dimensions indicate the geographical grid covering European Seas. Their bounds (“lat_bnds” and “lon_bnds”) provide the exact cell edges.
  - **nv:** This is a secondary index used for representing cell bounds (commonly of size 2 for lower and upper bounds).

- **Data Variables**
  - **crs:** The coordinate reference system variable that gives information about the projection.
  - **lat_bnds & lon_bnds:** The boundaries of each grid cell in latitude and longitude.
  - **sla (Sea Level Anomaly):** The deviation of the measured sea surface height from a long-term mean. This is a key variable used to derive ocean current estimates.
  - **err_sla:** The associated uncertainty of the SLA measurements.
  - **ugosa & vgosa:** The geostrophic velocity components computed from SLA. Here, **ugosa** is the eastward (zonal) component, and **vgosa** is the northward (meridional) component. Their errors are provided in **err_ugosa** and **err_vgosa**.
  - **adt:** Likely representing Absolute Dynamic Topography (the sea surface height relative to the geoid).
  - **ugos & vgos:** These may be adjusted or smoothed versions of the geostrophic velocities.
  - **flag_ice:** A flag variable indicating the presence (or absence) of sea ice (typically 0 indicating no ice).

- **Attributes**
  - The attributes provide metadata such as the spatial resolution, temporal coverage, data source (satellite altimetry), creator details, and conventions followed. This dataset is part of the Copernicus Marine Environment Monitoring Service (CMEMS) products and has been processed to Level-4 (gridded, interpolated fields).

### Dataset notes
- Current data is available in NetCDF format, which is a common choice for storing multi-dimensional scientific data. You can use libraries like `xarray` or `netCDF4` in Python to read and manipulate this data.

- This dataset contain 11487 sample, 1 snapshot/day (uniform temporal sampling), from 1993-01-01 to 2024-06-13.
-  Resolution: 0.0625° grid
- Domain: Northeast Atlantic & Mediterranean Sea
- Land Mask: NaN values indicate land/ice

### Evaluation

For a forecast task of ocean surface currents the follwing metrics could be used to evaluate the model performance:

- Root Mean Squared Error (RMSE): Measures the average magnitude of the error between predicted and observed current components.
- Mean Absolute Error (MAE): Provides a robust measure of average error magnitude, less sensitive to outliers than RMSE.
- Correlation Coefficient: Assesses how well the predicted and observed time series correlate, capturing the model’s ability to track temporal variations.

- Vector Error Metrics:
  - Directional Error: Compare the angle (direction) of the predicted velocity vector versus the observed one.
  - Magnitude Error: Evaluate errors in the speed (magnitude) of the velocity vectors.
  - Vector RMSE: Compute RMSE for the combined vector field (possibly using Euclidean norm differences).

- Skill Scores:
  - Anomaly Correlation Coefficient (ACC): Especially useful when predicting anomalies (deviations from climatology) rather than absolute values.
  - Relative Improvement Metrics: Compare the performance of your model against a baseline (such as persistence or climatology).

- Spatial Metrics (if forecasting spatial fields):
  - Spatial Correlation: To assess how well the spatial patterns in the prediction match the observations.
  - Pattern Correlation and Mean Bias Error: To evaluate any systematic spatial discrepancies.
 
- Spectral Score: Energy at different spatial scales
- Eddy Detection Rate: Using SLA-based vortex detection
- Trajectory Simulation: Compare virtual drifter paths


## References
- [Machine-Learning Mesoscale and Submesoscale Surface Dynamics from Lagrangian Ocean Drifter Trajectories
](https://journals.ametsoc.org/view/journals/phoc/50/5/jpo-d-19-0238.1.xml)

- [DriftNet - A novel fully-convolutional architecture inspired from the Eulerian Fokker-Planck representation of Lagrangian dynamics at sea surface](https://github.com/CIA-Oceanix/DriftNet/tree/main)
- [Relative Fluid Stretching and Rotation for Sparse Trajectory Observations](https://github.com/EncinasBartos/Relative-Fluid-Stretching-and-Rotation-for-Sparse-Trajectory-Observations/tree/main)
- [A Deep Framework for Eddy Detection and Tracking From Satellite Sea Surface Height Data (2020)](https://ieeexplore.ieee.org/document/9247537)
- [Neural Network Training for the Detection and Classification of Oceanic Mesoscale Eddies (2020)](https://www.mdpi.com/2072-4292/12/16/2625)
- [A lightweight deep learning model for ocean eddy detection (2023).](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1266452/full)
- [Multicore structures and the splitting and merging of eddies in global oceans from satellite altimeter data (2019)](https://os.copernicus.org/articles/15/413/2019/)
- [EddyGraph: The Tracking of Mesoscale Eddy Splitting and Merging Events in the Northwest Pacific Ocean (2021)](https://www.mdpi.com/2072-4292/13/17/3435)


## Glossary

- **Mesosclae**: Mid-size oceanic features, typically ranging from 10 to 100 km in diameter, such as eddies, currents, and fronts
- **Submesoscale**: Smaller oceanic features, typically ranging from 1 to 10 km, that can influence the dynamics of larger mesoscale features.
