# Provide highly accurate approximations of ocean surface currents using satellite data

### Objective:
Develop a predictive model to forecast ocean surface currents (specifically, the geostrophic components derived from altimetry data) with enough accuracy to capture day-to-day changes. This forecast should ideally capture both the spatial and temporal dynamics of ocean currents in the European Seas.

### Background:
Data Source: Satellite altimetry products (e.g., SLA, ugosa, vgosa) that provide gridded fields of sea surface height and derived velocities.
Challenge: The data are spatially distributed and temporally evolving. Predicting the changes in ocean currents involves modeling complex physical processes such as geostrophic balance, mesoscale eddies, and possibly submesoscale features.


### Dataset description
https://data.marine.copernicus.eu/product/SEALEVEL_EUR_PHY_L4_NRT_008_060/description

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

- This dataset contain 1060 sample, 1 snapshot/day (uniform temporal sampling), from 2022-01-01 to 2024-11-25.
-  Resolution: 0.125° grid (~14 km at equator)
- Domain: Northeast Atlantic & Mediterranean Sea
- Land Mask: NaN values indicate land/ice

### Evaluation

For a forecast task of ocean surface currents, consider the following evaluation metrics:

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



### References
- [Machine-Learning Mesoscale and Submesoscale Surface Dynamics from Lagrangian Ocean Drifter Trajectories
](https://journals.ametsoc.org/view/journals/phoc/50/5/jpo-d-19-0238.1.xml)

- [DriftNet](https://github.com/CIA-Oceanix/DriftNet/tree/main)
- [Relative Fluid Stretching and Rotation for Sparse Trajectory Observations](https://github.com/EncinasBartos/Relative-Fluid-Stretching-and-Rotation-for-Sparse-Trajectory-Observations/tree/main)


### LLM assisted research in this project
- [ChatGPT](https://chatgpt.com/c/67bbd338-9018-8000-8782-5aa83f300419)
- [Perplexity](https://www.perplexity.ai/search/machine-learning-mesoscale-and-dZCx0OEqTq6vwdVF1setPA)
- [DeepSeek](https://chat.deepseek.com/a/chat/s/01c04727-fa4c-46bc-be87-1a39b82f07cd)
- [Grok](https://x.com/i/grok?focus=1&conversation=1893844088757367096)
- [Claude](https://claude.ai/chat/c5360fea-d742-45b3-9cf8-d5bddec36238)
