# Flusight-CU-Ensemble
Implementation of the CU-Ensemble model for the Flusight forecasting hub

# Description:
This is the code for Columbia University model ensemble for forecasting flu. 
The ensemble is currenlty composed of five models - four statistical models and one dynamical model.
The ensemble employs two statistical models implemented within the python library darts: 
1) Holt Winter’s Exponential Smoothing (ES), a classical statistical model that decomposes a time series to a baseline, trend and seasonal components,
2) Light Gradient Boosting Machine (LightGBM), a ML ensemble decision-tree method designed for classification and regression tasks that has been effectively adapted for time series forecasting, and

A third statistical model included in the ensemble is the 'seasonal_drift' model, a random drift model where the mean and variance of the drift for each week and location are derived by sampling first-order lags in incidence from historical data (transformed ILI data and past hospitalization data). Sampling is performed using a window around the current week of the year (e.g., during epiweek 47, the model samples from a window around epiweek 47 in previous years).

The fourth statistical model employed in the ensemble is a new deep-learning model - details and code availability TBD.

The ensemble also incorporates an SIR model together with an Ensemble Adjusted Kalman Filter (EAKF) procedure to assimilate the model with available observed data before generating predictions.

To build the ensemble, the predictions of the component models are weighted using the sum of inverse WIS scores over past weeks, with greater importance given to more recent weeks. The period for calculating weights is horizon-specific and includes only the weeks where WIS scores could be evaluated (e.g., weights for the 4-week horizon are calculated using a longer historical window than those for the 1-week horizon). The weights are location-specific and are recomputed for each forecast week.

Peak week distribution and incidence are currently forecasted using a combination of the seasonal_drift model and the SIR+EAKF model.
