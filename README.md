# Flusight-CU-Ensemble
Implementation of the CU-Ensemble model for the Flusight forecasting hub

# Description:
This is the code for Columbia University model ensemble for forecasting flu. 
The ensemble is currenlty composed of five models - four statistical models and one dynamical model.
The ensemble employs three statistical models implemented within the python library darts: 
1) Holt Winter’s Exponential Smoothing (ES), a classical statistical model that decomposes a time series to a baseline, trend and seasonal components,
2) Light Gradient Boosting Machine (LightGBM), a ML ensemble decision-tree method designed for classification and regression tasks that has been effectively adapted for time series forecasting, and
3) Temporal Fusion Transformer (TFT) - a transformer-based neural-network architecture tailored for time series forecasting.
Past years ILI data is transformed to resemble hospitalization data and is used to train the models. Labratory data is used as covariate with model fitting and predicitons. 

Another statistical model included in the ensemble is the 'historical_drift' model, a random drift model where the mean and variance of the drift for each week and location are derived by sampling first-order lags in incidence from historical data (transformed ILI data and past hospitalization data). Sampling is performed using a window around the current week of the year (e.g., during epiweek 47, the model samples from a window around epiweek 47 in previous years).

The ensemble incorporates an SIR model that leverages location-specific daily, annual-averaged absolute humidity data to modulate R0. Additionally, it employs an Ensemble Adjusted Kalman Filter (EAKF) procedure to assimilate the model with available observed data before generating predictions.

To build the ensemble, the predictions of the component models are weighted using the sum of inverse WIS scores over past weeks, with greater importance given to more recent weeks. The period for calculating weights is horizon-specific and includes only the weeks where WIS scores could be evaluated (e.g., weights for the 4-week horizon are calculated using a longer historical window than those for the 1-week horizon). The weights are location-specific and are recomputed for each forecast week.

Peak week distribution and incidence are currently being forecasted using historical stats gathered from the combination of transformed ILI data and hospitalization data. Peak week distribution is smoothed using non-parameteric kernel density estimation (KDE). In the near future, we intend to enhance these forecasts using the SIR model forecasts and forecasts of statistical models trained on these peak week targets. 
