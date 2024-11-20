# Flusight-CU-Ensemble
Implementation of the CU-Ensemble model for the Flusight forecasting hub

# Description:
These is the code for the python-based components of the ensemble (not including the dynamical SEIRS model implementation). We employ three statistical models implemented within the python library darts: 
1) Holt Winter’s Exponential Smoothing (ES), a classical statistical model that decomposes a time series to a baseline, trend and seasonal components,
2) Light Gradient Boosting Machine (LightGBM), a ML ensemble decision tree method designed for classification and regression tasks has been effectively adapted for time series forecasting, and
3) Temporal Fusion Transformer (TFT) - a transformer-based neural-network architecture tailored for time series forecasting.
Past years ILI data is transformed to resemble hospitalization data and is used to train the models. Labratory data is used as covariate with model fitting and predicitons. To build the ensemble, the quantile distributions of the component models are weighted by the sum of inverse-WIS scores, over last 4 weeks. The 4-week window is target-specific and only includes weeks for which WIS scores could be evaluated (i.e. weights for 4-wk target are calculated with a window further back in time than for 1-wk target). Weights are location-specific and recomputed at each forecast week. Peak week distribution and incidence is currently being forecasted using historical stats gathered from the combination of transformed ILI data and hospitalization data. Peak week distribution is smoothed using non-parameteric kernel density estimation (KDE). In the near future, we intend to enhance these forecasts using the SEIRS model forecasts and forecasts of statistical models trained on these peak week targets. 
