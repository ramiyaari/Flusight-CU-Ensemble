import numpy as np
import pandas as pd
from datetime import timedelta

import torch
from darts.models import ExponentialSmoothing, LightGBMModel,TFTModel #, NHiTSModel, BlockRNNModel, TransformerModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.utils import ModelMode, SeasonalityMode

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
#from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle, quantile_loss
from darts.utils.missing_values import fill_missing_values

import darts.utils.likelihood_models as Likelihood
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import *

import warnings
import os

# print(darts.__version__)
# print(torch.cuda.is_available())

# Pytorch early stopping rules
my_stopper = EarlyStopping(
    monitor="train_loss",  # Over which values we are optimizing
    patience=10,           # After how many iterations if the loss doesn't improve then it stops optimizing
    min_delta=0.001,       # Round-off error to consider that it didn't improved
    mode="min"
)

cuda_available = torch.cuda.is_available()
print("GPU available: {}".format(cuda_available))
if(cuda_available):
    # device = 'gpu'
    pl_trainer_kwargs1={"callbacks": [my_stopper], "accelerator": 'gpu', "devices": -1}
else:
    # device = 'cpu'
    pl_trainer_kwargs1 = {"callbacks": [my_stopper], "accelerator": "cpu"}


#Return the defined lists of local and global forecasting models
def get_darts_models(quantiles): 

    # List of local models
    model_list_local = {
        
        # # "ExponentialSmoothing": ExponentialSmoothing(
        # #     seasonal_periods=52, trend=ModelMode.ADDITIVE, damped=True,seasonal=SeasonalityMode.MULTIPLICATIVE),
        
        "ExponentialSmoothing": ExponentialSmoothing(
                   seasonal_periods=52, seasonal=SeasonalityMode.ADDITIVE),

        "LightGBM": LightGBMModel(
                lags=[-1,-2,-53], #
                lags_past_covariates=2, 
                lags_future_covariates=[2,2], 
                output_chunk_length=1,
                likelihood="quantile",
                quantiles=quantiles,
                add_encoders={"cyclic": {"past": ["weekofyear","month"],
                                        "future": ["weekofyear","month"]},
                                        'transformer': Scaler() },
                show_warnings=False,
                verbose=-1
            ),

        "TFT": TFTModel(
                input_chunk_length=52, 
                output_chunk_length=1,
                likelihood=Likelihood.QuantileRegression(quantiles=quantiles),
                n_epochs=200,
                batch_size=32,
                dropout=0.2, 
                optimizer_kwargs={'lr': 1e-3}, 
                pl_trainer_kwargs={"callbacks": [my_stopper], "accelerator": "auto", "devices": -1},
                add_encoders={"cyclic": {"past": ["weekofyear","month"],
                                        "future": ["weekofyear","month"]},
                                        'transformer': Scaler() }),

        # "NHiTS": NHiTSModel(
        #         input_chunk_length=52,
        #         output_chunk_length=1,
        #         likelihood=Likelihood.QuantileRegression(quantiles=quantiles),
        #         n_epochs=200,
        #         batch_size=32,
        #         num_stacks=2, num_blocks=2, num_layers=2, 
        #         dropout=0.2,
        #         optimizer_kwargs={'lr': 1e-3}, 
        #         pl_trainer_kwargs=pl_trainer_kwargs1,
        #         add_encoders={"cyclic": {"past": ["weekofyear","month"]},'transformer': Scaler() },
        #         show_warnings=False,
        #     ),

    }
    # List of global models
    model_list_global = {
        # "LightGBM_G": LightGBMModel(
        #         lags=[-1,-2,-53], #
        #         lags_past_covariates=2, 
        #         lags_future_covariates=[2,2], 
        #         output_chunk_length=1,
        #         likelihood="quantile",
        #         quantiles=quantiles,
        #         # add_encoders={"cyclic": {"past": ["weekofyear","month"]},'transformer': Scaler() },
        #         add_encoders={"cyclic": {"past": ["weekofyear","month"],
        #                                 "future": ["weekofyear","month"]},
        #                                 'transformer': Scaler() },
        #         show_warnings=False,
        #         verbose=-1
        #     ),
        # "NHiTS_G": NHiTSModel(
        #         input_chunk_length=52,
        #         output_chunk_length=1,
        #         likelihood=Likelihood.QuantileRegression(quantiles=quantiles),
        #         n_epochs=200,
        #         batch_size=32,
        #         num_stacks=2, num_blocks=2, num_layers=2, 
        #         dropout=0.2,
        #         optimizer_kwargs={'lr': 1e-3}, 
        #         pl_trainer_kwargs=pl_trainer_kwargs1,
        #         add_encoders={"cyclic": {"past": ["weekofyear","month"]},'transformer': Scaler() },
        #         show_warnings=False,
        #     ),
        # "TFT_G": TFTModel(
        #         input_chunk_length=52, 
        #         output_chunk_length=1,
        #         likelihood=Likelihood.QuantileRegression(quantiles=quantiles),
        #         n_epochs=200,
        #         batch_size=32,
        #         dropout=0.2, 
        #         optimizer_kwargs={'lr': 1e-3}, 
        #         pl_trainer_kwargs={"callbacks": [my_stopper], "accelerator": "auto", "devices": -1},
        #         add_encoders={"cyclic": {"past": ["weekofyear","month"],
        #                                 "future": ["weekofyear","month"]},
        #                                 'transformer': Scaler() })
    }
    return (model_list_local, model_list_global)


#Return a TimeSeries object for all given states
def get_states_timeseries(df, states):
    series = TimeSeries.from_dataframe(df, 
                                       time_col='date', 
                                       value_cols=states,
                                       fill_missing_dates=True, 
                                       freq="W-SAT")
    series = fill_missing_values(series)
    series = series.astype(np.float32)  
    return series


def fit_and_predict(series, model, model_desc, pred_start_date, weeks_to_predict, num_samples,
                    series_past_covar, series_future_covar, pred_only=False):
    
    last_series_time = series.time_index[series.n_timesteps-1]
    if(last_series_time < pred_start_date):
        train = series
        if(last_series_time + timedelta(weeks=1) != pred_start_date):
            print('missing values between end of training data and prediction start time...')
    else:
        train, _ = series.split_before(pred_start_date)

    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)

    if(series_past_covar is not None):
        if(np.isnan(series_past_covar.values()).all()):
            series_past_covar = None
        else:
            series_past_covar = Scaler().fit_transform(series_past_covar)
    if(series_future_covar is not None):
        if(np.isnan(series_future_covar.values()).all()):
            series_future_covar = None
        else:
            series_future_covar = Scaler().fit_transform(series_future_covar)

    try:
        if((not model.supports_past_covariates) and (not model.supports_future_covariates)):
            if(not pred_only):
                model.fit(train_transformed)
            pred = model.predict(weeks_to_predict,num_samples=num_samples)
        elif((model.supports_past_covariates) and (not model.supports_future_covariates)):
            if(not pred_only):
                model.fit(train_transformed,past_covariates=series_past_covar)
            pred = model.predict(weeks_to_predict,past_covariates=series_past_covar,num_samples=num_samples)
        elif((not model.supports_past_covariates) and (model.supports_future_covariates)):
            if(not pred_only):
                model.fit(train_transformed,future_covariates=series_future_covar)
            pred = model.predict(weeks_to_predict,future_covariates=series_future_covar,num_samples=num_samples)
        else:
            if(not pred_only):
                model.fit(train_transformed,past_covariates=series_past_covar,future_covariates=series_future_covar) 
            pred = model.predict(weeks_to_predict, past_covariates=series_past_covar,future_covariates=series_future_covar,num_samples=num_samples)

        pred = transformer.inverse_transform(pred)
        pred = pred.map(lambda x: np.clip(x,0,np.inf)) 

    except (ValueError, TypeError) as err:
        warnings.warn(
            ("Unable to run model {}." +
                "\nThrows error {}").format(model_desc, err))
    
    return (pred)


def fit_and_predict_univariate(df, states, model, model_desc, pred_start_date, weeks_to_predict, num_samples, 
                               fit_single_model, df_past_covar=None, df_future_covar=None):

    pred_all = None
    series_all = get_states_timeseries(df, states)
    if(df_past_covar is not None):
        series_past_covar_all = get_states_timeseries(df_past_covar, states)
    if(df_future_covar is not None):
        series_future_covar_all = get_states_timeseries(df_future_covar, states)

    
    if(not fit_single_model):
        # fit and predict each state separately
        for state in states:
            print("-----------state: {}-----------".format(state))
            series = series_all[state]
            series_past_covar = series_past_covar_all[state] if df_past_covar is not None else None
            series_future_covar = series_future_covar_all[state] if df_future_covar is not None else None
            model = model.untrained_model()
            pred = fit_and_predict(series, model, model_desc, pred_start_date, weeks_to_predict, num_samples,
                                series_past_covar, series_future_covar)
            if(pred_all is None):
                pred_all = pred
            else:
                pred_all = pred_all.stack(pred)
    else: 
        # fit_single_model
        # first fit all states then predict them
        for state in states:
            print("----------- fitting state: {}-----------".format(state))
            series = series_all[state]
            series_past_covar = series_past_covar_all[state] if df_past_covar is not None else None
            series_future_covar = series_future_covar_all[state] if df_future_covar is not None else None
            fit_and_predict(series, model, model_desc, pred_start_date, weeks_to_predict, num_samples,
                            series_past_covar, series_future_covar)
            
        for state in states:
            print("----------- predicting state: {}-----------".format(state))
            series = series_all[state]
            series_past_covar = series_past_covar_all[state] if df_past_covar is not None else None
            series_future_covar = series_future_covar_all[state] if df_future_covar is not None else None
            pred = fit_and_predict(series, model, model_desc, pred_start_date, weeks_to_predict, num_samples,
                                    series_past_covar, series_future_covar, pred_only=True)
            pred = pred.with_columns_renamed(col_names=pred.components, col_names_new=[state])
            if(pred_all is None):
                pred_all = pred
            else:
                pred_all = pred_all.stack(pred)

    return (pred_all)
    

def fit_and_predict_multivariate(df, states, model, model_desc, pred_start_date, weeks_to_predict, num_samples, 
                                 df_past_covar=None, df_future_covar=None):

    series = get_states_timeseries(df, states)
    series_past_covar = None
    series_future_covar = None
    if(df_past_covar is not None):
        series_past_covar = get_states_timeseries(df_past_covar, states)
    if(df_future_covar is not None):
        series_future_covar = get_states_timeseries(df_future_covar, states)
    
    pred = fit_and_predict(series, model, model_desc, pred_start_date, weeks_to_predict, num_samples,
                           series_past_covar, series_future_covar)
    return pred


def get_quantiles_df(pred, quantiles):
    quantiles_df = pred.quantile_df(quantiles[0]).clip(lower=0)
    for quantile in quantiles[1:]:
        quantiles_df = quantiles_df.merge(pred.quantile_df(quantile).clip(lower=0),on="date")
    return (quantiles_df)


def save_darts_pred_results_to_file(pred, ref_date, weeks_to_predict, locations, quantiles, 
                                    dat_changerate_ref, basedir, model_desc):
   
    pred_results = []
    # horizons = ((pred.time_index-ref_date).days.values/7).astype(int)
    horizons = range(weeks_to_predict)
    locations_abbr = locations.index #pred.components
    for loc_abbr in locations_abbr:
        location = locations.loc[loc_abbr].location
        pred_state = pred[loc_abbr]

        pred_quantiles = get_quantiles_df(pred_state,quantiles)
        for hind, horizon in enumerate(horizons):
            for qind, quantile in enumerate(quantiles):
                pred_results.append([format(ref_date,'%Y-%m-%d'),'wk inc flu hosp', horizon,
                          format(ref_date + timedelta(weeks=horizon),'%Y-%m-%d'),
                          location, 'quantile', np.round(quantile,3), pred_quantiles.iloc[hind,qind]])


    locations_abbr = locations.index #pred.components
    for loc_abbr in locations_abbr:
        location = locations.loc[loc_abbr].location
        pop_size = locations.loc[loc_abbr].population
        ref_val = dat_changerate_ref[loc_abbr].values[0]
        for horizon in horizons:
            pred_vals = pred[ref_date+timedelta(weeks=horizon)][loc_abbr].all_values()[0][0]
            pred_qual = generate_qualtitative_pred(ref_date, location, horizon, pred_vals, ref_val, pop_size)
            pred_results = pred_results+pred_qual
            
    df_pred_results = pd.DataFrame(pred_results,columns=['reference_date','target','horizon','target_end_date',
                                                         'location','output_type','output_type_id','value']) 
    
    df_pred_results['location'] = df_pred_results['location'].astype(str).str.zfill(2)
    
    output_dir = "{}/{}".format(basedir, model_desc)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    df_pred_results.to_csv("{}/{}-CU-{}.csv".format(output_dir,format(ref_date,'%Y-%m-%d'),model_desc), index=False) 
    return df_pred_results