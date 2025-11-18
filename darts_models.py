import numpy as np
import pandas as pd
from datetime import timedelta

import torch
# from darts.models import ExponentialSmoothing, LightGBMModel,TFTModel #, NHiTSModel, BlockRNNModel, TransformerModel
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.lgbm import LightGBMModel
# from darts.models.forecasting.tft_model import TFTModel
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
        
        "ExponentialSmoothing": ExponentialSmoothing(
            seasonal_periods=52,seasonal=SeasonalityMode.MULTIPLICATIVE),
        
        # "ExponentialSmoothing_additive": ExponentialSmoothing(
        #            seasonal_periods=52, seasonal=SeasonalityMode.ADDITIVE),

        "LightGBM": LightGBMModel(
                lags=[-1,-2,-52], #
                lags_past_covariates=1,
                lags_future_covariates=(0, 1),
                output_chunk_length=1,
                likelihood="quantile",
                quantiles=quantiles,
                add_encoders={"cyclic": {"past": ["weekofyear","month"],
                                        "future": ["weekofyear","month"]} },
                n_jobs=-1,                       # parallelize across quantiles/series
                random_state=42,
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                min_data_in_leaf=100,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                lambda_l2=1.0,
                multi_models=False,              
                show_warnings=False,
                verbose=-1,
            ),
    }
    # List of global models
    model_list_global = {
        # "LightGBM_G": LightGBMModel(
        #         lags=[-1,-2,-52], #
        #         lags_past_covariates=1,
        #         lags_future_covariates=(0, 1),
        #         output_chunk_length=1,
        #         likelihood="quantile",
        #         quantiles=quantiles,
        #         add_encoders={"cyclic": {"past": ["weekofyear","month"],
        #                                 "future": ["weekofyear","month"]} },
        #         random_state=42,
        #         n_estimators=1000,
        #         learning_rate=0.05,
        #         num_leaves=31,
        #         min_data_in_leaf=100,
        #         feature_fraction=0.8,
        #         bagging_fraction=0.8,
        #         bagging_freq=1,
        #         lambda_l2=1.0,
        #         n_jobs=-1,
        #         show_warnings=False,
        #         verbose=-1,
        #     ),
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


def _is_all_nan(ts):
    try:
        vals = ts.values(copy=False)
        return np.isnan(vals).all()
    except Exception:
        # if dtype/object weirdness
        vals = ts.values()
        return np.all(np.isnan(vals))


def fit_and_predict(series, model, model_desc, pred_start_date, 
                    weeks_to_predict, num_samples,
                    series_past_covar=None, series_future_covar=None):

    # ------------------------
    # 1) Define train window
    # ------------------------
    last_series_time = series.time_index[-1]
    if last_series_time < pred_start_date:
        train = series
        if last_series_time + timedelta(weeks=1) != pred_start_date:
            print('missing values between end of training data and prediction start time...')
    else:
        # split_before: right part starts at pred_start_date
        train, _ = series.split_before(pred_start_date)

    # ------------------------
    # 2) Fit scalers on TRAIN ONLY
    # ------------------------
    y_scaler = Scaler()
    train_y = y_scaler.fit_transform(train)+1
    # train_y = y_scaler.fit_transform(train)

    past_scaler = None
    fut_scaler = None

    # Build train-slices for covariates (fit) and full slices (predict)
    if series_past_covar is not None:
        if _is_all_nan(series_past_covar):
            series_past_covar = None
        else:
            past_scaler = Scaler()
            past_covar_train = series_past_covar.slice(start_ts=series_past_covar.start_time(),
                                                       end_ts=pred_start_date - timedelta(weeks=1))
            past_scaler.fit(past_covar_train)
            series_past_covar = past_scaler.transform(series_past_covar)

    if series_future_covar is not None:
        if _is_all_nan(series_future_covar):
            series_future_covar = None
        else:
            fut_scaler = Scaler()
            # fit scaler only on history up to the start of prediction
            fut_covar_hist = series_future_covar.slice(start_ts=series_future_covar.start_time(),
                                                       end_ts=pred_start_date - timedelta(weeks=1))
            fut_scaler.fit(fut_covar_hist)
            series_future_covar = fut_scaler.transform(series_future_covar)
            # soft check that horizon is covered
            fut_last = series_future_covar.time_index[-1]
            need_until = pred_start_date + timedelta(weeks=weeks_to_predict-1)
            if fut_last < need_until:
                warnings.warn(
                    f"[{model_desc}] future covariates end at {fut_last.date()}, "
                    f"but need at least {need_until.date()} for horizon={weeks_to_predict}."
                )

    # For fit: use covariates up to pred_start_date-1 week
    past_cov_fit = None
    fut_cov_fit = None
    if series_past_covar is not None:
        past_cov_fit = series_past_covar.slice(start_ts=series_past_covar.start_time(),
                                               end_ts=pred_start_date - timedelta(weeks=1))
    if series_future_covar is not None:
        fut_cov_fit = series_future_covar.slice(start_ts=series_future_covar.start_time(),
                                                end_ts=pred_start_date - timedelta(weeks=1))

    pred = None
    try:
        # ------------------------
        # 3) Fit 
        # ------------------------
        if (not model.supports_past_covariates) and (not model.supports_future_covariates):
            model.fit(train_y)
        elif model.supports_past_covariates and (not model.supports_future_covariates):
            model.fit(train_y, past_covariates=past_cov_fit)
        elif (not model.supports_past_covariates) and model.supports_future_covariates:
            model.fit(train_y, future_covariates=fut_cov_fit)
        else:
            model.fit(train_y, past_covariates=past_cov_fit, future_covariates=fut_cov_fit)

        # ------------------------
        # 4) Predict (pass full scaled covariates; Darts slices internally)
        # ------------------------
        if (not model.supports_past_covariates) and (not model.supports_future_covariates):
            pred = model.predict(weeks_to_predict, num_samples=num_samples)
        elif model.supports_past_covariates and (not model.supports_future_covariates):
            pred = model.predict(weeks_to_predict, past_covariates=series_past_covar, num_samples=num_samples)
        elif (not model.supports_past_covariates) and model.supports_future_covariates:
            pred = model.predict(weeks_to_predict, future_covariates=series_future_covar, num_samples=num_samples)
        else:
            pred = model.predict(
                weeks_to_predict,
                past_covariates=series_past_covar,
                future_covariates=series_future_covar,
                num_samples=num_samples
            )

        # ------------------------
        # 5) Invert scaling & post-process
        # ------------------------
        pred = y_scaler.inverse_transform(pred-1)
        # pred = y_scaler.inverse_transform(pred)
        pred = pred.map(lambda x: np.clip(x, 0, np.inf))

    except Exception as err:
        warnings.warn(f"Unable to run model {model_desc}. Error: {err}")
        return None

    return pred


def fit_and_predict_univariate(df, states, model, model_desc, pred_start_date, weeks_to_predict, num_samples, 
                               df_past_covar=None, df_future_covar=None, 
                               nb_inflate=True, df_prev_pred=None):

    pred_all = None
    series_all = get_states_timeseries(df, states)
    series_past_covar_all = get_states_timeseries(df_past_covar, states) if df_past_covar is not None else None
    series_future_covar_all = get_states_timeseries(df_future_covar, states) if df_future_covar is not None else None

    if(nb_inflate):
        df_truth = pd.melt(df,id_vars=['date','year','week'],value_vars=df.columns[3:],var_name='location',value_name='value')
        # init_nb_inflation_params(model_desc)
        nb_inflate_pars = update_nb_inflation_params(model_desc, df_prev_pred, df_truth, weeks_to_predict)
        # nb_alpha = nb_inflate_pars["alpha_global"] 
        # nb_horizon_scale = nb_inflate_pars["scales_h"]
        # print(f"nb_alpha={nb_alpha}")
        # print(f"nb_horizon_scale={nb_horizon_scale}")


              
    for state in states:
        # print(f"-----------state: {state}-----------")
        series = series_all[state]
        series_past_covar = None
        series_future_covar = None
        if(series_past_covar_all is not None):
            series_past_covar = series_past_covar_all[state]
        if(series_future_covar_all is not None):
            series_future_covar = series_future_covar_all[state]

        # Fresh estimator per state
        state_model = model.untrained_model()

        pred = fit_and_predict(
            series, state_model, model_desc, pred_start_date, weeks_to_predict, num_samples,
            series_past_covar, series_future_covar
        )

        if pred is None:
            continue

        if nb_inflate: # and getattr(pred, "n_samples", 1) > 1:
            pred = process_nb_inflation(pred, nb_inflate_pars)

        pred_all = pred if pred_all is None else pred_all.stack(pred)

    if pred_all is None:
        raise RuntimeError(f"No predictions produced for model '{model_desc}' (all states failed).")

    return pred_all


def fit_and_predict_multivariate(df, states, model, model_desc, pred_start_date, weeks_to_predict, num_samples, 
                                 df_past_covar=None, df_future_covar=None):

    series = get_states_timeseries(df, states)
    series_past_covar = get_states_timeseries(df_past_covar, states) if df_past_covar is not None else None
    series_future_covar = get_states_timeseries(df_future_covar, states) if df_future_covar is not None else None

    pred = fit_and_predict(
        series, model.untrained_model(), model_desc, pred_start_date, weeks_to_predict, num_samples,
        series_past_covar, series_future_covar
    )

    if pred is None:
        raise RuntimeError(f"Prediction failed for model '{model_desc}' in multivariate mode.")
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


#####################################################################################
# functions related to NB Inflation
import json, re, tempfile, shutil

def make_default_state(max_h: int,
                       alpha0: float = 0.15,
                       prior_scales: dict[int, float] | None = None):
    """
    Build a fresh state for horizons 0..max_h-1.
    prior_scales: optional {h: scale}; others get a gentle increasing prior like 1.0, 1.05, ...
    """
    if prior_scales is None:
        # mild widening with horizon as a neutral prior
        prior_scales = {h: 1.0 + 0.05*h for h in range(max_h)}
    scales_h = {str(h): float(prior_scales.get(h, 1.0)) for h in range(max_h)}
    cov_ema_by_h = {str(h): None for h in range(max_h)}
    return {
        "alpha_global": float(alpha0),
        "scales_h": scales_h,
        "cov_ema_by_h": cov_ema_by_h,
    }


def ensure_state_schema(state: dict | None, max_h: int,
                        alpha0: float = 0.15) -> dict:
    """
    Backfill/repair any missing fields to match horizons 0..max_h-1.
    Keeps existing values where present.
    """
    s = state.copy() if state else {}
    # alpha
    s["alpha_global"] = float(s.get("alpha_global", alpha0))
    # scales
    s_scales = {str(k): float(v) for k, v in s.get("scales_h", {}).items()}
    for h in range(max_h):
        s_scales.setdefault(str(h), 1.0 + 0.05*h)  # same neutral prior
    s["scales_h"] = s_scales
    # EMA
    s_ema = {str(k): (None if v is None else float(v))
             for k, v in s.get("cov_ema_by_h", {}).items()}
    for h in range(max_h):
        s_ema.setdefault(str(h), None)
    s["cov_ema_by_h"] = s_ema
    return s


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)

def load_nb_state(filepath: str, max_h : int):
    """
    Load state if present; otherwise return init defaults.
    Never raises FileNotFoundError.
    """
    try:
        with open(filepath, "r") as f:
            state = json.load(f)
    except FileNotFoundError:
        # first run: return defaults (caller may immediately save)
        state = make_default_state(max_h, alpha0=0.15)  # or load last season's final state
    except json.JSONDecodeError:
        # corrupted or empty file -> fall back to defaults
        state = make_default_state(max_h, alpha0=0.15)  # or load last season's final state
    return state

def save_nb_state(filepath: str, state: dict) -> None:
    """
    Create parent dir if missing and write atomically to avoid partial files.
    """
    ensure_parent_dir(filepath)
    # atomic write: write to temp then replace
    dirpath = os.path.dirname(os.path.abspath(filepath)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_nb_state_", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        # On Windows, use replace via shutil.move to handle cross-device cases
        shutil.move(tmp, filepath)
    finally:
        # If something failed before move, clean up tmp
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass

NB_LOG_PATH = "./nb_inflation_params"
NB_LOG_HIST = f"{NB_LOG_PATH}/nb_history.csv"
def append_nb_history(model, as_of_date, state, max_h, log_path=NB_LOG_HIST):
    """
    Append one row of NB state for horizons 0..max_h-1.
    Creates the folder/file if missing. Keeps it simple.
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)

    row = {
        "model": model,
        "as_of_date": pd.to_datetime(as_of_date),
        "alpha_global": float(state["alpha_global"]),
    }

    scales = state.get("scales_h", {})
    for h in range(max_h):
        v = scales.get(str(h), scales.get(h, np.nan))
        row[f"scale_h{h}"] = float(v) if v is not None else np.nan

    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(log_path)
    df_row.to_csv(log_path, mode="a", header=write_header, index=False)

# ---------- helpers ----------
_qnum_rx = re.compile(r"(\d+\.?\d*)")
def parse_quantile(x):
    """
    Returns float in [0,1] if x encodes a quantile; else np.nan.
    Accepts 0.05, "0.05", "q0.05", "quantile0.95", "p5", "5%", "50" (→0.5) etc.
    """
    if isinstance(x, (float, int)):
        x = float(x)
        if 0 <= x <= 1:
            return x
        if 1 < x <= 100:
            return x / 100.0
        return np.nan
    s = str(x).lower()
    m = _qnum_rx.search(s)
    if not m:
        return np.nan
    v = float(m.group(1))
    if "%" in s or v > 1:
        v = v / 100.0
    return v if 0 <= v <= 1 else np.nan

def nearest_col(cols, q_target, tol=1e-6):
    """Find the column in 'cols' closest to q_target within tol; else None."""
    diffs = [(abs(float(c) - q_target), c) for c in cols]
    diffs.sort()
    if diffs and diffs[0][0] <= tol:
        return diffs[0][1]
    return None

# ---------- coverage from results ----------
def compute_coverage_by_horizon(df_pred, df_truth, K=6, q_lo=0.05, q_hi=0.95,
                                loc_col="location", horizon_col="horizon",
                                date_pred_col="target_end_date", val_col="value",
                                date_truth_col="date", truth_val_col="value"):
    """
    Fast pooled coverage across ALL locations:
    - filters to {q_lo, q_hi} rows before pivot (tiny tables)
    - parses quantiles once per unique label
    - tails K per (loc,h) without re-sorting full tables
    """
    if (df_pred is None) or df_pred.empty or (df_truth is None) or df_truth.empty:
        return {}, {}

    # 1) Parse unique output_type_id once
    uq = pd.DataFrame({"output_type_id": df_pred["output_type_id"].unique()})
    uq["q"] = uq["output_type_id"].map(parse_quantile)  # uses the global helper you defined above
    uq = uq.dropna(subset=["q"])
    # Keep only rows near the two quantiles we need
    def nearest(target):
        diffs = (uq["q"] - float(target)).abs()
        return set(uq.loc[diffs == diffs.min(), "output_type_id"].tolist())

    keep_ids = nearest(q_lo) | nearest(q_hi)
    dfp = df_pred[df_pred["output_type_id"].isin(keep_ids)].copy()
    if dfp.empty:
        return {}, {}

    # 2) Map to numeric quantiles and pivot the tiny table
    q_map = uq.set_index("output_type_id")["q"].to_dict()
    dfp["q"] = dfp["output_type_id"].map(q_map)

    wide = (dfp.pivot_table(index=[loc_col, horizon_col, date_pred_col],
                            columns="q", values=val_col, aggfunc="mean").reset_index())
    wide.columns.name = None
    qcols = [c for c in wide.columns if isinstance(c, float)]
    if not qcols:
        return {}, {}

    # Pick exact/nearest numeric keys
    qcols_arr = np.array(qcols, dtype=float)
    lo_key = float(qcols_arr[np.argmin(np.abs(qcols_arr - float(q_lo)))])
    hi_key = float(qcols_arr[np.argmin(np.abs(qcols_arr - float(q_hi)))])

    # 3) Truth once per (loc,date)
    truth = (df_truth.rename(columns={date_truth_col: date_pred_col, truth_val_col: "truth"})
                    [[loc_col, date_pred_col, "truth"]])
    truth[date_pred_col] = pd.to_datetime(truth[date_pred_col])
    truth_1 = (truth.sort_values(date_pred_col)
                    .groupby([loc_col, date_pred_col], as_index=False)
                    .agg(truth=("truth", "last")))

    df = wide.merge(truth_1, on=[loc_col, date_pred_col], how="inner")
    if df.empty:
        return {}, {}

    # 4) Last K per (loc,h) (use groupby.tail without re-sorting big frames repeatedly)
    df = df.sort_values([loc_col, horizon_col, date_pred_col])
    tailK = df.groupby([loc_col, horizon_col]).tail(K)

    covered = (tailK[lo_key] <= tailK["truth"]) & (tailK["truth"] <= tailK[hi_key])
    cov_loc_h = covered.groupby([tailK[loc_col], tailK[horizon_col]]).mean()

    cov_by_h = cov_loc_h.groupby(level=1).mean().to_dict()
    counts_by_h = (covered.groupby([tailK[loc_col], tailK[horizon_col]]).size()
                   .groupby(level=1).sum().to_dict())
    return cov_by_h, counts_by_h


def update_nb_state(
    state: dict,
    cov_by_h: dict[int, float],          # keys: int horizons (0..max_h-1)
    counts_by_h: dict[int, int],         # keys: int horizons (0..max_h-1)
    max_h: int,
    target: float = 0.90,
    eta_alpha: float = 0.15,
    eta_scale_by_h: float | dict[int, float] = 0.25,  # scalar or per-h dict
    scale_clip: tuple[float, float] = (0.7, 1.5),
    alpha_clip: tuple[float, float] = (1e-4, 1.0),
    max_weekly_mult_change: float = 1.15,
    min_points_per_h: int = 4,
    ema_decay: float = 0.6,
    shrink_to_global: float = 0.0, #0.25,
    enforce_monotone: bool = True,
    monotone_respect_weekly_cap: bool = False,
):
    """
    Update alpha_global and scales_h for horizons 0..max_h-1.
    """
    horizons = list(range(max_h))
    state = ensure_state_schema(state, max_h)  # repair / fill defaults

    # coerce params
    if isinstance(eta_scale_by_h, (int, float)):
        eta_scale_map = {h: float(eta_scale_by_h) for h in horizons}
    else:
        # default to scalar 0.25 if a horizon is missing
        eta_scale_map = {h: float(eta_scale_by_h.get(h, 0.25)) for h in horizons}

    # read state (string keys in JSON)
    alpha = float(state["alpha_global"])
    scales_h = {int(k): float(v) for k, v in state["scales_h"].items()}
    cov_ema_raw = state.get("cov_ema_by_h", {})
    cov_ema = {h: (np.nan if cov_ema_raw.get(str(h)) is None else float(cov_ema_raw.get(str(h))))
               for h in horizons}

    # 1) EMA coverage per horizon
    new_cov_ema = {}
    for h in horizons:
        cov = cov_by_h.get(h, np.nan)
        prev = cov_ema.get(h, np.nan)
        if np.isnan(cov):
            new_cov_ema[h] = prev if not np.isnan(prev) else np.nan
        else:
            new_cov_ema[h] = cov if np.isnan(prev) else ema_decay*prev + (1.0-ema_decay)*cov

    # 2) global alpha update from available EMA
    valid_emas = [v for v in new_cov_ema.values() if not np.isnan(v)]
    if valid_emas:
        e_bar = target - float(np.mean(valid_emas))
        alpha = float(np.clip(np.exp(np.log(alpha) + eta_alpha*e_bar), *alpha_clip))

    # 3) per-horizon scales
    g_ref = float(np.median(list(scales_h.values())))  # for shrinkage
    new_scales = {}
    for h in horizons:
        s_old = scales_h[h]
        ema_cov = new_cov_ema.get(h, np.nan)
        n_pts = int(counts_by_h.get(h, 0))

        if (n_pts < min_points_per_h) or np.isnan(ema_cov):
            s_new = s_old
        else:
            e = target - ema_cov
            step = eta_scale_map[h] * e
            s_new_raw = float(np.exp(np.log(s_old) + step))

            # anti-windup vs absolute clip
            if (s_old >= scale_clip[1] and s_new_raw > s_old) or (s_old <= scale_clip[0] and s_new_raw < s_old):
                s_new = s_old
            else:
                # weekly change cap, then absolute clip
                s_cap_hi = s_old * max_weekly_mult_change
                s_cap_lo = s_old / max_weekly_mult_change
                s_new = float(np.clip(s_new_raw, s_cap_lo, s_cap_hi))
                s_new = float(np.clip(s_new, *scale_clip))

        # shrinkage toward pooled median scale
        if shrink_to_global > 0:
            s_new = float(np.exp((1.0 - shrink_to_global)*np.log(s_new) + shrink_to_global*np.log(g_ref)))

        new_scales[h] = s_new

    # 4) enforce non-decreasing scales with horizon (h0 ≤ h1 ≤ ...)
    if enforce_monotone and len(horizons) > 1:
        s_old = {h: scales_h[h] for h in horizons}  # for weekly-cap if needed
        seq = [new_scales[h] for h in horizons]
        for i in range(1, len(seq)):
            if seq[i] < seq[i-1]:
                bump = seq[i-1]
                if monotone_respect_weekly_cap:
                    cap_hi = s_old[horizons[i]] * max_weekly_mult_change
                    bump = min(bump, cap_hi)
                seq[i] = bump
        seq = [float(np.clip(v, *scale_clip)) for v in seq]
        for i, h in enumerate(horizons):
            new_scales[h] = seq[i]

    # 5) return JSON-safe state (string keys; NaN -> None for EMA)
    return {
        "alpha_global": alpha,
        "scales_h": {str(h): float(new_scales[h]) for h in horizons},
        "cov_ema_by_h": {str(h): (None if np.isnan(new_cov_ema[h]) else float(new_cov_ema[h])) for h in horizons},
    }


def init_nb_inflation_params(model): 
    filepath = f"{NB_LOG_PATH}/{model}.json"
    if os.path.exists(filepath):
            try: os.remove(filepath)
            except OSError: pass

def update_nb_inflation_params(
        model, df_pred, df_truth, max_h, K=6, target=0.90, lo=0.05, hi=0.95):
    
    filepath = f"{NB_LOG_PATH}/{model}.json"
    state = load_nb_state(filepath, max_h)
    cov_by_h, counts_by_h = compute_coverage_by_horizon(
        df_pred, df_truth, K=K, q_lo=lo, q_hi=hi
    )
    state = update_nb_state(
        state, cov_by_h, counts_by_h, max_h, target=target
    )
    save_nb_state(filepath, state)
    #keep log
    as_of_date = df_pred["target_end_date"].max() if (df_pred is not None and not df_pred.empty) else pd.Timestamp("today")
    append_nb_history(model, as_of_date, state, max_h)
    return state


def process_nb_inflation(pred, nb_inflation_params,
                         nb_clip_nonneg=True, nb_round_int=True):

    nb_alpha = nb_inflation_params["alpha_global"] 
    nb_horizon_scale = nb_inflation_params["scales_h"]

    # current samples: shape (H, C, S)
    vals = pred.all_values(copy=False)
    H, C, S = vals.shape

    # center each (time, component) across samples
    mu = np.median(vals, axis=2, keepdims=True)   # (H, C, 1)
    # you can switch to mean if you prefer:
    # mu = vals.mean(axis=2, keepdims=True)

    # horizon-wise alpha multiplier (optional)
    alpha = float(nb_alpha)
    if nb_horizon_scale:
        hmult = np.array([nb_horizon_scale.get(h+1, 1.0) for h in range(H)],
                        dtype=float).reshape(H, 1, 1)
        alpha = alpha * hmult  # broadcasts

    # NB parameterization via Poisson–Gamma mixture:
    # lambda ~ Gamma(shape=k, scale=mu/k), y ~ Poisson(lambda)
    # with k = 1/alpha (allowing non-integer k robustly)
    k = 1.0 / np.maximum(1e-12, alpha)   # (H, C, 1)
    # draw one lambda per existing sample (vectorized)
    # Gamma(shape=k, scale=mu/k) => mean mu, var mu^2/k
    lam = np.random.gamma(shape=k, scale=np.maximum(1e-12, mu/k), size=(H, C, S))

    # Poisson draws around new lambdas
    new_vals = np.random.poisson(lam).astype(float)  # (H, C, S)

    if nb_clip_nonneg:
        new_vals = np.clip(new_vals, 0.0, np.inf)
    if nb_round_int:
        new_vals = np.rint(new_vals)

    # Keep the same time index / components / number of samples
    pred = TimeSeries.from_times_and_values(pred.time_index, new_vals, columns=pred.components)
    return pred