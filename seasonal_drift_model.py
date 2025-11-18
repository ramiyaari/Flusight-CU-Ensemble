import numpy as np
import pandas as pd
from datetime import timedelta

import os

from utils import *

import calendar
from functools import lru_cache

@lru_cache(maxsize=None)
def last_epiweek_in_year(epi_year: int) -> int:
    """
    CDC/MMWR rule (weeks start Sunday, Week 1 has >=4 days in Jan):
    53-week epi-years are when Jan 1 is Wednesday, or Tuesday in a leap year.
    """
    jan1 = date(epi_year, 1, 1)
    wd = jan1.weekday()          # Monday=0 ... Sunday=6
    leap = calendar.isleap(epi_year)
    return 53 if (wd == 2 or (wd == 1 and leap)) else 52


def filter_centered_epiweek_window(df: pd.DataFrame, current_week: int, window: int) -> pd.DataFrame:
    """
    Keep rows whose 'week' is within ±window of current_week, per-row using that row's
    epi-year ring size (52 or 53). Returns a sorted DataFrame.
    Expects columns: ['year','week', ...].
    """
    y = df['year'].astype(int).to_numpy()
    w = df['week'].astype(int).to_numpy()

    # per-row ring size (52 or 53) based on that row's epi-year
    last = np.fromiter((last_epiweek_in_year(yy) for yy in y), dtype=int, count=len(y))

    # shortest circular distance to current_week on each row's ring
    d_forward = (w - current_week) % last
    d = np.minimum(d_forward, last - d_forward)

    mask = d <= window
    return df.loc[mask].sort_values(['year', 'week']).copy()


def prepare_diffs_long(all_states_long: pd.DataFrame) -> pd.DataFrame:
    """
    Add prev_value and diff for each (location, year, week) using a single self-merge.
    Expects columns: ['location','year','week','value'] with ints for year/week.
    Returns the same columns + ['prev_value','diff'].
    """
    df = all_states_long.copy()
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)

    # vectorized predecessor (year, week)
    prev_year = np.where(df["week"].to_numpy() > 1, df["year"].to_numpy(), df["year"].to_numpy() - 1)
    prev_week = df["week"].to_numpy() - 1
    mask_w1 = (df["week"].to_numpy() == 1)
    prev_week_for_w1 = np.array(
        [last_epiweek_in_year(int(y) - 1) for y in df["year"].to_numpy()],
        dtype=int
    )
    prev_week = np.where(mask_w1, prev_week_for_w1, prev_week)

    df["prev_year"] = prev_year
    df["prev_week"] = prev_week

    # self-merge per location to fetch prev_value (single merge for all rows)
    lookback = df[["location","year","week","value"]].rename(
        columns={"year":"prev_year","week":"prev_week","value":"prev_value"}
    )
    out = df.merge(lookback, on=["location","prev_year","prev_week"], how="left")

    # compute diff once
    out["diff"] = out["value"] - out["prev_value"]
    return out

def estimate_seasonal_drift_and_noise(
    self_series: pd.DataFrame,            # ['year','week','value']
    diffs_all: pd.DataFrame,              # ['location','year','week','value','prev_year','prev_week','prev_value','diff']
    current_location: str,                
    current_epiweek: int,
    epiweek_window: int = 0,
    pool_weight: float = 0.0,             # nonnegative ratio; peer weight = p/(1+p)
    residual_cap: int = 2000,
    rng: np.random.Generator | None = None
):
    if rng is None:
        rng = np.random.default_rng(12345)

    w_peer = np.clip(pool_weight, 0.0, 1.0)
    w_self = 1.0 - w_peer

    # --- use your canonical windowing for BOTH baseline and diffs ---
    self_win = filter_centered_epiweek_window(self_series, current_epiweek, epiweek_window)
    baseline_level = float(self_win["value"].median()) if not self_win.empty else 0.0

    diffs_slice = filter_centered_epiweek_window(diffs_all, current_epiweek, epiweek_window)

    # self vs peers diffs
    diffs_self  = diffs_slice.loc[diffs_slice["location"] == current_location, "diff"].dropna().to_numpy()
    diffs_peers = diffs_slice.loc[diffs_slice["location"] != current_location, "diff"].dropna().to_numpy()

    # moments
    mu_self  = float(diffs_self.mean()) if diffs_self.size > 0 else 0.0
    var_self = float(diffs_self.var(ddof=1)) if diffs_self.size > 1 else 0.0
    mu_peer  = float(diffs_peers.mean()) if diffs_peers.size > 0 else 0.0
    var_peer = float(diffs_peers.var(ddof=1)) if diffs_peers.size > 1 else 0.0

    mu = w_self * mu_self + w_peer * mu_peer
    var_mix = (w_self * (var_self + (mu_self - mu)**2) +
               w_peer * (var_peer + (mu_peer - mu)**2))
    sd = float(np.sqrt(max(var_mix, 0.0)))

    # residual draw (vectorized) centered at blended mean
    if (diffs_self.size + diffs_peers.size) == 0:
        centered_residuals = pd.Series([], dtype=float)
    else:
        res_self = diffs_self - mu
        res_peer = diffs_peers - mu
        M = min(int(residual_cap), max(res_self.size + res_peer.size, 512))

        if res_self.size == 0:
            draw = res_peer[rng.choice(res_peer.size, size=M, replace=(res_peer.size < M))]
        elif res_peer.size == 0:
            draw = res_self[rng.choice(res_self.size, size=M, replace=(res_self.size < M))]
        else:
            m_self = int(round(w_self * M))
            m_peer = M - m_self
            sel_self = res_self[rng.choice(res_self.size, size=m_self, replace=(res_self.size < m_self))]
            sel_peer = res_peer[rng.choice(res_peer.size, size=m_peer, replace=(res_peer.size < m_peer))]
            draw = np.concatenate([sel_self, sel_peer], axis=0)

        centered_residuals = pd.Series(draw, dtype=float)

    return mu, sd, centered_residuals, baseline_level


def generate_seasonal_drift_pred(df, ref_date, weeks_to_predict, locations, quantiles, num_samples, epiweek_window, 
                                   dat_changerate_ref, basedir, 
                                   model_desc='seasonal_drift', pool_weight = 0.5,
                                   generate_qual_pred=True, save_results=True,
                                   return_samples=False, random_state=None):

    
    rng = np.random.default_rng(random_state)

    locations_abbr = locations.index 

    # Ensure dates are datetime
    df['date'] = pd.to_datetime(df['date'])
    ref_date = pd.to_datetime(ref_date)

    # Filter data up to (but not including) the reference date
    past_data = df[df['date'] < ref_date].copy()

    for loc_abbr in locations_abbr:
        pop_size = locations.loc[loc_abbr].population
        past_data[loc_abbr] = past_data[loc_abbr]/pop_size*1e5
    
    all_long = []
    for loc_abbr in locations_abbr:
        loc_col = loc_abbr
        tmp = past_data[["year","week",loc_col]].rename(columns={loc_col:"value"}).copy()
        tmp["location"] = loc_abbr
        all_long.append(tmp)
    all_long = pd.concat(all_long, ignore_index=True)
    diffs_all = prepare_diffs_long(all_long)  

    pred_results = []
    samples_results = []        # will hold per-location samples in long format

    # Loop through each location
    for loc_abbr in locations_abbr:

        # print(loc_abbr)

        location = locations.loc[loc_abbr].location
        pop_size = locations.loc[loc_abbr].population
        ref_val = dat_changerate_ref[loc_abbr].values[0]

        # Historical data for the location
        series = past_data[['year', 'week', loc_abbr]].rename(columns={loc_abbr: 'value'}).dropna()

        # Get the last observed value
        last_value = series['value'].iloc[-1]

        # Pre-allocate sample path storage for this location: [horizon, sample]
        samples_matrix = np.empty((weeks_to_predict, num_samples), dtype=float)

        # Initialize current_samples as the last observed value repeated
        current_samples = np.full(num_samples, last_value)

        # Monte Carlo sampling for predictions
        for horizon in range(weeks_to_predict):
            target_date = ref_date + timedelta(weeks=horizon)

            target_epiweek = date_to_epiweek(target_date)[1]

            mean_drift, std_drift, residuals, baseline_level = estimate_seasonal_drift_and_noise(
                    self_series=series,                 
                    diffs_all=diffs_all,                 
                    current_location=loc_abbr,
                    current_epiweek=target_epiweek,
                    epiweek_window=epiweek_window,
                    pool_weight=pool_weight,
                    residual_cap=1000,
                    rng=rng
                )

            # reversion strengths (tune)   
            if target_epiweek >= 26 or target_epiweek < 6: #first part of new season
                lambda_add = 0.0     
            elif target_epiweek >= 22:
                lambda_add = 0.5   
            elif target_epiweek >= 18:
                lambda_add = 0.4    
            elif target_epiweek >= 14:
                lambda_add = 0.3
            elif target_epiweek >= 10:
                lambda_add = 0.2
            elif target_epiweek >= 6:
                lambda_add = 0.1

            # lambda_add /= 2

            base = np.maximum(current_samples, 0.0)
            delta = base - baseline_level                    # how far from seasonal level (linear scale)
            mu_adj = mean_drift - lambda_add * delta         # tilt mean back toward baseline
            st = rng.normal(loc=mu_adj, scale=std_drift, size=base.size) if std_drift > 0 else 0.0
            # (optional) add residual resampling back if you want extra realism:
            # nz = rng.choice(residuals_add, size=base.size, replace=True) if len(residuals_add) else 0.0
            nz = rng.normal(loc=np.mean(residuals), scale=np.std(residuals), size=base.size) if len(residuals) else 0.0
            step = st  + nz
            samples = np.maximum(base + step, 0.0)

            current_samples = samples

            # move back from rates to numbers
            samples = samples*pop_size/1e5

            # Store this horizon’s full draw
            samples_matrix[horizon, :] = samples

            # Compute specified quantiles from the samples
            for quantile in quantiles:
                predicted_value = np.quantile(samples, quantile)
                pred_results.append([format(ref_date,'%Y-%m-%d'),'wk inc flu hosp',horizon,format(target_date,'%Y-%m-%d'),
                                     location,'quantile',np.round(quantile,3),np.round(predicted_value,3)])

            if(generate_qual_pred):
                pred_qual = generate_qualtitative_pred(ref_date, location, horizon, samples, ref_val, pop_size)
                pred_results = pred_results+pred_qual

        # --------- Collect samples for this location (optional) ----------
        if return_samples:
            # Build long/tidy DataFrame for this location
            horizons = np.arange(weeks_to_predict)
            dates = (pd.to_datetime(ref_date) + pd.to_timedelta(horizons, unit="W")).to_numpy()

            # Repeat/tile to long
            H = weeks_to_predict
            S = num_samples
            df_loc = pd.DataFrame({
                "reference_date": np.repeat(ref_date, H*S),
                "location": np.repeat(str(location).zfill(2), H*S),
                "horizon": np.repeat(horizons, S),
                "target_end_date": np.repeat(dates, S),
                "sample_id": np.tile(np.arange(S), H),
                "value": samples_matrix.reshape(-1)
            })
            samples_results.append(df_loc)

    # df_pred_results = pd.DataFrame(pred_results)
    df_pred_results = pd.DataFrame(pred_results,columns=['reference_date','target','horizon','target_end_date',
                                                         'location','output_type','output_type_id','value']) 

    if(generate_qual_pred):
        target_order = ['wk inc flu hosp', 'wk flu hosp rate change']
        df_pred_results['target'] = pd.Categorical(df_pred_results['target'], categories=target_order, ordered=True)
        df_pred_results = df_pred_results.sort_values(by=['target', 'location','horizon'])

    df_pred_results['location'] = df_pred_results['location'].astype(str).str.zfill(2)

    if(save_results):
        output_dir = "{}/{}".format(basedir, model_desc)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        file_path = "{}/{}-CU-{}.csv".format(output_dir,format(ref_date,'%Y-%m-%d'),model_desc)
        # print(file_path)
        df_pred_results.to_csv(file_path, index=False) 
    
    if return_samples:
        df_samples = pd.concat(samples_results, ignore_index=True) 
        return df_pred_results, df_samples
        
    return df_pred_results