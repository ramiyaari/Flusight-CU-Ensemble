import numpy as np
import pandas as pd
from datetime import timedelta

import os

from utils import *



def get_epiweek_window(current_epiweek, epiweek_window):
    """
    Generate a list of epiweeks in the window around the current epiweek.
    """
    max_weeks = 52  # Maximum number of epiweeks in a year
    return [(current_epiweek + offset - 1) % max_weeks + 1 for offset in range(-epiweek_window, epiweek_window + 1)]


def estimate_seasonal_drift_and_noise(data, current_epiweek, epiweek_window=0):
    """
    Estimate seasonal drift, its standard deviation, and noise for a given epiweek.
    Returns:
    - tuple: (mean_drift, sd_drift, centered_residuals)
    """
    # Determine the epiweek window (handles wrap-around)
    epiweeks_in_window = get_epiweek_window(current_epiweek, epiweek_window)

    # Filter data for the relevant epiweeks
    relevant_data = data[data['week'].isin(epiweeks_in_window)]

    # Ensure data is sorted by epiyear and epiweek
    relevant_data = relevant_data.sort_values(by=['year', 'week'])

    # Calculate first differences (drift) by year
    relevant_data['diff'] = relevant_data.groupby('year')['value'].diff()

    # Mean drift
    mean_drift = relevant_data['diff'].mean()

    # Standard deviation of drift
    sd_drift = relevant_data['diff'].std()

    # Calculate residuals (actual - predicted drift)
    relevant_data['residual'] = relevant_data['diff'] - mean_drift
    residuals = relevant_data['residual'].dropna()

    # Center residuals
    centered_residuals = residuals - residuals.mean()

    return mean_drift, sd_drift, centered_residuals


def generate_historical_drift_pred(df, ref_date, weeks_to_predict, locations, quantiles, num_samples, epiweek_window, 
                                   dat_changerate_ref, basedir, model_desc='historical_drift'):

    # Ensure dates are datetime
    df['date'] = pd.to_datetime(df['date'])
    ref_date = pd.to_datetime(ref_date)

    # Filter data up to (but not including) the reference date
    past_data = df[df['date'] < ref_date].copy()

    # Extract epiweek of the reference date
    ref_epiweek = date_to_epiweek(ref_date)[1]

    # Criteria for qualitative forecasts
    stable_criteria = [0.3, 0.5, 0.7, 1.0]
    change_criteria = [1.7, 3.0, 4.0, 5.0]

    pred_results = []

    # Loop through each location
    locations_abbr = locations.index 
    for loc_abbr in locations_abbr:

        # print(loc_abbr)

        location = locations.loc[loc_abbr].location
        pop_size = locations.loc[loc_abbr].population
        ref_vals = dat_changerate_ref[loc_abbr].values

        # Historical data for the location
        series = past_data[['year', 'week', loc_abbr]].rename(columns={loc_abbr: 'value'}).dropna()

        # Estimate seasonal drift and residual-based noise
        mean_drift, std_drift, residuals = estimate_seasonal_drift_and_noise(series, ref_epiweek, epiweek_window=epiweek_window)

        # Get the last observed value
        last_value = series['value'].iloc[-1]

        # Initialize current_samples as the last observed value repeated
        current_samples = np.full(num_samples, last_value)

        # Monte Carlo sampling for predictions
        for horizon in range(weeks_to_predict):
            target_date = ref_date + timedelta(weeks=horizon)
            samples = []
            for sample in current_samples:
                stochastic_drift = np.random.normal(loc=mean_drift, scale=std_drift)
                noise = np.random.choice(residuals)
                # noise = np.random.normal(loc=0, scale=noise_std)
                new_sample = sample + stochastic_drift + noise                    
                samples.append(max(new_sample,0))

            # Compute specified quantiles from the samples
            for quantile in quantiles:
                predicted_value = np.quantile(samples, quantile)
                pred_results.append([format(ref_date,'%Y-%m-%d'),'wk inc flu hosp',horizon,format(target_date,'%Y-%m-%d'),
                                     location,'quantile',np.round(quantile,3),np.round(predicted_value,3)])

            # current_samples = np.full(num_samples, np.mean(samples)) # Use the mean of samples as the next value
            current_samples = samples

            cases_change = samples-ref_vals
            rate_changes = cases_change/pop_size*1e5
            num_samples = len(rate_changes)
            prob_stable = np.sum(np.abs(cases_change) < 10 |
                                 ((rate_changes>-stable_criteria[horizon]) & 
                                 (rate_changes<stable_criteria[horizon])))/num_samples
            prob_increase = np.sum((rate_changes>=stable_criteria[horizon]) & 
                                   (rate_changes<change_criteria[horizon]))/num_samples
            prob_large_increase = np.sum(rate_changes>=change_criteria[horizon])/num_samples
            prob_decrease = np.sum((rate_changes<=-stable_criteria[horizon]) & 
                                   (rate_changes>-change_criteria[horizon]))/num_samples 
            prob_large_decrease = np.sum(rate_changes<=-change_criteria[horizon])/num_samples

            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change', horizon,
                    format(ref_date + timedelta(weeks=horizon),'%Y-%m-%d'),
                    location, 'pmf', 'stable', prob_stable])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change', horizon,
                    format(ref_date + timedelta(weeks=horizon),'%Y-%m-%d'),
                    location, 'pmf', 'increase', prob_increase])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change', horizon,
                    format(ref_date + timedelta(weeks=horizon),'%Y-%m-%d'),
                    location, 'pmf', 'large_increase', prob_large_increase])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change', horizon,
                    format(ref_date + timedelta(weeks=horizon),'%Y-%m-%d'),
                    location, 'pmf', 'decrease', prob_decrease])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change', horizon,
                    format(ref_date + timedelta(weeks=horizon),'%Y-%m-%d'),
                    location, 'pmf', 'large_decrease', prob_large_decrease])

    # df_pred_results = pd.DataFrame(pred_results)
    df_pred_results = pd.DataFrame(pred_results,columns=['reference_date','target','horizon','target_end_date',
                                                         'location','output_type','output_type_id','value']) 

    target_order = ['wk inc flu hosp', 'wk flu hosp rate change']
    df_pred_results['target'] = pd.Categorical(df_pred_results['target'], categories=target_order, ordered=True)
    df_pred_results = df_pred_results.sort_values(by=['target', 'location','horizon'])

    df_pred_results['location'] = df_pred_results['location'].astype(str).str.zfill(2)

    output_dir = "{}/{}".format(basedir, model_desc)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    df_pred_results.to_csv("{}/{}-CU-{}.csv".format(output_dir,format(ref_date,'%Y-%m-%d'),model_desc), index=False) 
    return df_pred_results