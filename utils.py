import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from epiweeks import Week
from datetime import timedelta, datetime, date
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
#from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle, quantile_loss
from darts.utils.missing_values import fill_missing_values

#import scipy.stats as st
from scipy.stats import gaussian_kde


#from delphi_epidata import Epidata
#import inspect

import requests
import warnings
import torch
import os

# print(darts.__version__)
# print(torch.cuda.is_available())


def date_to_epiweek(date):
    """
    This function takes a date and returns a tuple with the year and epidemiological week number.
    It uses the epiweeks package, which is designed to follow the CDC epidemiological week rules.
    """
    epi_week = Week.fromdate(date)
    return epi_week.year, epi_week.week


def epiweek_to_dates(year, epiweek):
    dates = Week(year, epiweek)
    return dates


def format_hosp_data_2024(filename, states):
    
    # Load the data
    df = pd.read_csv(filename)
    df = df.fillna('NA')

    # Define the starting and current date
    first_date = pd.to_datetime("2020-10-04")  # start from a Sunday

    # Filter out rows where state is 'AS' and limit to first_date or later
    df = df[(df['state'] != 'AS') & (pd.to_datetime(df['date']) >= first_date)]

    # select and format required columns
    df = df[['date', 'state', 'previous_day_admission_influenza_confirmed']]
    df.rename(columns={'previous_day_admission_influenza_confirmed': 'cases'}, inplace=True)
    df['cases'] = df['cases'].replace(['NA', 'nan', ''], pd.NA)
    df['cases'] = df['cases'].astype('Int64')
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d') #format='%m/%d/%Y') #

    # Calculate National (US) aggregate
    national = df.groupby('date', as_index=False).agg(
        state=('cases', lambda x: 'US'),
        cases=('cases', 'sum')
    )
    df = pd.concat([df, national], ignore_index=True)

    dat_wide = df.pivot(index='date', columns='state', values='cases')
    dat_wide.index =  dat_wide.index + pd.DateOffset(days=6)  # set to the last day of epiweek (Saturday)

    # Resample weekly (ending on Saturday) and sum
    dat_weekly = dat_wide.resample('W-SAT').sum()

    # Reset index and finalize the weekly data frame
    dat_weekly_wide = dat_weekly.reset_index()

    # Add epidemiological week info
    dat_weekly_wide['year'] = dat_weekly_wide['date'].apply(lambda d: date_to_epiweek(d)[0])
    dat_weekly_wide['week'] = dat_weekly_wide['date'].apply(lambda d: date_to_epiweek(d)[1])

    # Filter the weekly data to include only the selected states
    dat_weekly_wide = dat_weekly_wide[['date', 'year', 'week'] + states.tolist()]
    return dat_weekly_wide


def format_hosp_data(filename, states):
    
    # Load the data
    df = pd.read_csv(filename)
    df = df.fillna('NA')

    # select and format required columns
    df = df[['Week Ending Date', 'Geographic aggregation', 'Total Influenza Admissions']]
    df.rename(columns={'Week Ending Date' : 'date',
                       'Geographic aggregation': 'state',
                       'Total Influenza Admissions': 'cases',
                       }, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d') #'%m/%d/%Y' ) #'%Y/%m/%d')
    df['cases'] = df['cases'].replace(['NA', 'nan', ''], pd.NA)
    df['cases'] = df['cases'].astype('Int64')
    df['state'] = df['state'].replace('USA', 'US')
    df = df[(df['state'] != 'AS')]

    dat_wide = df.pivot(index='date', columns='state', values='cases')

    # Resample weekly (ending on Saturday) and sum
    dat_weekly = dat_wide.resample('W-SAT').sum()

    # Reset index and finalize the weekly data frame
    dat_weekly_wide = dat_weekly.reset_index()

    # Add epidemiological week info
    dat_weekly_wide['year'] = dat_weekly_wide['date'].apply(lambda d: date_to_epiweek(d)[0])
    dat_weekly_wide['week'] = dat_weekly_wide['date'].apply(lambda d: date_to_epiweek(d)[1])

    # Filter the weekly data to include only the selected states
    dat_weekly_wide = dat_weekly_wide[['date', 'year', 'week'] + states.tolist()]
    return dat_weekly_wide



def fetch_ili_data(regions, epiweek_range):
    """
    Fetches CDC ILI data for a specified region and epiweek range, extracting only weighted_ili values per date.
    
    Args:
        region (str): The region code, e.g., 'nat' for national or a state abbreviation like 'ca'.
        epiweek_range (str): The range of epiweeks in the format 'YYYYWW-YYYYWW', e.g., '201001-202353'.    
    Returns:
        pd.DataFrame: A DataFrame with 'date' and 'w' columns.
    """
    base_url = "https://api.delphi.cmu.edu/epidata/fluview/"

    api_key = "<the_api_key>"

    regions = regions.copy()
    regions[regions=='US'] = 'nat'

    # Parameters and headers for the request
    params = {'regions': regions, 'epiweeks': epiweek_range}
    headers = {'Authorization': f'Bearer {api_key}'}
    
    # Make the API request
    response = requests.get(base_url, params=params, headers=headers)
    
    # Process the response
    if response.status_code == 200:
        data = response.json()
        if data['result'] == 1:
            # Extract only date and weighted_ili columns
            ili_data = pd.DataFrame(data['epidata'])
            df_ili = ili_data[['epiweek','region','wili']].copy(deep=True)
            df_ili['year'] = df_ili['epiweek'] // 100
            df_ili['week'] = df_ili['epiweek'] % 100 
            df_ili['region'] = df_ili['region'].str.upper().replace('NAT', 'US')
            df_ili['date'] = df_ili.apply(lambda row: epiweek_to_dates(row['year'], row['week']).enddate(), axis=1)
            df_ili_wide = df_ili.pivot(index=['date', 'year', 'week'], columns='region', values='wili').reset_index()
            df_ili_wide.columns.name = ''
            if 'US' in df_ili_wide.columns:
                cols = [col for col in df_ili_wide.columns if col != 'US']
                df_ili_wide = df_ili_wide[cols + ['US']]
            return df_ili_wide
        else:
            print(f"Error: {data['message']}")
            return pd.DataFrame()  # Return empty DataFrame on error
    else:
        print(f"Failed to fetch data: {response.status_code} - {response.text}")
        return pd.DataFrame()  # Return empty DataFrame on request failure
    

def read_hosp_incidence_data(data_dir, epiyear, epiweek, states, new_format=True, download=True, plot=True):
    
    hosp_data_fname = data_dir + "hosp_cases_{}_{}.csv".format(epiyear,epiweek)
    if(not os.path.exists(hosp_data_fname)):
        if(new_format): #2024-2025 format
            filename = 'Weekly_Hospital_Respiratory_Data__HRD__Metrics_by_Jurisdiction__National_Healthcare_Safety_Network__NHSN.csv'
            url = ""
        else:
            filename = 'COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv'
            url = 'https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD'
        hosp_rates_file = f"{data_dir}{filename}"
        if(download):
            df = pd.read_csv(url)
            df = df.fillna('NA')
            df.to_csv(hosp_rates_file, index=False)
        if(new_format):
            df_hosp = format_hosp_data(hosp_rates_file, states)
        else:
            df_hosp = format_hosp_data_2024(hosp_rates_file, states)
        max_date = df_hosp['date'].max()
        cut_date = pd.Timestamp(epiweek_to_dates(epiyear, epiweek).enddate())
        if(max_date > cut_date):
            print("Max date {} is larger than the cut date {} - data will be truncated".format(max_date,cut_date))
            df_hosp = df_hosp[df_hosp['date']<=cut_date]
        df_hosp.to_csv(hosp_data_fname,index=False)
    else:
        df_hosp = pd.read_csv(hosp_data_fname)

    df_hosp['date'] = pd.to_datetime(df_hosp['date'], format='%Y-%m-%d') #format='%m/%d/%Y') #        
    if(plot):
        df_hosp_long = pd.melt(df_hosp,id_vars=['date','year','week'],value_vars=df_hosp.columns[3:],var_name='state',value_name='hosp cases')
        g = sns.FacetGrid(df_hosp_long, col="state", col_wrap=5, hue="state", sharey=False, sharex=True, height=3, aspect=1.33)
        g.map(sns.lineplot, "date", "hosp cases")
        [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

    return (df_hosp)


def read_lab_selected_data(data_dir, epiyear, epiweek, states, loc_name2abbr, sel_col, plot=True):

    lab_data_fname = data_dir + "lab_cases_{}_{}.csv".format(epiyear,epiweek)
    df_lab = pd.read_csv(lab_data_fname, na_values="X")
    df_lab['date'] = df_lab.apply(lambda row: epiweek_to_dates(row['year'], row['week']).enddate(), axis=1)
    df_lab['date'] = pd.to_datetime(df_lab['date'], format='%Y-%m-%d') #format='%m/%d/%Y') #
    df_lab = df_lab[['date'] + [col for col in df_lab.columns if col != 'date']]
    df_lab['location'] = df_lab['location'].replace(loc_name2abbr)
    df_lab = df_lab[df_lab.location.isin(states)]
    max_date = df_lab['date'].max()
    cut_date = pd.Timestamp(epiweek_to_dates(epiyear, epiweek).enddate())
    if(max_date > cut_date):
        print("Max date {} is larger than the cut date {} - data will be truncated".format(max_date,cut_date))
        df_lab = df_lab[df_lab['date']<=cut_date]
    
    us_totals = df_lab.groupby('date').agg(
        total_specimens=('total_specimens', 'sum'),
        total_A=('total_A', 'sum'),
        total_B=('total_B', 'sum')
    ).reset_index()

    us_totals['percent_positive'] = np.round(100*(us_totals['total_A'] + us_totals['total_B']) / us_totals['total_specimens'],2)
    us_totals['percent_A'] = np.round(100*us_totals['total_A'] / us_totals['total_specimens'],2)
    us_totals['percent_B'] = np.round(100*us_totals['total_B'] / us_totals['total_specimens'],2)
    us_totals['location'] = 'US'
    us_totals['year'] = pd.to_datetime(us_totals['date']).dt.year
    us_totals['week'] = pd.to_datetime(us_totals['date']).dt.isocalendar().week
    df_lab = pd.concat([df_lab, us_totals], ignore_index=True)
    # df_lab = df_lab.sort_values(by=['date', 'location']).reset_index(drop=True)

    if(plot):
        g = sns.FacetGrid(df_lab, col="location", col_wrap=5, hue="location", sharey=False, sharex=True, height=3, aspect=1.33)
        g.map(sns.lineplot, "date", sel_col)
        [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

    df_lab_sel = df_lab.pivot(index='date', columns='location', values=sel_col).reset_index()
    return df_lab_sel


def read_ili_incidence_data(data_dir, epiyear, epiweek, states, df_hosp, smooth=True, scale=True, plot=False):

    ili_data_fname = data_dir + "ili_cases_{}_{}.csv".format(epiyear,epiweek)
    if not os.path.exists(ili_data_fname):
        epiweek_range = f"201001-{epiyear}{epiweek:02d}"
        df_ili = fetch_ili_data(states, epiweek_range)
        df_ili.to_csv(ili_data_fname,index=False)
    else:
        df_ili = pd.read_csv(ili_data_fname)

    df_ili['date'] = pd.to_datetime(df_ili['date'], format='%Y-%m-%d')       
    if(smooth):
        df_ili.iloc[:, 3:] = df_ili.iloc[:, 3:].astype(float)
        df_ili.iloc[:, 3:] = df_ili.iloc[:, 3:].rolling(window=3, min_periods=1).mean()
    
    if(scale):
        df_hosp = df_hosp[df_hosp['year']>=2022]
        years = df_ili['year'].unique()
        for state in states:
            df_ili[state] = df_ili.groupby('year')[state].transform(lambda x: x - x.min())
            state_ratios = []
            for year in years:
                ratio = None
                max_hosp = df_hosp[(df_hosp['year'] == year)][state].max()
                max_ili = df_ili[(df_ili['year'] == year)][state].max()
                if (not np.isnan(max_hosp)) and (not np.isnan(max_ili)) and (max_ili>0):
                    ratio = max_hosp / max_ili
                state_ratios.append(ratio)
            valid_ratios = [r for r in state_ratios if r is not None and np.isfinite(r)]
            mean_ratio = np.nanmean(valid_ratios) if valid_ratios else 1
            for yi, year in enumerate(years):
                ratio = state_ratios[yi]
                if ratio is not None:
                    df_ili.loc[df_ili['year']==year, state] *= ratio
                else:
                    df_ili.loc[df_ili['year']==year, state] *= mean_ratio

    if(plot):
        df_ili_long = pd.melt(df_ili,id_vars=['date','year','week'],value_vars=df_ili.columns[3:],var_name='state',value_name='weighted ili')
        g = sns.FacetGrid(df_ili_long, col="state", col_wrap=5, hue="state", sharey=False, sharex=True, height=3, aspect=1.33)
        g.map(sns.lineplot, "date", "weighted ili")
        [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

    return (df_ili)

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


def calculate_wis(truth, df_pred, alpha_vals):
    wis_vals = 0
    if hasattr(truth, "__len__"):
       wis_vals = np.zeros(len(truth)) 
    for alpha in alpha_vals:
        low_vals = df_pred.loc[df_pred['quantile']==np.round(alpha/2,3),'value'].values
        high_vals = df_pred.loc[df_pred['quantile']==np.round(1-alpha/2,3),'value'].values
        int_width = high_vals - low_vals
        too_low = (truth < low_vals).astype(int)
        too_high = (truth > high_vals).astype(int)
        too_low_penalty = (2/alpha)*too_low*(low_vals - truth)
        too_high_penalty = (2/alpha)*too_high*(truth - high_vals)
        wis_vals += (alpha/2)*(int_width + too_low_penalty + too_high_penalty)
    K = len(alpha_vals)-1
    wis_vals = wis_vals/(K+0.5)
    #wis_vals = wis_vals/np.sum(np.array(alpha_vals)/2)
    return wis_vals


def save_pred_results_to_file(pred, ref_date, model_desc, locations, quantiles, dat_hosp_changerate_ref, basedir):
   
    pred_results = []
    horizons = ((pred.time_index-ref_date).days.values/7).astype(int)
    locations_abbr = locations.index #pred.components
    for loc_abbr in locations_abbr:
        location = locations.loc[loc_abbr].location
        pred_state = pred[loc_abbr]

        pred_quantiles = get_quantiles_df(pred_state,quantiles)
        for hind, horizon in enumerate(horizons):
            for qind, quantile in enumerate(quantiles):
                pred_results.append([format(ref_date,'%Y-%m-%d'),'wk inc flu hosp', horizon,
                          format(ref_date + timedelta(weeks=horizon.item()),'%Y-%m-%d'),
                          location, 'quantile', np.round(quantile,3), pred_quantiles.iloc[hind,qind]])


    stable_criteria = [0.3, 0.5, 0.7, 1.0]
    change_criteria = [1.7, 3.0, 4.0, 5.0]

    locations_abbr = locations.index #pred.components
    for loc_abbr in locations_abbr:
        location = locations.loc[loc_abbr].location
        pop_size = locations.loc[loc_abbr].population
        hosp_vals = dat_hosp_changerate_ref[loc_abbr].values
        for horizon in horizons:
            pred_vals = pred[ref_date+timedelta(weeks=horizon.item())][loc_abbr].all_values()[0][0]
            cases_change = pred_vals-hosp_vals
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
                    format(ref_date + timedelta(weeks=horizon.item()),'%Y-%m-%d'),
                    location, 'pmf', 'stable', prob_stable])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change', horizon,
                    format(ref_date + timedelta(weeks=horizon.item()),'%Y-%m-%d'),
                    location, 'pmf', 'increase', prob_increase])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change', horizon,
                    format(ref_date + timedelta(weeks=horizon.item()),'%Y-%m-%d'),
                    location, 'pmf', 'large_increase', prob_large_increase])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change', horizon,
                    format(ref_date + timedelta(weeks=horizon.item()),'%Y-%m-%d'),
                    location, 'pmf', 'decrease', prob_decrease])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change', horizon,
                    format(ref_date + timedelta(weeks=horizon.item()),'%Y-%m-%d'),
                    location, 'pmf', 'large_decrease', prob_large_decrease])
            
    df_pred_results = pd.DataFrame(pred_results,columns=['reference_date','target','horizon','target_end_date',
                                                         'location','output_type','output_type_id','value']) 
    
    output_dir = "{}/{}".format(basedir, model_desc)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    df_pred_results.to_csv("{}/{}-CU-{}.csv".format(output_dir,format(ref_date,'%Y-%m-%d'),model_desc), index=False) 
    return df_pred_results


def load_pred_result_files(basedir, model_desc, locations):

    results_dir = "{}/{}".format(basedir,model_desc)
    file_list = os.listdir(results_dir)
    df_results = None
    for i, file in enumerate(file_list):
        df_results_cur = pd.read_csv("{}/{}".format(results_dir,file))
        if(i==0):
            df_results = df_results_cur
        else:
            df_results = pd.concat([df_results,df_results_cur])
    
    df_results['reference_date'] = pd.to_datetime(df_results['reference_date'], format='%Y-%m-%d')
    df_results['target_end_date'] = pd.to_datetime(df_results['target_end_date'], format='%Y-%m-%d')
    #df_results['location'] = [loc.lstrip('0') for loc in df_results['location']]
    df_results = df_results.astype({"output_type_id": str})
    df_results['output_type_id'] = [val.rstrip('0') for val in df_results['output_type_id']]
    df_results = df_results[df_results['horizon']>=0]
    df_results['horizon'] = df_results['horizon'].astype('Int32')
    df_results = df_results[df_results.location.isin(locations.location)]
    return (df_results)


def fix_change_rate_values(df):
    
    # set minimum for rate change values and checks that add up to 1 after rounding
    locations = df.location.unique()
    horizons =  df.horizon.unique()
    for location in locations:
        for horizon in horizons:
            df_ens_hl = df[(df.target=='wk flu hosp rate change') & (df.location==location) & (df.horizon==horizon)]
            df_ens_hl.loc[:,'value'] = np.clip(df_ens_hl['value'],a_min=0.001,a_max=None)
            df_ens_hl.loc[:,'value'] = df_ens_hl['value']/np.sum(df_ens_hl['value'])
            df_ens_hl.loc[:,'value'] = np.round(df_ens_hl['value'],3)
            df_ens_hl.loc[df_ens_hl.output_type_id=='stable','value'] += np.round(1-np.sum(df_ens_hl['value']),3)
            df.loc[(df.target=='wk flu hosp rate change') & (df.location==location) & (df.horizon==horizon),'value'] = df_ens_hl['value']
    return (df)


def merge_pred_results(results_dict):

    if(len(results_dict)<2) :
        print('Error: size of dictionary is smaller than 2')
        return None
    
    models = list(results_dict.keys())
    model = models[0]
    df_merged = results_dict[model].rename(columns={'value': model})
    for model in models[1:]:
        df_merged2 = results_dict[model].rename(columns={'value': model })
        df_merged = df_merged.merge(df_merged2,how='inner',
                                        on=['reference_date','target','horizon','target_end_date',
                                            'location','output_type','output_type_id']) 
    return (df_merged)
        
    
def generate_mean_ensemble_pred_results(results_dict, basedir, ensemble_name):

    models = list(results_dict.keys())
    df_ensemble = merge_pred_results(results_dict)
    df_ensemble['value'] = np.round(df_ensemble[models].mean(axis=1),3)
    df_ensemble = df_ensemble.drop(labels=models,axis=1)

    df_ensemble.loc[df_ensemble.target=='wk inc flu hosp','value'] = np.round(df_ensemble.loc[df_ensemble.target=='wk inc flu hosp','value'],0)
    
    output_dir = "{}/{}".format(basedir, ensemble_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    ref_dates = df_ensemble['reference_date'].unique()
    for ref_date in ref_dates:
        df_ensemble1 = df_ensemble[df_ensemble['reference_date']==ref_date]
        df_ensemble1 = fix_change_rate_values(df_ensemble1)
        df_ensemble[df_ensemble['reference_date']==ref_date] = df_ensemble1
        df_ensemble1.to_csv("{}/{}-CU-{}.csv".format(output_dir,format(ref_date,'%Y-%m-%d'),ensemble_name), index=False) 

    return df_ensemble


def generate_pred_weights(results_dict, df_metrics, metric, weights_win, locations):

    models = list(results_dict.keys())
    ref_dates = results_dict[models[0]]['reference_date'].unique()
    horizons = results_dict[models[0]]['horizon'].unique()

    df_metrics = df_metrics[(df_metrics.metric==metric) & (df_metrics.model.isin(models))].drop(columns=['metric'])
    df_metrics['target_date'] = pd.to_datetime(df_metrics['target_date'], format='%Y-%m-%d') 

    df_weights = []
    for loc_abbr in locations.index:
        # print("-----------Location: {}-----------".format(loc_abbr))
        for horizon in horizons:
            # print("-----------Horizon: {}-----------".format(horizon))
            df_metrics_hl = df_metrics[(df_metrics.location==loc_abbr) & (df_metrics.horizon==horizon)].drop(columns=['location','horizon'])
            df_metrics_hl = df_metrics_hl.pivot(index='target_date', columns='model', values='value')
            df_metrics_hl = df_metrics_hl[models]
            df_metrics_hl = df_metrics_hl.fillna(1e4)
            df_metrics_hl = np.clip(df_metrics_hl,1,np.inf)
            df_metrics_hl = df_metrics_hl.apply(lambda row : 1/row, axis = 1)
            for t in ref_dates[0:(weights_win+horizon)]:   
                df_weights.append(np.concatenate((np.array([horizon,loc_abbr,t]),np.array([1/len(models)]*len(models)))))
            for t in ref_dates[(weights_win+horizon):len(ref_dates)]:
                df_metrics_hlt = df_metrics_hl[(df_metrics_hl.index<t) & (df_metrics_hl.index>=t-pd.Timedelta(weeks=weights_win))]
                weights = degenerate_em_weights(df_metrics_hlt, models_name=models, tol_stop=1e-4).T
                df_weights.append(np.concatenate((np.array([horizon,loc_abbr,t]),weights.values[0,:])))
                  
    cols = np.concatenate((np.array(['horizon','location','reference_date']),np.array(models)))
    df_weights = pd.DataFrame(df_weights,columns=cols)
    return (df_weights)


def generate_weighted_pred_results(results_dict, df_weights, locations, 
                                   basedir, ensemble_name, average_over_horizons=False):

    if(average_over_horizons):
        df_weights = df_weights.groupby(['location','reference_date']).mean('weight').reset_index().drop(columns=['horizon'])     
    
    models = list(results_dict.keys())
    df_ensemble = merge_pred_results(results_dict)
    df_ensemble['value'] = np.round(df_ensemble[models].mean(axis=1),3)
    ref_dates = df_ensemble['reference_date'].unique()
    horizons = df_ensemble.horizon.unique()
    for loc_abbr in locations.index:
        #print("-----------Location: {}-----------".format(loc_abbr))
        location = locations.loc[loc_abbr].location
        for horizon in horizons:
            # print("-----------Horizon: {}-----------".format(horizon))
            df_ens_hl = df_ensemble[(df_ensemble.location==location) & (df_ensemble.horizon==horizon) & (df_ensemble.output_type=='quantile')]
            for t in ref_dates:
                if(average_over_horizons):
                    weights = df_weights[(df_weights.location==loc_abbr) & (df_weights.reference_date==t)]
                else:
                    weights = df_weights[(df_weights.location==loc_abbr) & (df_weights.horizon==horizon) & (df_weights.reference_date==t)]
                df_ens_hlt = df_ens_hl.loc[df_ens_hl.reference_date==t,models]
                quantile_vals = [np.sum(df_ens_hlt.iloc[r,:].values*weights[models].values) for r in range(df_ens_hlt.shape[0])]
                if(np.isnan(quantile_vals).any()==False):
                    df_ens_hl.loc[df_ens_hl.reference_date==t,'value'] = np.round(quantile_vals,3)
            df_ensemble.loc[(df_ensemble.location==location) & (df_ensemble.horizon==horizon) & (df_ensemble.output_type=='quantile'),'value'] = df_ens_hl['value']

    df_ensemble = df_ensemble.drop(labels=models,axis=1)
    df_ensemble.loc[df_ensemble.target=='wk inc flu hosp','value'] = np.round(df_ensemble.loc[df_ensemble.target=='wk inc flu hosp','value'],0)

    output_dir = "{}/{}".format(basedir, ensemble_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for ref_date in ref_dates:
        df_ensemble1 = df_ensemble[df_ensemble['reference_date']==ref_date]
        df_ensemble1 = fix_change_rate_values(df_ensemble1)
        df_ensemble[df_ensemble['reference_date']==ref_date] = df_ensemble1
        df_ensemble1.to_csv("{}/{}-CU-{}.csv".format(output_dir,format(ref_date,'%Y-%m-%d'),ensemble_name), index=False) 

    return df_ensemble


def calc_pred_fit_ex(df_pred, df_dat, alpha_vals, pop_size):    
    df_pred = df_pred.merge(df_dat, how='inner',on='date',suffixes=('', '_truth'))
    fit_dates = df_pred.date.unique()
    truth = df_pred[df_pred['quantile']==0.5]['value_truth'].values
    pred =  df_pred[df_pred['quantile']==0.5]['value'].values
    dev_vals = (pred-truth)**2
    wis_vals = calculate_wis(truth, df_pred, alpha_vals)
    pred_std = np.round(np.std(pred),2)
    pred_bias = np.round(np.mean(np.abs(pred-truth)),2)
    truth_rate = truth/pop_size*1e5
    pred_rate = pred/pop_size*1e5
    df_pred_rate = df_pred
    df_pred_rate.loc[:,'value'] = df_pred_rate['value'].values/pop_size*1e5
    dev_rate_vals = (pred_rate-truth_rate)**2
    wis_rate_vals = calculate_wis(truth_rate, df_pred_rate, alpha_vals)
    return (fit_dates, dev_vals, wis_vals, pred_std, pred_bias, dev_rate_vals, wis_rate_vals)


def calc_pred_fit(df_pred, df_dat, alpha_vals):
    df_pred = df_pred.merge(df_dat, how='inner',on='date',suffixes=('', '_truth'))
    fit_dates = df_pred.date.unique()
    truth = df_pred[df_pred['quantile']==0.5]['value_truth'].values
    pred =  df_pred[df_pred['quantile']==0.5]['value'].values
    dev_vals = (pred-truth)**2
    wis_vals = calculate_wis(truth, df_pred, alpha_vals)
    return (fit_dates, dev_vals, wis_vals)


def calc_and_plot_pred_results_fit(df_results, df_dat, locations, loc_abbr, alpha_vals, basedir, model_desc, plot=True):

    df_metrics = pd.DataFrame(columns=['location','model','horizon','target_date','metric','value'])

    location = locations.loc[loc_abbr].location

    df_results = df_results[(df_results.target=='wk inc flu hosp') & (df_results.location==location)]
    df_results = df_results.rename(columns={'target_end_date':'date', 'output_type_id':'quantile'})
    df_results = df_results.astype({"quantile": float})
    horizons = df_results.horizon.unique()
    pred_dates = df_results.date.unique()

    df_dat = df_dat.loc[:,['date',loc_abbr]].rename(columns={loc_abbr:'value'})
    df_dat = df_dat[df_dat.date.isin(pred_dates)]

    if(df_dat.empty):
        return df_metrics
    
    if(plot):
        _, axes = plt.subplots(len(horizons), 1, figsize=(5, 10)) 
        min_date = min(df_results.date)
        max_date = max(df_results.date)

    for i, horizon in enumerate(horizons):
        df_pred = df_results[df_results.horizon==horizon]
        #(fit_dates, dev_vals, wis_vals, pred_std, pred_bias, dev_rate_vals, wis_rate_vals) = calc_pred_fit_ex(df_pred, df_dat, alpha_vals, pop_size)
        (fit_dates, dev_vals, wis_vals) = calc_pred_fit(df_pred, df_dat, alpha_vals)
    
        for j, fit_date in enumerate(fit_dates):
            date = format(fit_date,'%Y-%m-%d')
            df_metrics.loc[len(df_metrics.index)] = [loc_abbr, model_desc, horizon, date, 'dev', dev_vals[j]]
            df_metrics.loc[len(df_metrics.index)] = [loc_abbr, model_desc, horizon, date, 'wis', wis_vals[j]]

        rmse = np.round(np.sqrt(np.mean(dev_vals)),2)
        wis = np.round(np.mean(wis_vals),2)

        if(plot):
            axes[i].plot(df_dat.date, df_dat.value ,label='data')
            axes[i].plot(df_pred.date[df_pred['quantile']==0.5], 
                        df_pred.value[df_pred['quantile']==0.5], 
                        label="pred", color='blue', alpha=1)
            axes[i].fill_between(df_pred.date[df_pred['quantile']==0.25], 
                            df_pred.value[df_pred['quantile']==0.25], 
                            df_pred.value[df_pred['quantile']==0.75],
                            color='blue', alpha=0.3)
            axes[i].fill_between(df_pred.date[df_pred['quantile']==0.025],
                            df_pred.value[df_pred['quantile']==0.025], 
                            df_pred.value[df_pred['quantile']==0.975],
                            color='blue', alpha=0.1)
            #axes[i].xticks(rotation=90)
            axes[i].set_xlim([min_date, max_date])

            if(i < len(horizons)-1):
                axes[i].set_xticklabels([])
            axes[i].set_ylabel('hosp cases')
            axes[i].set_title('Horizon={} weeks: RMSE={}, WIS={}'.format(horizon,rmse,wis))
            axes[i].legend()

    if(plot):
        [plt.setp(ax.get_xticklabels(), rotation=90) for ax in axes.flat]
        plt.suptitle('Model {} prediction for {}'.format(model_desc,locations.loc[loc_abbr].location_name))
        output_dir = "{}/{}".format(basedir, model_desc)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir +"/" + loc_abbr + ".png", format="png", bbox_inches="tight", dpi=300)

    return df_metrics


def map_week_to_week_season(week):
    if week >= 40:  # Weeks 40–52
        return week - 40
    else:  # Weeks 1–39
        return week + 12
        
def calculate_historical_stats(df, states, quantiles):

    # Filter data for the given states
    df = df[df['state'].isin(states)]

    # Get rid of week 53
    df = df[~(df.week==53)]

    # Define a season starting from week 40
    df['season'] = df.apply(
        lambda row: f"{row['year']}-{row['year'] + 1}" if row['week'] >= 40 else f"{row['year'] - 1}-{row['year']}",
        axis=1
    )
    #remove covid seasons
    exclude_seasons = ['2020-2021','2021-2022']
    df = df[~df['season'].isin(exclude_seasons)]

    # Calculate mean and quantiles across all years for each state and each week
    weekly_stats = df.groupby(['state', 'week'])['cases'].agg(
        mean='mean',
        **{f"quantile_{int(q * 100)}": lambda x, q=q: x.quantile(q) for q in quantiles}
    ).reset_index()

    # Group by state and season, find the week of the max value for each season
    def get_seasonal_max(group):
        group = group.dropna(subset=['cases'])  # Drop rows where 'cases' is NaN
        if not group.empty:
            return group.loc[group['cases'].idxmax()]
        else:
            return None  # Return None if the group is empty

    seasonal_max = df.groupby(['state', 'season']).apply(get_seasonal_max).dropna().reset_index(drop=True)

    # Calculate probabilities of the max week for each state
    peak_week_probabilities = seasonal_max.groupby(['state', 'week']).size().div(
        seasonal_max.groupby('state').size(), level='state'
    ).reset_index(name='probability')

    # peek week probabilities combined over all states
    peak_week_probabilities_combined = seasonal_max.groupby('week').size().div(len(seasonal_max)).reset_index(name='probability')
    peak_week_probabilities_combined['probability'] /= peak_week_probabilities_combined['probability'].sum()
        
    peak_week_probabilities['week_season'] = peak_week_probabilities['week'].apply(map_week_to_week_season)
    peak_week_probabilities_combined['week_season'] = peak_week_probabilities_combined['week'].apply(map_week_to_week_season)
    peak_week_probabilities = peak_week_probabilities.sort_values(by='week_season')
    peak_week_probabilities_combined = peak_week_probabilities_combined.sort_values(by='week_season')

    # Calculate quantiles for the max cases in each season for each state
    peak_value_quantiles = seasonal_max.groupby('state')['cases'].quantile(quantiles).reset_index()
    peak_value_quantiles.columns = ['state', 'quantile', 'cases']

    # Return results
    return {
        'weekly_stats': weekly_stats,
        'peak_value_quantiles': peak_value_quantiles,
        'peak_week_probabilities': peak_week_probabilities, 
        'peak_week_probabilities_combined': peak_week_probabilities_combined, 
    }


def fit_smooth_distribution_using_kde(probabilities_df, bandwidth=1.0):
    """
    Fits a KDE with adjustable bandwidth to the given probabilities.

    Parameters:
        probabilities_df (pd.DataFrame): DataFrame with columns ['week_season', 'probability'].
        bandwidth (float): Bandwidth adjustment factor (e.g., <1 for narrower, >1 for wider).

    Returns:
        pd.DataFrame: DataFrame with columns ['week_season', 'smooth_probability'], where probabilities sum to 1.
    """

    x = probabilities_df['week_season'].values
    y = probabilities_df['probability'].values

    # Ensure weights sum to 1 for proper scaling
    weights = y / np.sum(y)

    # Fit KDE with bandwidth adjustment
    kde = gaussian_kde(x, weights=weights)
    kde.set_bandwidth(bw_method=kde.factor * bandwidth)

    # Evaluate KDE at discrete points from 1 to 53
    x_full = np.arange(1, 54)
    smooth_probabilities = kde(x_full)

    # Normalize smooth probabilities to sum to 1
    smooth_probabilities /= np.sum(smooth_probabilities)

    # Return results as a DataFrame
    return pd.DataFrame({
        "week_season": x_full,
        "smooth_probability": smooth_probabilities
    })

def generate_peak_week_pred(df_his_stats, locations, ref_date, last_date, 
                            kde_bandwith=0.5, plot_peak_week_prob=False):

    df_peak_week_inc = df_his_stats['peak_value_quantiles'].copy()
    df_peak_week_prob = df_his_stats['peak_week_probabilities'].copy()
    df_peak_week_prob_combnined = df_his_stats['peak_week_probabilities_combined'].copy()

    epiweek = date_to_epiweek(ref_date)[1]
    epiweek_last = date_to_epiweek(last_date)[1]
    epiweek_season = map_week_to_week_season(epiweek)
    epiweek_season_last = map_week_to_week_season(epiweek_last)

    locations_abbr = locations.index #pred.components
    pred_results = []

    # peek week incidence pred
    for loc_abbr in locations_abbr:
        location = locations.loc[loc_abbr].location
        df_peak_week_inc_state = df_peak_week_inc[df_peak_week_inc.state==loc_abbr]
        quantiles = df_peak_week_inc_state['quantile'].values
        for quantile in quantiles:
            pred_results.append([ref_date.strftime("%Y-%m-%d"),'peak inc flu hosp', 'NA','NA',
                            location, 'quantile', np.round(quantile,3), 
                            np.round(df_peak_week_inc_state.loc[df_peak_week_inc_state['quantile']==quantile,'cases'].values[0],0)])
    
    # peek week prob pred
    for loc_abbr in locations_abbr:
        location = locations.loc[loc_abbr].location
        df_peak_week_prob_state = df_peak_week_prob[df_peak_week_prob.state==loc_abbr]
        if(df_peak_week_prob_state.shape[0]<5):
            print(f"Not enough data points for state {loc_abbr} - using combined data")
            df_peak_week_prob_state = df_peak_week_prob_combnined

        # FIX - what if we are passed the peak?
        df_peak_week_prob_state.loc[df_peak_week_prob_state.week_season<epiweek_season,'probability'] = 0
        df_peak_week_prob_state.loc[df_peak_week_prob_state.week_season>epiweek_season_last,'probability'] = 0
        df_peak_week_prob_state.loc[:,'probability'] = df_peak_week_prob_state.loc[:,'probability']/df_peak_week_prob_state['probability'].sum()
        probs = fit_smooth_distribution_using_kde(df_peak_week_prob_state, bandwidth=kde_bandwith)
        probs.loc[probs.week_season<epiweek_season,'smooth_probability'] = 0
        probs.loc[probs.week_season>epiweek_season_last,'smooth_probability'] = 0
        probs['smooth_probability'] = probs['smooth_probability']/probs['smooth_probability'].sum()
        for week in range(epiweek_season,epiweek_season_last+1):
            week_date = (ref_date+timedelta(weeks=(week-epiweek_season+1))).strftime("%Y-%m-%d")
            week_prob = probs.loc[probs.week_season==week,'smooth_probability'].values[0]
            pred_results.append([ref_date.strftime("%Y-%m-%d"),'peak week inc flu hosp', 'NA','NA',
                            location, 'pmf', week_date, np.round(week_prob,5)])
            
        if(plot_peak_week_prob):
            df_peak_week_prob_state = df_peak_week_prob_state[df_peak_week_prob_state.week_season.isin(range(epiweek_season,epiweek_season_last+1))]
            probs = probs[probs.week_season.isin(range(epiweek_season,epiweek_season_last+1))]
            week_dates = [(ref_date + timedelta(weeks=w)).strftime("%Y-%m-%d") for w in range(0,epiweek_season_last-epiweek_season+1)]
            plt.figure(figsize=(7,3))
            plt.bar(probs["week_season"], 
                    probs["smooth_probability"], 
                    label="Smooth Probabilities", alpha=0.7)
            plt.scatter(df_peak_week_prob_state['week_season'],
                        df_peak_week_prob_state['probability'],
                        color="red", label="Calculated Probabilities")
            plt.xticks(ticks=range(epiweek_season,epiweek_season_last+1), labels=week_dates,rotation=90)
            plt.xlabel("")
            plt.ylabel("Probability")
            # plt.legend()
            plt.title(f"Peak week distribution for {loc_abbr}")
            plt.show()

    df_results = pd.DataFrame(pred_results,columns=['reference_date','target','horizon','target_end_date',
                                                    'location','output_type','output_type_id','value'])
    df_results['reference_date'] = pd.to_datetime(df_results['reference_date'], format='%Y-%m-%d')
    return df_results


def save_pred_peak_with_model_pred(df_pred_peak, basedir, model_desc, locations, ref_date):
    df_pred = load_pred_result_files(basedir, model_desc, locations)
    df_pred = df_pred[df_pred['reference_date']==ref_date.strftime("%Y-%m-%d")]
    df_pred = pd.concat([df_pred,df_pred_peak])
    output_dir = "{}/{}".format(basedir, model_desc)
    df_pred.to_csv("{}/{}-CU-{}.csv".format(output_dir,format(ref_date,'%Y-%m-%d'),model_desc), index=False) 


def degenerate_em_weights(dist_cond_likelihood, init_weights=None, obs_weights=None, models_name=None, tol_stop = 1e-13):
    """ Degenerate expectation maximization weights.
        Compute weights to create a posterior distribution averaging over given set of quantiles.

    Args:
        dist_cond_likelihood (_type_): P_k(wi | hi), k in [1, num_models], i in [1, num_quantiles] [K, Q]
        init_weights         (_type_): Naive weights.
        tol_stop             (_type_): _description_. Defaults to 1e-13.
    """

    # Return number of observations and number of models.
    num_obs, num_model = dist_cond_likelihood.shape
    if obs_weights:
        obs_weights = np.array(obs_weights)
        obs_weights = obs_weights / np.sum(obs_weights)
    else:
        # Initialize all weights equally
        obs_weights = np.ones(num_obs) * 1/num_obs

    if init_weights:
        weights = init_weights
    else:
        # Initialize all weights equally
        weights = np.ones((num_obs, num_model)) * 1 / num_model

    lkhds     = weights * dist_cond_likelihood # Likelihoods   | [num_obs, num_models]
    marg_dist = np.sum(lkhds.to_numpy(), axis=-1) # Marginal dist | [num_obs]

    # Average log likelihood across observation space.
    log_lklhd     = np.average(np.log(marg_dist), weights=obs_weights)
    old_log_lklhd = -1000

    while log_lklhd > old_log_lklhd and (log_lklhd-old_log_lklhd >= tol_stop): #or ((log_lklhd - old_log_lklhd) / -log_lklhd >= tol_stop):

        old_log_lklhd  = log_lklhd # Save new log-likelihood value.
        weights        = np.divide(lkhds, np.expand_dims(marg_dist, -1))
        weights        = np.average(weights, weights=obs_weights, axis=0) # Recompute weights | [num_models]
        # Recompute likelihoods
        lkhds          = weights * dist_cond_likelihood                     # Likelihoods    | [num_obs, num_models]
        marg_dist      = np.sum(lkhds.to_numpy(), axis=-1)                  # Marginal dist  | [num_obs]
        log_lklhd      = np.average(np.log(marg_dist), weights=obs_weights) # Log-likelihood | Scalar

    w_df           = pd.DataFrame(columns=["weight", "model_name"])
    w_df["weight"] = weights
    if models_name:
        w_df["model_name"] = models_name
    else:
        w_df["model_name"] = [f"model_{str(i)}" for i in range(len(w_df))]

    return w_df.set_index("model_name")