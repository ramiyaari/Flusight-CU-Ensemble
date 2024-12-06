import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import epiweeks as ew
from datetime import timedelta #, datetime, date

import requests
# import warnings
import os



def date_to_epiweek(date):
    """
    This function takes a date and returns a tuple with the year and epidemiological week number.
    It uses the epiweeks package, which is designed to follow the CDC epidemiological week rules.
    """
    epi_week = ew.Week.fromdate(date)
    return epi_week.year, epi_week.week


def epiweek_to_dates(year, epiweek):
    dates = ew.Week(year, epiweek)
    return dates


def generate_pop_per_week(states, populations):
    years = list(range(min(populations['year']),max(populations['year'])))
    dates = []
    for year in years:
        for week in ew.Year(year).iterweeks():
            dates.append(week.startdate()+ timedelta(days=1))
    dates = pd.to_datetime(dates, format='%m/%d/%Y')
    df_pop = pd.DataFrame(index=dates,columns=states)
    for yind, year in enumerate(years):
        row_ind = df_pop[df_pop.index.year==year].index
        weeks_num = len(row_ind)
        for state in states:
            pop_start = populations[state].values[yind]
            pop_end = populations[state].values[yind+1]
            pop_weekly = (np.linspace(start=pop_start, stop=pop_end, num=weeks_num+1)).astype('int')
            df_pop.loc[row_ind,state] = pop_weekly[:-1]
    df_pop = df_pop.astype(np.double)
    return (df_pop)


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
    
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y' ) #format='%Y-%m-%d') #
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

    df_ili['date'] = pd.to_datetime(df_ili['date'], format='%Y-%m-%d') #format='%m/%d/%Y') #        
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
    locations = df.location.unique()
    horizons =  df.horizon.unique()
    for location in locations:
        for horizon in horizons:
            df_ens_hl = df[(df.target=='wk flu hosp rate change') & (df.location==location) & (df.horizon==horizon)]
            df_ens_hl.loc[:,'value'] = df_ens_hl['value']/np.sum(df_ens_hl['value'])
            df_ens_hl.loc[:,'value'] = np.round(df_ens_hl['value'],3)
            df_ens_hl.loc[df_ens_hl.output_type_id=='stable','value'] += np.round(1-np.sum(df_ens_hl['value']),3)
            df.loc[(df.target=='wk flu hosp rate change') & (df.location==location) & (df.horizon==horizon),'value'] = df_ens_hl['value']
    return (df)

def generate_qualtitative_pred(ref_date, location, horizon, samples, ref_val, pop_size):

    # Criteria for qualitative forecasts
    stable_criteria = [0.3, 0.5, 0.7, 1.0]
    change_criteria = [1.7, 3.0, 4.0, 5.0]

    num_samples = len(samples)

    cases_change = samples-ref_val
    rate_changes = cases_change/pop_size*1e5
    prob_stable = np.sum((np.abs(cases_change) < 10) |
                            ((rate_changes>-stable_criteria[horizon]) & 
                            (rate_changes<stable_criteria[horizon])))/num_samples
    prob_increase = np.sum((rate_changes>=stable_criteria[horizon]) & 
                            (rate_changes<change_criteria[horizon]))/num_samples
    prob_large_increase = np.sum(rate_changes>=change_criteria[horizon])/num_samples
    prob_decrease = np.sum((rate_changes<=-stable_criteria[horizon]) & 
                            (rate_changes>-change_criteria[horizon]))/num_samples 
    prob_large_decrease = np.sum(rate_changes<=-change_criteria[horizon])/num_samples

    # set minimum for rate change values and checks that add up to 1 after rounding
    prob_stable = np.clip(prob_stable,a_min=0.001, a_max=None)
    prob_increase = np.clip(prob_increase,a_min=0.001, a_max=None)
    prob_large_increase = np.clip(prob_large_increase,a_min=0.001, a_max=None)
    prob_decrease = np.clip(prob_decrease,a_min=0.001, a_max=None)
    prob_large_decrease = np.clip(prob_large_decrease,a_min=0.001, a_max=None)
    prob_total = prob_stable+prob_increase+prob_large_increase+prob_decrease+prob_large_decrease
    prob_stable = prob_stable + (1-prob_total)

    pred_results = []
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
    
    return pred_results


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
    df_ensemble['location'] = df_ensemble['location'].astype(str).str.zfill(2)

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


def generate_pred_weights(results_dict, df_metrics, locations, 
                          metric='wis', decay_rate=0.5, threshold_precentile=None):

    models = list(results_dict.keys())
    ref_dates = results_dict[models[0]]['reference_date'].unique()
    horizons = results_dict[models[0]]['horizon'].unique()

    # Filter and preprocess metrics data
    df_metrics = df_metrics[
        (df_metrics.metric == metric) & (df_metrics.model.isin(models))
    ].drop(columns=['metric'])
    df_metrics['target_date'] = pd.to_datetime(df_metrics['target_date'], format='%Y-%m-%d')

    df_weights = []
    for loc_abbr in locations.index:
        for horizon in horizons:
            # Filter metrics for location and horizon
            df_metrics_hl = df_metrics[
                (df_metrics.location == loc_abbr) & (df_metrics.horizon == horizon)
            ].drop(columns=['location', 'horizon'])
            
            # Pivot metrics and preprocess
            df_metrics_hl = df_metrics_hl.pivot(index='target_date', columns='model', values='value')
            df_metrics_hl = df_metrics_hl[models].fillna(1e4)  # Fill missing values

            for t in ref_dates[:horizon]:
                # For initial time window, assign uniform weights
                default_weights = np.array([1 / len(models)] * len(models))
                df_weights.append(
                    np.concatenate(([horizon, loc_abbr, t], default_weights))
                )

            for t in ref_dates[horizon:]:
                # Filter historical data for weight calculation
                df_metrics_hlt = df_metrics_hl[df_metrics_hl.index < t]
                # Multiply by exp_weights to give more weight to recent values
                exp_weights = np.exp(-decay_rate * np.arange(df_metrics_hlt.shape[0])[::-1])
                exp_weights /= np.sum(exp_weights)
                row_vals = df_metrics_hlt * exp_weights[:, np.newaxis]
                # Sum rows, compute reciprocal, and normalize
                row_vals = row_vals**2
                row_sums = row_vals.sum(axis=0)
                reciprocal_sums = 1 / np.clip(row_sums, 1, np.inf)  # Reciprocal of sums
                weights = reciprocal_sums / reciprocal_sums.sum()  # Normalize to sum to 1

                if(threshold_precentile is not None):
                    threshold = np.percentile(weights,threshold_precentile)
                    if(sum(weights > threshold)>0):
                        weights[weights <= threshold] = 0
                        weights = weights / weights.sum()  # Normalize to sum to 1

                df_weights.append(
                    np.concatenate(([horizon, loc_abbr, t], weights))
                )
    # Create final DataFrame
    cols = ['horizon', 'location', 'reference_date'] + models
    df_weights = pd.DataFrame(df_weights, columns=cols)
    return df_weights


def generate_weighted_pred_results(results_dict, df_weights, locations, 
                                   basedir, ensemble_name, average_over_horizons=False):

    if(average_over_horizons):
        df_weights = df_weights.groupby(['location','reference_date']).mean('weight').reset_index().drop(columns=['horizon'])     
    
    models = list(results_dict.keys())
    df_ensemble = merge_pred_results(results_dict)
    df_ensemble['value'] = np.round(df_ensemble[models].mean(axis=1), 3)

    ref_dates = df_ensemble['reference_date'].unique()
    horizons = df_ensemble.horizon.unique()

    for loc_abbr in locations.index:
        location = locations.loc[loc_abbr].location

        # Filter the ensemble for the current location
        df_ens_location = df_ensemble[
            (df_ensemble.location == location) & (df_ensemble.output_type == 'quantile')
        ]

        for horizon in horizons:
            # Filter for the current horizon
            df_ens_horizon = df_ens_location[df_ens_location.horizon == horizon]

            for t in ref_dates:
                # Retrieve weights
                if average_over_horizons:
                    weights = df_weights[
                        (df_weights.location == loc_abbr) & (df_weights.reference_date == t)
                    ]
                else:
                    weights = df_weights[
                        (df_weights.location == loc_abbr)
                        & (df_weights.horizon == horizon)
                        & (df_weights.reference_date == t)
                    ]

                if weights.empty:
                    continue

                # Filter for the reference date and compute weighted ensemble
                df_ens_hlt = df_ens_horizon[df_ens_horizon.reference_date == t][models]

                if not df_ens_hlt.empty:
                    quantile_vals = np.sum(
                        df_ens_hlt.values * weights[models].values.reshape(1, -1), axis=1
                    )
                    df_ensemble.loc[
                        (df_ensemble.location == location)
                        & (df_ensemble.horizon == horizon)
                        & (df_ensemble.reference_date == t)
                        & (df_ensemble.output_type == 'quantile'),
                        'value',
                    ] = np.round(quantile_vals, 3)

    # Drop model columns
    df_ensemble = df_ensemble.drop(labels=models, axis=1)

    # Round specific target values
    flu_hosp_mask = df_ensemble.target == 'wk inc flu hosp'
    df_ensemble.loc[flu_hosp_mask, 'value'] = np.round(df_ensemble.loc[flu_hosp_mask, 'value'], 0)

    df_ensemble['location'] = df_ensemble['location'].astype(str).str.zfill(2)

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




