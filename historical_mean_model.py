import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

from epiweeks import Week

import os


def generate_historical_mean_pred(df, states, quantiles, plot=False):

    # exclude_start = epiweek_to_dates(2020,26).enddate()
    # exclude_end = epiweek_to_dates(2022,25).enddate()
    # exclude_range = pd.date_range(start=exclude_start, end=exclude_end, freq='W-SAT')
    exclude_range = pd.DatetimeIndex([])

    df_states_list = []
    for state in states:
        df_state = df[['date','year','week',state]].copy()
        df_state['cases'] = df_state[state]
        df_state['state'] = state
        df_state = df_state.drop(state,axis=1)
        df_state.loc[df_state['date'].isin(exclude_range),'cases'] = np.nan

        # Initialize empty columns for mean and quantiles
        df_state['mean'] = np.nan
        for q in quantiles:
            df_state[f'q_{q:0.3f}'] = np.nan

        # Calculate rolling statistics up to each date for each week
        for week in df_state['week'].unique():
            # Filter data up to each date within each week
            if week == 53:
                # Combine data for weeks 52, 53, and 1 for week 53 calculations
                week_52_53_1_data = df_state[df_state['week'].isin([52, 53, 1])].copy()
                # week_52_53_1_data = week_52_53_1_data.dropna(subset=['cases'])  
                week_52_53_1_data['mean'] = week_52_53_1_data['cases'].expanding().mean()
                for q in quantiles:
                    week_52_53_1_data[f'q_{q:0.3f}'] = week_52_53_1_data['cases'].expanding().quantile(q)
                # Assign only the rows for week 53 back to the main DataFrame
                week_53_data = week_52_53_1_data[week_52_53_1_data['week'] == 53]
                df_state.loc[week_53_data.index, 'mean'] = week_53_data['mean']
                for q in quantiles:
                    df_state.loc[week_53_data.index, f'q_{q:0.3f}'] = week_53_data[f'q_{q:0.3f}']     
            else:
                weekly_data = df_state[df_state['week'] == week].copy()
                # Calculate cumulative mean and quantiles for each entry
                df_state.loc[weekly_data.index, 'mean'] = weekly_data['cases'].expanding().mean()
                for q in quantiles:
                    df_state.loc[weekly_data.index, f'q_{q:0.3f}'] = weekly_data['cases'].expanding().quantile(q)

        df_states_list.append(df_state)

        if(plot):
            plt.figure(figsize=(12,4))
            plt.plot(df_state['date'],df_state['cases'],label='data')
            plt.plot(df_state['date'],df_state['mean'],color='red',label='historical mean pred')
            plt.fill_between(df_state['date'], df_state['q_0.250'], df_state['q_0.750'], color='red', alpha=0.333)
            plt.fill_between(df_state['date'], df_state['q_0.025'], df_state['q_0.975'], color='red', alpha=0.1)
            plt.title("{}".format(state))
            plt.legend()

    df_his_pred = pd.concat(df_states_list, ignore_index=True) 
    return (df_his_pred)


def save_historical_mean_pred_results_to_file(df_his_pred, ref_date, weeks_to_predict, locations, 
                                              quantiles, basedir, model_desc="historical_mean"):

    df_his_pred = df_his_pred[df_his_pred['date']<=ref_date]
    pred_results = []
    locations_abbr = locations.index #
    for loc_abbr in locations_abbr:
        location = locations.loc[loc_abbr].location
        df_state = df_his_pred[df_his_pred.state==loc_abbr]
        for horizon in range(weeks_to_predict):
            target_date = ref_date + timedelta(weeks=horizon)
            year_week = Week.fromdate(target_date.date())
            epiyear = year_week.year-1 #use last year's values
            epiweek = year_week.week
            for quantile in quantiles:
                value = df_state.loc[(df_state['year']==epiyear) & (df_state['week']==epiweek),f"q_{quantile:0.3f}"].values[0]
                pred_results.append([format(ref_date,'%Y-%m-%d'),'wk inc flu hosp',horizon,
                                    format(target_date,'%Y-%m-%d'),location,'quantile',
                                    np.round(quantile,3),np.round(value,3)])
                
    for loc_abbr in locations_abbr:
        location = locations.loc[loc_abbr].location
        for horizon in range(weeks_to_predict):
            target_date = ref_date + timedelta(weeks=horizon)
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change',horizon,
                                    format(target_date,'%Y-%m-%d'),location,'pmf','large_decrease','NA'])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change',horizon,
                                    format(target_date,'%Y-%m-%d'),location,'pmf','decrease','NA'])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change',horizon,
                                    format(target_date,'%Y-%m-%d'),location,'pmf','stable','NA'])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change',horizon,
                                    format(target_date,'%Y-%m-%d'),location,'pmf','increase','NA'])
            pred_results.append([format(ref_date,'%Y-%m-%d'),'wk flu hosp rate change',horizon,
                                    format(target_date,'%Y-%m-%d'),location,'pmf','large_increase','NA'])

    df_pred_results = pd.DataFrame(pred_results,columns=['reference_date','target','horizon','target_end_date',
                                                         'location','output_type','output_type_id','value']) 
           
    output_dir = "{}/{}".format(basedir, model_desc)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    df_pred_results.to_csv("{}/{}-CU-{}.csv".format(output_dir,format(ref_date,'%Y-%m-%d'),model_desc), index=False) 
    return df_pred_results


