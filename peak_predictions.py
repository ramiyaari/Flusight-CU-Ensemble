import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scipy.stats as st
from scipy.stats import gaussian_kde

from datetime import timedelta 
from utils import *

   

def calculate_historical_stats(df, states, quantiles, populations):

    # Filter data for the given states
    df = df[df['state'].isin(states)]

    # Get rid of week 53
    df = df[~(df.week==53)]

    df_pop = pd.melt(populations,id_vars=['year'],value_vars=populations.columns[1:],var_name='state',value_name='pop')

    # Define a season starting from week 40
    df['season'] = df.apply(lambda row: get_season(row['year'], row['week']), axis=1)

    #remove covid seasons and current season
    exclude_seasons = ['2020-2021','2021-2022','2024-2025']
    df = df[~df['season'].isin(exclude_seasons)]

    # Calculate mean and quantiles across all years for each state and each week
    weekly_stats = df.groupby(['state', 'week'])['cases'].agg(
        mean='mean',
        **{f"quantile_{int(q * 100)}": lambda x, q=q: x.quantile(q) for q in quantiles}
    ).reset_index()

    # Group by state and season, find the week of the max value for each season
    # def get_seasonal_max(group):
    #     group = group.dropna(subset=['cases'])  # Drop rows where 'cases' is NaN
    #     if not group.empty:
    #         return group.loc[group['cases'].idxmax()]
    #     else:
    #         return None  # Return None if the group is empty

    # Group by state and season, find the week of the max value for each season
    def get_seasonal_max(group):
        group = group.dropna(subset=['cases'])  # Drop rows where 'cases' is NaN
        if not group.empty:
            max_row = group.loc[group['cases'].idxmax()]
            # Match population from df_pop using the year and state
            pop_row = df_pop.loc[(df_pop['year'] == max_row['year']) & (df_pop['state'] == max_row['state'])]
            if not pop_row.empty:
                max_row['pop'] = pop_row.iloc[0]['pop']  # Add population to the max_row
                max_row['rate1M'] = 1e6 * max_row['cases'] / max_row['pop']  # Calculate rate
            else:
                max_row['pop'] = None  # Set pop to None if no match is found
                max_row['rate1M'] = None  # Set rate to None if no match is found
            return max_row
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

    peak_rate1M_quantiles = seasonal_max.groupby('state')['rate1M'].quantile(quantiles).reset_index()
    peak_rate1M_quantiles.columns = ['state', 'quantile', 'rate1M']

    # Return results
    return {
        'weekly_stats': weekly_stats,
        'peak_value_quantiles': peak_value_quantiles,
        'peak_rate1M_quantiles': peak_rate1M_quantiles,
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


def fix_peak_week_prob_values(df):
    locations = df.location.unique()
    horizons =  df.horizon.unique()
    for location in locations:
        for horizon in horizons:
            df_ens_hl = df[(df.target=='peak week inc flu hosp') & (df.location==location) & (df.horizon==horizon)]
            # df_ens_hl.loc[:,'value'] = np.clip(df_ens_hl['value'],a_min=0.001,a_max=None)
            df_ens_hl.loc[:,'value'] = df_ens_hl['value']/np.sum(df_ens_hl['value'])
            df_ens_hl.loc[:,'value'] = np.round(df_ens_hl['value'],3)
            df_ens_hl.loc[df_ens_hl['value'].idxmax(),'value'] += np.round(1-np.sum(df_ens_hl['value']),3)
            df.loc[(df.target=='peak week inc flu hosp') & (df.location==location) & (df.horizon==horizon),'value'] = df_ens_hl['value']
    return (df)


def generate_peak_week_pred(df_his_stats, df_hosp, locations, ref_date, first_date, last_date, 
                            min_peak_date=None, kde_bandwith=0.5, plot_peak_week_prob=False):

    # df_peak_week_inc = df_his_stats['peak_value_quantiles'].copy()
    df_peak_week_rates = df_his_stats['peak_rate1M_quantiles'].copy()
    df_peak_week_prob = df_his_stats['peak_week_probabilities'].copy()
    df_peak_week_prob_combnined = df_his_stats['peak_week_probabilities_combined'].copy()

    pop_states = locations['population']
    def calculate_value_from_rate(row):
        pop = pop_states.get(row['state'])  
        return row['rate1M'] * pop / 1e6  
    
    df_peak_week_inc = df_peak_week_rates.assign(cases=df_peak_week_rates.apply(calculate_value_from_rate, axis=1))
    df_peak_week_inc = df_peak_week_inc[['state', 'quantile', 'cases']]

    epiweek_ref = date_to_epiweek(ref_date)[1]
    epiweek_first = date_to_epiweek(first_date)[1]
    epiweek_last = date_to_epiweek(last_date)[1]
    epiweek_season_ref = map_week_to_week_season(epiweek_ref)
    epiweek_season_first = map_week_to_week_season(epiweek_first)
    epiweek_season_last = map_week_to_week_season(epiweek_last)

    if(min_peak_date is None):
        min_peak_date = first_date
    epiweek_min = date_to_epiweek(min_peak_date)[1]
    epiweek_season_min = map_week_to_week_season(epiweek_min)

    df_hosp_season = df_hosp[df_hosp['date']>=first_date]

    locations_abbr = locations.index #pred.components
    pred_results = []

    # peek week incidence pred
    for loc_abbr in locations_abbr:
        location = locations.loc[loc_abbr].location
        df_peak_week_inc_state = df_peak_week_inc[df_peak_week_inc.state==loc_abbr]
        cur_max = df_hosp_season[loc_abbr].max()
        quantiles = df_peak_week_inc_state['quantile'].values
        for quantile in quantiles:
            value = df_peak_week_inc_state.loc[df_peak_week_inc_state['quantile']==quantile,'cases'].values[0]
            value = int(max(value,cur_max))
            pred_results.append([ref_date.strftime("%Y-%m-%d"),'peak inc flu hosp', '','',
                            location, 'quantile', np.round(quantile,3), value])
    
    # peek week prob pred
    for loc_abbr in locations_abbr:
        location = locations.loc[loc_abbr].location
        epiweek_curmax = df_hosp_season.loc[df_hosp_season[loc_abbr].idxmax(),'week']
        epiweek_season_curmax = map_week_to_week_season(epiweek_curmax)
        df_peak_week_prob_state = df_peak_week_prob[df_peak_week_prob.state==loc_abbr]
        if(df_peak_week_prob_state.shape[0]<5):
            print(f"Not enough data points for state {loc_abbr} - using combined data")
            df_peak_week_prob_state = df_peak_week_prob_combnined

        df_peak_week_prob_state.loc[((df_peak_week_prob_state.week_season<epiweek_season_ref) &
                                    (df_peak_week_prob_state.week_season!=epiweek_season_curmax)) |
                                    (df_peak_week_prob_state.week_season<epiweek_season_min),
                                    'probability'] = 0
        df_peak_week_prob_state.loc[:,'probability'] = df_peak_week_prob_state.loc[:,'probability']/df_peak_week_prob_state['probability'].sum()
        probs = fit_smooth_distribution_using_kde(df_peak_week_prob_state, bandwidth=kde_bandwith)
        probs.loc[((probs.week_season<epiweek_season_ref) &
                  (probs.week_season!=epiweek_season_curmax)) |
                  (probs.week_season<epiweek_season_min),'smooth_probability'] = 0

        probs['smooth_probability'] = probs['smooth_probability']/probs['smooth_probability'].sum()
        for week in range(epiweek_season_first,epiweek_season_last+1):
            week_date = (first_date+timedelta(weeks=(week-epiweek_season_first))).strftime("%Y-%m-%d")
            week_prob = probs.loc[probs.week_season==week,'smooth_probability'].values[0]
            pred_results.append([ref_date.strftime("%Y-%m-%d"),'peak week inc flu hosp', '','',
                            location, 'pmf', week_date, np.round(week_prob,5)])
            
        if(plot_peak_week_prob):
            df_peak_week_prob_state = df_peak_week_prob_state[df_peak_week_prob_state.week_season.isin(range(epiweek_season_first,epiweek_season_last+1))]
            probs = probs[probs.week_season.isin(range(epiweek_season_first,epiweek_season_last+1))]
            week_dates = [(first_date + timedelta(weeks=w)).strftime("%Y-%m-%d") for w in range(0,epiweek_season_last-epiweek_season_first+1)]
            plt.figure(figsize=(7,3))
            plt.bar(probs["week_season"], 
                    probs["smooth_probability"], 
                    label="Smooth Probabilities", alpha=0.7)
            plt.scatter(df_peak_week_prob_state['week_season'],
                        df_peak_week_prob_state['probability'],
                        color="red", label="Calculated Probabilities")
            plt.xticks(ticks=range(epiweek_season_first,epiweek_season_last+1), labels=week_dates,rotation=90)
            plt.xlabel("")
            plt.ylabel("Probability")
            # plt.legend()
            plt.title(f"Peak week distribution for {loc_abbr}")
            plt.show()

    df_results = pd.DataFrame(pred_results,columns=['reference_date','target','horizon','target_end_date',
                                                    'location','output_type','output_type_id','value'])
    df_results['reference_date'] = pd.to_datetime(df_results['reference_date'], format='%Y-%m-%d')
    df_results = df_results.astype({"output_type_id": str})
    df_results = fix_peak_week_prob_values(df_results)
    return df_results