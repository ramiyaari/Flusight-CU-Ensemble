import numpy as np
import pandas as pd
# from scipy.stats import norm

from utils import *

def compute_quantiles_from_ensemble(ens, quantiles):
    """
    Compute specified quantiles from the weekly infection ensemble.

    Parameters:
        ens: np.ndarray
            A 2D array of shape (weeks, num_ensembles) representing weekly infection data.
        quantiles: list
            A list of quantiles to compute (e.g., [0.025, 0.5, 0.975]).

    Returns:
        quantile_df: pd.DataFrame
            A DataFrame containing quantiles for each week.
            Columns include ["week"] + quantile columns (e.g., "2.5%", "50%", "97.5%").
    """
    # Ensure quantiles are valid
    if not all(0 <= q <= 1 for q in quantiles):
        raise ValueError("Quantiles must be between 0 and 1.")

    quantile_labels = [f"{q * 100:.1f}%" for q in quantiles]

    # Compute quantiles across ensembles for each week
    quantile_values = np.percentile(ens, [q * 100 for q in quantiles], axis=1).T

    # Create a DataFrame with week indices and quantile columns
    quantile_df = pd.DataFrame(quantile_values, columns=quantile_labels)
    quantile_df.insert(0, "week", np.arange(ens.shape[0]))
    return quantile_df


def remove_outliers(ensemble, threshold_factor=4):
    """
    Check if certain values in the ensemble are too far off from the others
    (e.g., more than `threshold_factor` times the std from the mean).
    If they are, set them to the mean of the ensemble.

    Parameters:
        ensemble: np.ndarray
            1D array representing the ensemble for a particular state or parameter.
        threshold_factor: float
            Factor of the standard deviation used to determine outliers (default: 3).

    Returns:
        ensemble: np.ndarray
            The updated ensemble with outliers replaced by the mean.
    """
    mean_value = np.mean(ensemble)
    std_value = np.std(ensemble)

    # Identify outliers
    lower_bound = mean_value - threshold_factor * std_value
    upper_bound = mean_value + threshold_factor * std_value
    outliers = (ensemble < lower_bound) | (ensemble > upper_bound)

    # Replace outliers with the mean
    ensemble[outliers] = mean_value
    return ensemble


# SIR model equations
def sir_step(S, I, beta, gamma, N):
    new_infected = beta * S * I / N
    new_recovered = gamma * I

    S_next = S - new_infected
    I_next = I + new_infected - new_recovered

    # Ensure no negative values
    S_next = np.maximum(S_next, 0)
    I_next = np.maximum(I_next, 0)

    return S_next, I_next, new_infected


def eakf_update(state_ensembles, obs_ensemble, obs_truth, obs_var,
                min_values=None, max_values=None, inflation_factor=1.02):
    """
    Perform EAKF update with covariance-based adjustments, inflation, and bounds enforcement.

    Parameters:
        state_ensembles: np.ndarray
            A 2D array of shape (num_states, num_ensembles), where each row represents a state
            or parameter ensemble (e.g., S, I, beta, gamma).
        obs_ensemble: np.ndarray
            1D array representing the ensemble of the observed variable (e.g., weekly infections).
        obs_truth: float
            The observed value (truth) to assimilate.
        obs_var: float
            The variance of the observation.
        min_values: list or None
            Minimum bounds for each state/parameter (default: None).
        max_values: list or None
            Maximum bounds for each state/parameter (default: None).
        inflation_factor: float
            Factor to inflate the ensemble (default: 1.02).

    Returns:
        updated_state_ensembles: np.ndarray
            The updated state ensembles after assimilation, inflation, and bounds enforcement.
    """
    # Calculate prior statistics
    prior_mean = np.mean(obs_ensemble)
    prior_var = np.var(obs_ensemble)

    if prior_var == 0:
        prior_var = 1e-3  # Avoid division by zero
        post_var = 0
    else:
        post_var = prior_var * obs_var / (prior_var + obs_var)

    # Calculate posterior mean
    post_mean = post_var * (prior_mean / prior_var + obs_truth / obs_var)

    # Compute alpha (adjustment factor)
    alpha = np.sqrt(obs_var / (obs_var + prior_var))

    # Compute dy (ensemble adjustments for observed variable)
    dy = post_mean + alpha * (obs_ensemble - prior_mean) - obs_ensemble

    # Update each state variable based on covariance with obs_ensemble
    num_states = state_ensembles.shape[0]
    updated_state_ensembles = state_ensembles.copy()

    for i in range(num_states):
        # Calculate covariance between current state and obs_ensemble
        cov = np.cov(state_ensembles[i, :], obs_ensemble)[0, 1]
        rr = cov / prior_var  # Scaling factor

        # Apply update
        updated_state_ensembles[i, :] += rr * dy

    # Inflate ensemble to maintain variability
    for i in range(num_states):
        state_mean = np.mean(updated_state_ensembles[i, :])
        deviations = updated_state_ensembles[i, :] - state_mean
        updated_state_ensembles[i, :] = state_mean + inflation_factor * deviations
    
    # Enforce bounds after inflation
    if min_values is not None and max_values is not None:
        for i in range(num_states):
            lower_bound = min_values[i]
            upper_bound = max_values[i]
            ensemble = updated_state_ensembles[i, :]

            # Identify out-of-bound values
            out_of_bounds = (ensemble < lower_bound) | (ensemble > upper_bound)
            if np.any(out_of_bounds):
                # Replace out-of-bound values with the mean
                ensemble[out_of_bounds] = np.mean(ensemble)

            ensemble = remove_outliers(ensemble)
            ensemble = np.clip(ensemble,lower_bound,upper_bound)
            updated_state_ensembles[i, :] = ensemble

    return updated_state_ensembles


# Main SIR-EAKF model
def sir_eakf(num_ensembles, N, S_min, S_max, I_min, I_max, 
             beta_min, beta_max, gamma_min, gamma_max,
             weeks_to_predict, obs_data, obs_var=None,
             reporting_factor=1.0, inflation_factor=1.05):
    """
    Perform SIR modeling with EAKF assimilation on observed data, then forecast weeks_to_predict.

    Parameters:
        num_ensembles: int
            Number of ensemble members.
        N: int
            Total population size.
        S_min, S_max: float
            Minimum and maximum values for initializing S.
        I_min, I_max: float
            Minimum and maximum values for initializing I.
        beta_min, beta_max: float
            Minimum and maximum values for initializing beta.
        gamma_min, gamma_max: float
            Minimum and maximum values for initializing gamma.
        weeks_to_predict: int
            Number of weeks to predict.
        obs_data: np.ndarray
            Weekly observed infection data (length corresponds to available observations).
        reporting_factor: float
            Reporting factor from I_weekly to obs_data (default: 1.0).
        inflation_factor: float
            Factor to inflate the ensemble (default: 1.05).

    Returns:
        weekly_I_obs: np.ndarray
            A 2D array of shape (obs_len + weeks_to_predict, num_ensembles) containing observed weekly infections.
        beta_ensemble: np.ndarray
            A 2D array of shape (obs_len + weeks_to_predict, num_ensembles) containing beta values.
        gamma_ensemble: np.ndarray
            A 2D array of shape (obs_len + weeks_to_predict, num_ensembles) containing gamma values.
    """
    # Initialize ensembles
    S = np.random.uniform(S_min, S_max, num_ensembles)
    I = np.random.uniform(I_min, I_max, num_ensembles)
    beta_ensemble = np.random.uniform(beta_min, beta_max, num_ensembles)
    gamma_ensemble = np.random.uniform(gamma_min, gamma_max, num_ensembles)

    min_values = [0, 0, beta_min, gamma_min]
    max_values = [N, S_max, beta_max, gamma_max]

    obs_len = len(obs_data)

    if(obs_var is None):
        obs_var = 1e-5 + obs_data**2/100


    # Track weekly infections over time
    weekly_I = np.zeros((obs_len + weeks_to_predict, num_ensembles))

    # Assimilate observed data
    for t in range(obs_len):
        # Run the SIR model for one week (7 days)
        for _ in range(7):

            # Run SIR for one time step
            S, I, new_infected = sir_step(S, I, beta_ensemble, gamma_ensemble, N)

            # Accumulate weekly infections
            weekly_I[t, :] += new_infected

        # weekly_I_obs = weekly_I[t, :]*reporting_factor
        weekly_I_obs = np.random.binomial(weekly_I.astype(int), reporting_factor)

        # if(t>5):
        #     weekly_I_mean = np.mean(weekly_I_obs, axis=1)
        #     past_errors = obs_data[:t] - weekly_I_mean[:t]
        #     obs_based_var = np.mean(past_errors**2)
        #     spread = np.std(weekly_I_obs[t, :])
        #     spread_based_var = 0.5 * spread**2
        #     obs_var[t] = 0.2*spread_based_var + 0.8*obs_based_var

        print(f"Week {t}: obs_var={np.mean(obs_var[t]):.4f}")

        # EAKF update
        state_ensembles = np.array([S, I, beta_ensemble, gamma_ensemble])
        updated_ensembles = eakf_update(
            state_ensembles, weekly_I_obs[t, :], obs_data[t], obs_var[t], min_values, max_values, inflation_factor
        )

        # Unpack updated state variables and parameters
        S, I, beta_ensemble, gamma_ensemble = updated_ensembles

        # print(f"Week {t}: beta={np.mean(beta_ensemble):.4f}, gamma={np.mean(gamma_ensemble):.4f}, I={np.mean(I):.4f}, S={np.mean(S):.4f}")
    

    # Forecast weeks_to_predict from the last observed week
    for t in range(obs_len, obs_len + weeks_to_predict):
        for _ in range(7):

            # Run SIR for one time step
            S, I, new_infected = sir_step(S, I, beta_ensemble, gamma_ensemble, N)

            # Accumulate weekly infections
            weekly_I[t, :] += new_infected

        weekly_I[t, :] = remove_outliers(weekly_I[t, :])

    weekly_I_obs = weekly_I*reporting_factor
    # weekly_I_obs = np.random.binomial(weekly_I.astype(int), reporting_factor)
    return weekly_I_obs


def generate_sir_eakf_pred(df, start_date, ref_date, weeks_to_predict, locations, quantiles, num_samples, 
                           dat_changerate_ref, basedir, model_desc='SIR-EAKF'):
    
    N = 1e5 
    S_min, S_max = 0.2 * N, 0.8 * N #0.2 * N, 0.6 * N #
    # I_min, I_max = 1, 200
    beta_min, beta_max = 0.5, 1 
    gamma_min, gamma_max = 0.2, 0.5
    reporting_factor = 0.003
    inflation_factor = 1.05

    quantile_labels = [f"{q * 100:.1f}%" for q in quantiles]

    # Ensure dates are datetime
    df['date'] = pd.to_datetime(df['date'])
    ref_date = pd.to_datetime(ref_date)

    # Filter data from the start date up to (but not including) the reference date
    past_data = df[(df['date'] >= start_date) & (df['date'] < ref_date)].copy()
    past_data = past_data.drop(columns=['date','year','week'])

    # populations = locations['population'].reindex(past_data.columns)
    # normalized_data = past_data.div(populations, axis=1) * N
    # obs_var = normalized_data.var(axis=1).to_numpy()
    obs_var = None

    pred_results = []

    # Loop through each location
    locations_abbr = locations.index 
    for loc_abbr in locations_abbr:

        location = locations.loc[loc_abbr].location
        pop_size = locations.loc[loc_abbr].population
        ref_val = dat_changerate_ref[loc_abbr].values[0]

        # Historical data for the location
        obs_data = past_data[loc_abbr].values
        obs_data = obs_data/pop_size*N
        obs_len = len(obs_data)

        I0 = int(obs_data[0]/7/reporting_factor)
        I_min = int(max(1,I0/10))
        I_max = max(I0*10,10)
        # print(f"Location {loc_abbr}: I_min={I_min}, I_max={I_max}")

        obs_I_ens = sir_eakf(num_samples, N, S_min, S_max, I_min, I_max, 
                             beta_min, beta_max, gamma_min, gamma_max, 
                             weeks_to_predict, obs_data, obs_var, 
                             reporting_factor, inflation_factor)
        
        obs_I_ens = obs_I_ens*pop_size/N
        quantile_df = compute_quantiles_from_ensemble(obs_I_ens, quantiles)

        for horizon in range(weeks_to_predict):
            target_date = ref_date + timedelta(weeks=horizon)
            week = obs_len + horizon
            for i, quantile in enumerate(quantiles):
                predicted_value = quantile_df.loc[(quantile_df['week']==week),quantile_labels[i]].values[0]
                pred_results.append([format(ref_date,'%Y-%m-%d'),'wk inc flu hosp',horizon,format(target_date,'%Y-%m-%d'),
                                     location,'quantile',np.round(quantile,3),np.round(predicted_value,3)])

            pred_vals = obs_I_ens[week,:]
            pred_qual = generate_qualtitative_pred(ref_date, location, horizon, pred_vals, ref_val, pop_size)
            pred_results = pred_results+pred_qual

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