import numpy as np
import pandas as pd
# from scipy.stats import norm
import matplotlib.pyplot as plt

from utils import *


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_means_vars_cv_from_df(df: pd.DataFrame, state_names, location, out_dir="."):
    work = df.copy()
    eps = 1e-12

    # Build long tidy table
    long_rows = []
    for p in state_names:
        mcol = f"{p}_post_mean"
        prior_col = f"{p}_prior_var"
        post2_col = f"{p}_post2_var"

        m = pd.to_numeric(work[mcol], errors="coerce") if mcol in work.columns else pd.Series([np.nan]*len(work))
        prior = pd.to_numeric(work[prior_col], errors="coerce") if prior_col in work.columns else pd.Series([np.nan]*len(work))
        post2 = pd.to_numeric(work[post2_col], errors="coerce") if post2_col in work.columns else pd.Series([np.nan]*len(work))

        sd = np.sqrt(np.clip(post2.values, a_min=0, a_max=None))
        cv = sd / np.maximum(np.abs(m.values), eps)

        long_rows.append(pd.DataFrame({
            "time": work["time"],
            "variable": p,
            "mean": m,
            "prior_variance": prior,
            "post_variance": post2,          
            "cv": cv
        }))

    long_df = (pd.concat(long_rows, ignore_index=True)
               if long_rows else pd.DataFrame(columns=["time","variable","mean","prior_variance","post_variance","cv"]))
    long_df = long_df.sort_values("time").reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)

    def _figsize(nvars):
        # a little auto-scaling so labels don't cram
        return (10, max(4, 2.4 * nvars))

    def _get_axes(nvars, *, sharey=False):
        fig, axes = plt.subplots(nvars, 1, sharex=True, sharey=sharey, figsize=_figsize(nvars))
        # Normalize axes to an array
        if nvars == 1:
            axes = np.array([axes])
        return fig, axes

    # Mean
    def _plot_mean():
        if long_df.empty: return None
        vars_sorted = sorted(long_df["variable"].unique())
        nvars = len(vars_sorted)
        fig, axes = _get_axes(nvars)
        for idx, var in enumerate(vars_sorted):
            ax = axes[idx]
            sub = long_df[long_df["variable"] == var]
            ax.plot(sub["time"], sub["mean"])
            ax.set_title(var)
            ax.set_ylabel("Mean")
            if idx == nvars - 1:
                ax.set_xlabel("week")
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"vars_mean_{location}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # Variance
    def _plot_variance_prior_post():
        if long_df.empty: return None
        vars_sorted = sorted(long_df["variable"].unique())
        nvars = len(vars_sorted)
        fig, axes = _get_axes(nvars, sharey=False)
        for idx, var in enumerate(vars_sorted):
            ax = axes[idx]
            sub = long_df[long_df["variable"] == var]
            ax.plot(sub["time"], sub["prior_variance"], label="prior_var")
            ax.plot(sub["time"], sub["post_variance"], label="posterior_var")
            ax.set_title(var)
            ax.set_ylabel("Variance")
            if idx == nvars - 1:
                ax.set_xlabel("week")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"vars_var_{location}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # CV
    def _plot_cv():
        if long_df.empty: return None
        vars_sorted = sorted(long_df["variable"].unique())
        nvars = len(vars_sorted)
        fig, axes = _get_axes(nvars)
        for idx, var in enumerate(vars_sorted):
            ax = axes[idx]
            sub = long_df[long_df["variable"] == var]
            ax.plot(sub["time"], sub["cv"])
            ax.set_title(var)
            ax.set_ylabel("Coefficient of Variation")
            if idx == nvars - 1:
                ax.set_xlabel("week")
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"vars_cv_{location}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    _plot_mean()
    _plot_variance_prior_post()  
    _plot_cv()

# def plot_means_vars_cv_from_df(df: pd.DataFrame, state_names, location, out_dir="."):
    
#     work = df.copy()
#     eps = 1e-12

#     # Build long tidy table
#     long_rows = []
#     for p in state_names:
#         mcol = f"{p}_post_mean"
#         vcol = f"{p}_post2_var"
#         m = pd.to_numeric(work[mcol], errors="coerce") if mcol in work.columns else pd.Series([np.nan]*len(work))
#         v = pd.to_numeric(work[vcol], errors="coerce") if vcol in work.columns else pd.Series([np.nan]*len(work))
#         sd = np.sqrt(np.clip(v.values, a_min=0, a_max=None))
#         cv = sd / np.maximum(np.abs(m.values), eps)
#         long_rows.append(pd.DataFrame({
#             "time": work["time"],
#             "variable": p,
#             "mean": m,
#             "variance": v,
#             "cv": cv
#         }))
#     long_df = (pd.concat(long_rows, ignore_index=True)
#                if long_rows else pd.DataFrame(columns=["time","variable","mean","variance","cv"]))
#     long_df = long_df.sort_values("time").reset_index(drop=True)

#     os.makedirs(out_dir, exist_ok=True)

#     def _plot(metric_col: str, ylabel: str, filename: str):
#         if long_df.empty:
#             return None
#         vars_sorted = sorted(long_df["variable"].unique())
#         nvars = len(vars_sorted)
#         if nvars == 0:
#             return None

#         fig, axes = plt.subplots(nvars, 1, sharex=True, sharey=False, figsize=(10, 10))
#         for idx, var in enumerate(vars_sorted):
#             ax = axes[idx]
#             sub = long_df[long_df["variable"] == var]
#             ax.plot(sub["time"], sub[metric_col])
#             ax.set_title(var)
#             ax.set_ylabel(ylabel)
#             if idx == nvars - 1:
#                 ax.set_xlabel("week")
#             # light grid helps read-off
#             ax.grid(True, alpha=0.25)

#         fig.tight_layout()
#         out_path = os.path.join(out_dir, filename)
#         fig.savefig(out_path, dpi=150)
#         plt.close(fig)
#         return out_path

#     _plot("mean", "Mean", f"vars_mean_{location}.png")
#     _plot("variance", "Variance", f"vars_var_{location}.png")
#     _plot("cv", "Coefficient of Variation", f"vars_cv_{location}.png")

def plot_correlations(corr: np.ndarray,
                      times=None,
                      labels=None,
                      title=None,
                      rolling=None,  
                      save_path=None):
    """
    corr: (T, K) array of correlations in [-1,1]
    times: length-T array-like (ints or datetimes). If None, uses range(T)
    labels: length-K iterable of series names
    rolling: window size for optional rolling mean overlay (int or None)
    save_path: path to save the figure (PNG). If None, just shows it.
    """
    corr = np.asarray(corr)
    assert corr.ndim == 2, "corr must be 2D (T x K)"
    T, K = corr.shape
    if times is None:
        times = np.arange(T)
    else:
        times = np.asarray(times)

    # Mask impossible values or NaNs
    corr = np.where(np.isfinite(corr), corr, np.nan)
    corr = np.clip(corr, -1.0, 1.0)

    plt.figure(figsize=(10, 5))
    for k in range(K):
        y = corr[:, k]
        plt.plot(times, y, linewidth=1.5, label=labels[k] if k < len(labels) else f"var{k}")

        # Optional rolling mean overlay
        if rolling and rolling > 1 and rolling < T:
            # simple centered rolling; pad ends with NaN
            pad = rolling // 2
            kernel = np.ones(rolling) / rolling
            y_roll = np.convolve(np.where(np.isnan(y), 0.0, y), kernel, mode="same")
            # naive NaN handling: mask positions where original had too many NaNs in window
            isn = np.isnan(y)
            valid_counts = np.convolve((~isn).astype(float), np.ones(rolling), mode="same")
            y_roll[valid_counts < max(1, rolling//2)] = np.nan
            plt.plot(times, y_roll, linestyle="--", linewidth=1.0)

    # Reference lines
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.axhline(1.0, linestyle=":", linewidth=0.8)
    plt.axhline(-1.0, linestyle=":", linewidth=0.8)

    plt.ylim(-1.05, 1.05)
    plt.xlabel("Time")
    plt.ylabel("Correlation")
    plt.title(title)
    plt.legend(ncol=min(K, 4))
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

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


def remove_outliers(ensemble, threshold_factor=3):
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
    """
    Deterministic RK4 SIR step.
    """
    def rhs(S_, I_):
        Einf     = np.clip(beta * S_ * I_ / N, 0.0, S_)
        Erecov   = np.clip(I_ * gamma, 0.0, I_)
        dS = - Einf
        dI = Einf - Erecov
        return dS, dI, Einf

    s1, i1, i1i = rhs(S,            I)
    s2, i2, i2i = rhs(S + s1/2.0,   I + i1/2.0)
    s3, i3, i3i = rhs(S + s2/2.0,   I + i2/2.0)
    s4, i4, i4i = rhs(S + s3,       I + i3)

    S += (s1 + 2*s2 + 2*s3 + s4) / 6.0
    I += (i1 + 2*i2 + 2*i3 + i4) / 6.0
    new_infected = (i1i + 2*i2i + 2*i3i + i4i) / 6.0
    return S, I, new_infected 


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

    K_vals = np.empty(num_states, dtype=float)
    prior_var_vals = np.empty(num_states, dtype=float)
    post_var_vals = np.empty(num_states, dtype=float)
    post2_var_vals = np.empty(num_states, dtype=float)

    for i in range(num_states):
        # Calculate covariance between current state and obs_ensemble
        cov = np.cov(state_ensembles[i, :], obs_ensemble)[0, 1]
        K = cov / prior_var  # Kalman gain
        # Apply update
        updated_state_ensembles[i, :] += K * dy

        # save states for debugging
        K_vals[i] = K
        prior_var_vals[i] = np.var(state_ensembles[i, :])
        post_var_vals[i] = np.var(updated_state_ensembles[i, :])

        # Inflate ensemble to maintain variability
        state_mean = np.mean(updated_state_ensembles[i, :])
        deviations = updated_state_ensembles[i, :] - state_mean
        updated_state_ensembles[i, :] = state_mean + inflation_factor * deviations
        
        # Enforce bounds after inflation
        if min_values is not None and max_values is not None:
            lower_bound = min_values[i]
            upper_bound = max_values[i]
            updated_state_ensembles[i, :] = np.clip(updated_state_ensembles[i, :],lower_bound,upper_bound)
            # # Identify out-of-bound values
            #out_of_bounds = (ensemble < lower_bound) | (ensemble > upper_bound)
            #if np.any(out_of_bounds):
            #    # Replace out-of-bound values with the mean
            #    ensemble[out_of_bounds] = np.mean(ensemble)
            # ensemble = remove_outliers(ensemble)

        post2_var_vals[i] = np.var(updated_state_ensembles[i, :])


    return updated_state_ensembles, K_vals, prior_var_vals, post_var_vals, post2_var_vals


# Main SIR-EAKF model
def sir_eakf(num_ensembles, N, S_min, S_max, I_min, I_max, 
             gamma, Reff_min, Reff_max, rho_min, rho_max,
             weeks_to_predict, obs_data, obs_var=None,
             inflation_factor=1.02, location='NA', random_state=None):
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
        pred_vals: np.ndarray
            A 2D array of shape (weeks_to_predict, num_ensembles) containing observed weekly infections.
    """

    rng = np.random.default_rng(random_state)

    
  
    # Initialize ensembles
    S = rng.uniform(S_min, S_max, num_ensembles)
    I = rng.uniform(I_min, I_max, num_ensembles)
    Reff = rng.uniform(Reff_min, Reff_max, num_ensembles)
    beta_ensemble = Reff*gamma/S*N
    rho_ensemble = rng.uniform(rho_min, rho_max, num_ensembles)

    beta_min = Reff_min*gamma/S_max*N
    beta_max = Reff_max*gamma/S_min*N

    min_values = [0, 0, beta_min, rho_min]
    max_values = [N, N, beta_max, rho_max]

    obs_len = len(obs_data)

    calc_obs_var = (obs_var is None)
    if(calc_obs_var):
        # obs_var = 1e-5 + obs_data**2/100
        obs_var = np.zeros(obs_len,)
        use_nb = True
        k_nb = 100.0  # NegBin dispersion

    # Track weekly infections and observations (hospitalizations) over time
    weekly_I = np.zeros((obs_len + weeks_to_predict, num_ensembles))
    weekly_I_obs = np.zeros((obs_len + weeks_to_predict, num_ensembles))
    # print(f"Week 0: beta_mean={np.mean(beta_ensemble):.4f}, beta_std={np.std(beta_ensemble):.4f}, gamma_mean={np.mean(gamma_ensemble):.4f}, gamma_std={np.std(gamma_ensemble):.4f}")

    save_debug_info = False
    if(save_debug_info):
        state_names = ["S", "I", "beta", "rho"] 
        calc_types   = ["kalman_gain", "prior_var", "post_var", "post2_var", "post_mean"]
        calc_rows = []
        corr_mat = np.zeros((obs_len,4))

    # Assimilate observed data
    for t in range(obs_len):
        # Run the SIR model for one week (7 days)
        for _ in range(7):
            # Run SIR for one time step
            S, I, new_infected = sir_step(S, I, beta_ensemble, gamma, N)
            # Accumulate weekly infections
            weekly_I[t, :] += new_infected

        weekly_I_obs[t, :] = weekly_I[t, :]*rho_ensemble
        # weekly_I_obs[t,:] = remove_outliers(weekly_I_obs[t,:])

        if(calc_obs_var):
            mu = weekly_I_obs[t, :]
            if(use_nb):
                obs_var[t] = np.maximum((mu + (mu**2)/k_nb).mean(), 1e-5)
            else:
                obs_var[t] = np.maximum(mu.mean(), 1e-5)
                # rho = rho_ensemble.mean()
                # obs_var[t] = (rho * (1.0 - rho) * np.maximum(mu.mean(), 1.0))

        # EAKF update
        state_ensembles = np.array([S, I, beta_ensemble, rho_ensemble])
        updated_ensembles, K_vals, prior_var, post_var, post2_var = eakf_update(
            state_ensembles, weekly_I_obs[t, :], obs_data[t], obs_var[t], min_values, max_values, inflation_factor
        )
        # Unpack updated state variables and parameters
        S, I, beta_ensemble, rho_ensemble = updated_ensembles

        if(save_debug_info):
            row = {"time": t}
            post_mean = [S.mean(), I.mean(), beta_ensemble.mean(), rho_ensemble.mean() ]
            vals = {"kalman_gain": K_vals, "prior_var": prior_var, "post_var": post_var, "post2_var": post2_var, "post_mean": post_mean}
            for s_idx, s_name in enumerate(state_names):
                for vt in calc_types:
                    row[f"{s_name}_{vt}"] = float(vals[vt][s_idx])
            calc_rows.append(row)
            corr_mat[t,0] = np.corrcoef(S,weekly_I_obs[t, :])[0,1]
            corr_mat[t,1] = np.corrcoef(I,weekly_I_obs[t, :])[0,1]
            corr_mat[t,2] = np.corrcoef(beta_ensemble,weekly_I_obs[t, :])[0,1]
            corr_mat[t,3] = np.corrcoef(rho_ensemble,weekly_I_obs[t, :])[0,1]

        # print(f"Week {t}: beta={np.mean(beta_ensemble):.4f}, gamma={np.mean(gamma_ensemble):.4f}, I={np.mean(I):.4f}, S={np.mean(S):.4f}")
    
    if(save_debug_info):
        df_calc = pd.DataFrame(calc_rows)
        ordered_cols = (["time"] + [f"{s}_{vt}" for s in state_names for vt in calc_types])
        df_calc = df_calc[ordered_cols]
        df_calc.to_csv(f"./tmp/kalman_calc_{location}.csv", index=False)
        plot_means_vars_cv_from_df(df_calc,state_names,location,"./tmp")
        plot_correlations(corr_mat, labels=state_names, title=f'Correlation with computed observations {location}')


    # Forecast weeks_to_predict from the last observed week
    for t in range(obs_len, obs_len + weeks_to_predict):
        for _ in range(7):
            # Run SIR for one time step
            S, I, new_infected = sir_step(S, I, beta_ensemble, gamma, N)
            # Accumulate weekly infections
            weekly_I[t, :] += new_infected
        weekly_I_obs[t,:] = weekly_I[t, :]*rho_ensemble
        # weekly_I_obs[t,:] = remove_outliers(weekly_I_obs[t,:],2)

    pred_vals = weekly_I_obs[obs_len:,]
    return pred_vals


def generate_sir_eakf_pred(df, start_date, ref_date, weeks_to_predict, locations, quantiles, num_samples, 
                           dat_changerate_ref, basedir, model_desc='SIR-EAKF', 
                           generate_qual_pred=True, save_results=True,
                           return_samples=False, random_state=None):
    
    N = 1e5 
    S_min, S_max = 0.5 * N, 1 * N #0.2 * N, 0.8 * N #0.2 * N, 0.6 * N #
    # I_min, I_max = 1, 200
    gamma = 0.333
    Reff_min, Reff_max = 1.0, 1.4
    rho_min, rho_max = 0.001, 0.01 #0.003,0.003 #
    inflation_factor = 1.00

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
    samples_results = []        # will hold per-location samples in long format

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

        rho = ((rho_max+rho_min)/2)
        I0 = int(obs_data[0]/7/rho)
        I_min = int(max(1,I0/5))
        I_max = max(10,I0*5)
        # print(f"Location {loc_abbr}: I_min={I_min}, I_max={I_max}")

        pred_vals = sir_eakf(num_samples, N, S_min, S_max, I_min, I_max, 
                             gamma, Reff_min, Reff_max, rho_min, rho_max, 
                             weeks_to_predict, obs_data, obs_var, 
                             inflation_factor, location=loc_abbr, random_state=random_state)
        
        pred_vals = pred_vals*pop_size/N
        quantile_df = compute_quantiles_from_ensemble(pred_vals, quantiles)

        for horizon in range(weeks_to_predict):
            target_date = ref_date + timedelta(weeks=horizon)
            # week = obs_len + horizon
            for i, quantile in enumerate(quantiles):
                predicted_value = quantile_df.loc[(quantile_df['week']==horizon),quantile_labels[i]].values[0]
                pred_results.append([format(ref_date,'%Y-%m-%d'),'wk inc flu hosp',horizon,format(target_date,'%Y-%m-%d'),
                                     location,'quantile',np.round(quantile,3),np.round(predicted_value,3)])

            if(generate_qual_pred):
                pred_vals_h = pred_vals[horizon,:]
                pred_qual = generate_qualtitative_pred(ref_date, location, horizon, pred_vals_h, ref_val, pop_size)
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
                "value": pred_vals.reshape(-1)
            })
            samples_results.append(df_loc)

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
        df_pred_results.to_csv("{}/{}-CU-{}.csv".format(output_dir,format(ref_date,'%Y-%m-%d'),model_desc), index=False) 
    
    if return_samples:
        df_samples = pd.concat(samples_results, ignore_index=True) 
        return df_pred_results, df_samples
    
    return df_pred_results