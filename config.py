from utils import *

season = 2026 #2025 #2024 # 
epiyear = 2025 #2024 #
epiweek = 47 #15 #41 #
ref_date = epiweek_to_dates(epiyear, epiweek).enddate() #Saturday at the end of epiweek
ref_date = pd.Timestamp(ref_date)

data_dir = '../data/'
results_dir = '../results/'
figures_dir = f'../figures/{season}' 

locations_fname = data_dir +"locations.csv"
locations = pd.read_csv(locations_fname)
loc_name2abbr = dict(zip(locations['location_name'], locations['abbreviation']))
loc_name2loc = dict(zip(locations['location_name'], locations['location']))
locations = locations.set_index('abbreviation')
abbr2loc = dict(zip(locations.index, locations['location']))
loc2abbr = dict(zip(locations['location'],locations.index))
pop_per_loc_abbr = pd.Series(dict(zip(locations.index, locations['population'])))
pop_per_loc = pd.Series(dict(zip(locations['location'], locations['population'])))

states = locations.index.values
num_states = len(states)

# populations_fname = data_dir +"populations.csv"
# populations = pd.read_csv(populations_fname)
# df_pop = generate_pop_per_week(states, populations)

AH_daily, df_AH = read_AH(data_dir)

num_samples = 1000
alpha_vals = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9]
quantiles = np.append(np.append([0.01,0.025],np.arange(0.05,0.95+0.05,0.050)),[0.975,0.99])
weeks_to_predict = 4