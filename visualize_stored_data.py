from utilities import *

data_name = "warm_jupiter_like_integration_Case_A_1_8_m_earth_as_m_sm_more_realistic_sm_density.pickle"
# data_name = "warm_jupiter_like_integration_Case_A_submoon_mass_0_01_LUNA_mass.pickle"
# fn = data_name[:-14]  # cut the pickle ending and the microseconds
fn = data_name[:-7]  # cut only the .pickle ending


data = unpickle_me_this(f"data_storage/{data_name}")

results, termination_reason_counter, lifetimes, physically_varied0_ranges, computation_time = data
showcase_results(data, suppress_text=False, plot_initial_states=True, plot_final_states=True, save=False,
                 filename=fn, old=False, earth_like=False)




