from utilities import *
import numpy as np

data_name = "warm_jupiter_like_integration_2025-03-19 10:12:20.117477.pickle"
fn = data_name[:-14]  # cut the .pickle ending and the microseconds
data = unpickle_me_this(f"data_storage/{data_name}")

results, termination_reason_counter, lifetimes, physically_varied_ranges, computation_time = data
showcase_results(data, suppress_text=False, plot_initial_states=True, plot_final_states=True, save=True,
                 filename=fn)
