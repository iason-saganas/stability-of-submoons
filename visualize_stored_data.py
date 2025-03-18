from utilities import *
import numpy as np

data_name = "earth_like_integration_2025-03-18 13:28:34.819354.pickle"
fn = data_name[:-14]  # cut the .pickle ending and the microseconds
data = unpickle_me_this(f"data_storage/{data_name}")

results, termination_reason_counter, lifetimes, physically_varied_ranges, computation_time = data
showcase_results(data, suppress_text=False, plot_initial_states=True, plot_final_states=True, save=False,
                 filename=fn)
