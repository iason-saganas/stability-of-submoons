from utilities import *
import numpy as np

data_name = "integration_2024-11-17 21:51:02.236556.pickle"
fn = data_name[:-14]  # cut the .pickle ending and the microseconds
data = unpickle_me_this(f"data_storage/{data_name}")

results, termination_reason_counter, lifetimes, physically_varied_ranges, computation_time = data
showcase_results(data, suppress_text=False, plot_initial_states=True, plot_final_states=True, save=False,
                 filename=fn)
