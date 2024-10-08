from utilities import *
import numpy as np
data = unpickle_me_this("data_storage/Integration. Num of Pixels (pl, moon, sm) = (10, 10, 10); omega_init (moon, pl, star) = [2.669516160131521e-06, 5.555555555555556e-05, 4.6296296296296297e-07].pickle")

results, termination_reason_counter, lifetimes, physically_varied_ranges, computation_time = data
showcase_results(data, plot=True, suppress_text=False)
