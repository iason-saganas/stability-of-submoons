from utilities import *
import numpy as np
result = unpickle_me_this("data_storage/Integration. Num of Pixels (pl, moon, sm) = (30, 30, 30); omega_init (moon, pl, star) = [2.669516160131521e-06, 5.555555555555556e-05, 4.6296296296296297e-07].pickle")

results, termination_reason_counter, lifetimes, ranges = result
coordinates = np.array([lt[0] for lt in lifetimes])
showcase_results(result)
