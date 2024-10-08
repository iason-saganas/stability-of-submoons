import numpy as np
import matplotlib.pyplot as plt

extreme = False

if not extreme:
    # Example 1: Diverging y_array with decreasing step-size in t_array
    t_arr_1 = np.concatenate([np.linspace(0, 10, 100), np.linspace(10, 11, 50), np.linspace(11, 11.1, 50)])
    y_arr_1 = np.concatenate(
        [np.exp(np.linspace(0, 5, 100)), np.exp(np.linspace(5, 6, 50)), np.exp(np.linspace(6, 7, 50))])

    # Example 2: Diverging y_array with decreasing step-size in t_array
    t_arr_2 = np.concatenate([np.linspace(0, 10, 100), np.linspace(10, 11, 50), np.linspace(11, 11.05, 50)])
    y_arr_2 = np.concatenate(
        [np.exp(np.linspace(0, 4, 100)), np.exp(np.linspace(4, 5, 50)), np.exp(np.linspace(5, 6, 50))])

else:

    # Example 1: Extremely diverging y_array with rapidly decreasing step-size in t_array
    t_arr_1 = np.concatenate([np.linspace(0, 10, 100), np.linspace(10, 10.1, 50), np.linspace(10.1, 10.11, 50)])
    y_arr_1 = np.concatenate([np.exp(np.linspace(0, 10, 100)), np.exp(np.linspace(10, 20, 50)), np.exp(np.linspace(20, 30, 50))])

    # Example 2: Extremely diverging y_array with rapidly decreasing step-size in t_array
    t_arr_2 = np.concatenate([np.linspace(0, 10, 100), np.linspace(10, 10.05, 50), np.linspace(10.05, 10.051, 50)])
    y_arr_2 = np.concatenate([np.exp(np.linspace(0, 8, 100)), np.exp(np.linspace(8, 16, 50)), np.exp(np.linspace(16, 24, 50))])

# Example 3: Non-diverging y_array
t_arr_3 = np.linspace(0, 10, 200)
y_arr_3 = np.sin(t_arr_3)

# Example 4: Funky-looking y_array
t_arr_4 = np.linspace(0, 10, 200)
y_arr_4 = np.sin(t_arr_4) + np.sin(5 * t_arr_4)

# Function to detect divergence
def neighbouring_diffs(arr):
    return np.diff(arr) / arr[:-1]  # cut off first element

def detect_divergence(t_arr, y_arr):
    relative_progress = neighbouring_diffs(t_arr)  # in the t-direction
    try:
        tr_idx = np.where(relative_progress < 1e-3)[0][0]
        print("Possibly divergent. Relative progress: ", relative_progress)
    except IndexError:
        # No transition point was detected.
        return False, None
    grad = np.gradient(y_arr, t_arr)
    lambda_ratio = grad[tr_idx - 1] / grad[-1]
    print("Found lambda ratio: ", lambda_ratio)
    if lambda_ratio < 1e-6:
        return True, tr_idx
    else:
        return False, tr_idx


# Test the function on the examples
examples = [(t_arr_1, y_arr_1), (t_arr_2, y_arr_2), (t_arr_3, y_arr_3), (t_arr_4, y_arr_4)]
results = [detect_divergence(t, y) for t, y in examples]

# Print the results
for i, (result,_) in enumerate(results):
    print(f"Example {i}: {'Diverged' if result else 'Did not diverge'}")

# Plot the examples
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
titles = ["Example 1 (Diverging)", "Example 2 (Diverging)", "Example 3 (Non-diverging)", "Example 4 (Funky-looking)"]

for ax, (t, y), (result, tr_idx), title in zip(axs.ravel(), examples, results, titles):
    ax.plot(t, y, 'b.', lw=0)
    if tr_idx is not None:
        ax.axvline(t[tr_idx], color='red', linestyle='--', label=f'Transition at t={t[tr_idx]:.2e}')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Plot the gradients

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
titles = ["Example 1 (Diverging)", "Example 2 (Diverging)", "Example 3 (Non-diverging)", "Example 4 (Funky-looking)"]

for ax, (t, y), (result, tr_idx), title in zip(axs.ravel(), examples, results, titles):
    grad = np.gradient(y,t)
    ax.plot(t, grad, 'r.', lw=0)
    if tr_idx is not None:
        print("Grad tr index - 1 , grad[-1] and divied: ", grad[tr_idx-1], grad[-1], grad[tr_idx-1]/grad[-1])
        ax.axvline(t[tr_idx], color='red', linestyle='--', label=f'Transition at t={t[tr_idx]:.2e}')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

"""
Conclusion: 

In extreme cases, it does what it is supposed to do. 
In less milder cases, it does not detect divergences. Meaning that in the worst-case scenario in our workflow: 

>> An analytic solution has diverged, but a numeric has not. I.e.: Either significant deviation OR divergence
detection algorithm not good enough.  

"""
