import time
import numpy as np
from scipy.optimize import curve_fit

def exponential_decay(x, a, b):
    return a * np.exp(b * x)

# Create a NumPy array containing the values
values = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])

start_time = time.time()
iteration_numbers = np.arange(len(values))
params, covariance = curve_fit(exponential_decay, iteration_numbers, values)
end_time = time.time()

a, b = params

convergence_speed = 1 / abs(b)

print(f"Convergence speed: {convergence_speed:.4f}")
print(f"Elapsed time: {end_time - start_time:.4f}")
