import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Parameters
num_symbols = 10
sps = 8
num_taps = 101
beta = 3.4

# Generate random bits and map to pulses
bits = np.random.randint(0, 2, num_symbols)
x = np.repeat(bits * 2 - 1, sps)  # Repeat each bit value for sps samples

# Plot the unshaped signal
plt.figure(0)
plt.plot(x)
plt.title("Unshaped Signal")
plt.grid(True)
plt.show()

# Create raised-cosine filter
t = np.arange(num_taps) - (num_taps - 1) // 2
h = np.sinc(t / sps) * np.cos(np.pi * beta * t / sps) / (1 - (2 * beta * t / sps)**2)
h = h / np.sum(h)  # Normalize the filter

# Plot the filter
plt.figure(1)
plt.plot(t, h, '.')
plt.title("Raised-Cosine Filter")
plt.grid(True)
plt.show()

# Apply pulse shaping
x_shaped = np.convolve(x, h, mode='full')

# Plot the shaped signal
plt.figure(2)
plt.plot(x_shaped)
plt.title("Shaped Signal")
plt.grid(True)
plt.show()

# Generate and add noise
noise = 0.1 * np.random.normal(0, 2, len(x_shaped)) 
x_noisy = x_shaped + noise

# Plot the noisy signal
plt.figure(3)
plt.plot(x_noisy, label="Noisy Signal")
plt.title("Noise-Affected Signal")
plt.grid(True)
plt.show()

# Step 9: Convolution with matched filter
matched_output = convolve(x_noisy, h, mode='full')

# Step 10: Generate eye diagram
def plot_eye_diagram(signal, sps, num_symbols):
    plt.figure(4)
    for i in range(0, len(signal) - 2 * sps, sps):
        plt.plot(signal[i:i + 2 * sps], 'b')
    plt.title("Eye Diagram")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

plot_eye_diagram(matched_output, sps, num_symbols)

# Step 11: Obtain eye pattern for various roll-off rates
roll_off_rates = [0.2, 0.5, 0.8]
for roll_off in roll_off_rates:
    h_rc = np.sinc(t / sps) * np.cos(np.pi * roll_off * t / sps) / (1 - (2 * roll_off * t / sps)**2)
    h_rc = h_rc / np.sum(h_rc)
    x_filtered = convolve(x, h_rc, mode='full')
    x_noisy_filtered = x_filtered + noise
    matched_output_filtered = convolve(x_noisy_filtered, h_rc, mode='full')
    plot_eye_diagram(matched_output_filtered, sps, num_symbols)
