import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_symbols = 50 
sps = 8  # Samples per symbol
beta = 0.6 # Roll-off factor (zero for ideal Nyquist filter)
num_taps = 101  # Filter length

# Generate random bits and map to pulses (BPSK modulation)
bits = 2 * np.random.randint(0, 2, num_symbols) - 1  # Generate random bits (-1, 1)

# Map bits to a pulse train with sps samples per symbol
x = np.repeat(bits, sps)

# Raised-cosine filter construction
t = np.arange(num_taps) - (num_taps - 1) // 2  # Time indices centered around zero
h = np.sinc(t / sps) * np.cos(np.pi * beta * t / sps) / (1 - (2 * beta * t / sps) ** 2)
h[np.isinf(h)] = 0  # Replace infinities at t=0

# Apply pulse shaping (filtering the signal with the raised-cosine filter)
x_shaped = np.convolve(x, h, mode='full')

# Plotting the original bits, raised-cosine filter, and shaped signal
plt.figure(figsize=(10, 6))

# Plot the original bits
plt.subplot(3, 1, 1)
plt.plot(bits, drawstyle="steps-mid")
plt.title("Original Bits")
plt.grid(True)

# Plot the raised-cosine filter (impulse response)
plt.subplot(3, 1, 2)
plt.plot(h, '.-')
plt.title("Raised-Cosine Filter Impulse Response")
plt.grid(True)

# Plot the shaped signal after applying the filter
plt.subplot(3, 1, 3)
plt.plot(x_shaped, '.-')
plt.title("Shaped Signal")
plt.grid(True)

plt.tight_layout()
plt.show()

# Function to plot the eye diagram
def plot_eye_diagram(signal, sps, num_symbols):
    plt.figure(5)
    for i in range(50, len(signal) - 2 * sps, sps):
        plt.plot(signal[i:i + 2 * sps], 'b')
    plt.title("Eye Diagram")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

# Plot the eye diagram for the matched filtered signal
plot_eye_diagram(x_shaped, sps, num_symbols)
