import numpy as np
import matplotlib.pyplot as plt
# Parameters
num_symbols = 10
sps = 8
num_taps = 101
beta = 0.4
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
h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2)
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
noise =0.1* np.random.normal(0, 2, len(x_shaped)) 
x_noisy = x_shaped + noise
# Plot the noisy signal
plt.figure(3)
plt.plot(x_noisy, label="Noisy Signal")
plt.title("Noise-Affected Signal")
plt.grid(True)
plt.show()
# Apply the same filter to the noisy signal (optional for matched filtering)
x_filtered = np.convolve(x_noisy, h, mode='full')
# Sample at decision points
decision_points = np.arange(num_symbols) * sps + num_taps // 2
samples = x_noisy[decision_points.astype(int)]
# Decision device: Map samples to original bits
output_bits = (samples > 0).astype(int)
# Plot the output of the decision device
plt.figure(4)
plt.plot(output_bits,drawstyle='steps-mid')
plt.title("Output of Decision Device")
plt.xlabel("Symbol Index")
plt.ylabel("Decoded Bits")
plt.grid(True)
plt.show()
print("Original Bits: ", bits)
print("Decoded Bits:  ", output_bits)


