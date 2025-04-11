# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:08:38 2025

@author: Krishnapriyan R
"""

import numpy as np
import matplotlib.pyplot as plt


f = 100# Frequency of the sine wave (Hz)
fs = 16  # Sampling rate (samples per second)
duration = 1  # Duration of the signal (seconds)
n_bits = range(1, 9)  # Bit levels (1 to 8 bits)

# Step 1: Generate sine wave
t = np.linspace(0, duration, 100)  # Time vector
x_t = 1 + np.sin(2 * np.pi * f * t)  # Original sine wave

# Initialize SQNR values
sqnr_values = []

# Create plots
plt.figure(figsize=(12, 8))

# Original sine wave plot
plt.subplot(5, 1, 1)
plt.plot(t, x_t, label="Original Sine Wave", linewidth=1.5)
plt.title("Original and Sampled Sine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.subplot(5,1,2)
plt.title("sampled")
plt.stem(t,x_t)


# Process quantization and calculate SQNR
for bits in n_bits:
    levels = 2 ** bits  # Number of quantization levels
    min_val, max_val = np.min(x_t), np.max(x_t)
    delta = (max_val - min_val) / levels  # Step size
    quantized_x = np.round((x_t - min_val) / delta) *delta + min_val  # Quantization
    
    # Calculate SQNR
    signal_power = np.mean(x_t ** 2)
    noise_power = np.mean((x_t - quantized_x) ** 2)
    
    sqnr = 10* np.log(signal_power / noise_power)
    sqnr_values.append(sqnr)
    
    # Plot quantized signal for 3-bit example
    if bits == 3:
        plt.subplot(5, 1, 3)
        plt.step(t, quantized_x, label="Quantized Signal (3 bits)", color="red", where="mid", linewidth=1.5)
        plt.title("Quantized Sine Wave (3 bits)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()

plt.subplot(5,1,4)
plt.plot(x_t-quantized_x)

# SQNR vs. Bits plot
plt.subplot(5, 1, 5)
plt.plot(n_bits, sqnr_values)
plt.title("SQNR vs. Number of Bits")
plt.xlabel("Number of Bits")
plt.ylabel("SQNR (dB)")
plt.grid()
plt.legend()


plt.tight_layout()
plt.show()
