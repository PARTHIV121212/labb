# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:04:07 2025

@author: Krishnapriyan R
"""

import numpy as np
import matplotlib.pyplot as plt

# Message Signal Generation
fm = 100  # Message frequency
dc_offset = 2  # DC offset
tm = np.arange(0, 5/fm, 0.0001)  # Time vector for message signal
xm = dc_offset + np.sin(2 * np.pi * fm * tm)

# Sampling
fs = 30 * fm  # Sampling rate
ts = np.arange(0, 5/fm, 1/fs)  # Time vector for sampled signal
xs = dc_offset + np.sin(2 * np.pi * fm * ts)

# Plotting Message Signal
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(tm, xm)
plt.title("Message Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# Plotting Sampled Signal
plt.subplot(2, 1, 2)
plt.stem(ts, xs)
plt.title("Sampled Message Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()

# Quantization
L = 8  # Number of quantization levels
q_levels = np.linspace(min(xm), max(xm), L)  # Quantization levels
xq = np.digitize(xs, q_levels)  # Quantized values


# Plotting Quantized Signal
plt.figure(figsize=(8, 6))
plt.plot(ts, q_levels[xq - 1], 'co-')
plt.title("Quantized Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Calculating SQNR
q_noise = xs - q_levels[xq - 1]  # Quantization noise
signal_power = ((max(xm) - min(xm))**2) / 2  # Signal power
qnoise_power = np.mean(q_noise**2)  # Noise power
sqnr = 10 * np.log10(signal_power / qnoise_power)  # SQNR in dB
print(f"SQNR: {sqnr} dB")

# SQNR vs. Bits per Symbol
bits_per_symbol = range(1, 9)
sqnr_values = []
sqnr_theoretical_values = []
for b in bits_per_symbol:
    q_levels = np.linspace(min(xm), max(xm), 2**b)
    xq = np.digitize(xs, q_levels)
    q_noise = xs - q_levels[xq - 1]
    qnoise_power = np.mean(q_noise**2)
    sqnr = 10 * np.log10(signal_power / qnoise_power)
    sqnr_values.append(sqnr)
    sqnr_eqndB = 6.02 * b + 1.78
    sqnr_theoretical_values.append(sqnr_eqndB)
    
plt.figure(figsize=(8,6))    
plt.plot(ts,q_noise)

# Plotting SQNR vs Bits per Symbol
plt.figure(figsize=(8, 6))
plt.plot(bits_per_symbol, sqnr_values, 'go-',label="Calculated SQNR")
plt.plot(bits_per_symbol,sqnr_theoretical_values,label="Theoretical SQNR")
plt.title("SQNR vs Bits per Symbol")
plt.xlabel("Bits per Symbol")
plt.ylabel("SQNR (dB)")
plt.legend()
plt.grid(True)
plt.show()
