
import numpy as np
import matplotlib.pyplot as plt

N = 50 # Number of symbols to transmit ,to be more than 500 to observe proper BERcurve
L = 16  # oversampling factor,L=Tb/Ts(Tb=bit period,Ts=sampling period)
# if a carrier is used, use L = Fs/Fc, where Fs >> 2xFc
Fc =40  # carrier frequency
Fs = L*Fc  # sampling frequency

ak = np.random.randint(0, 2, N)#generate random binary data
odd=ak[0::2]
even=ak[1::2]

nrz1=2*odd-1 #NRZ encoded data
upsampleddata1=np.repeat(nrz1,L)#upsampled NRZ data

nrz2=2*even-1 #NRZ encoded data
upsampleddata2=np.repeat(nrz2,L)#upsampled NRZ data

t = np.arange(len(upsampleddata1)) / Fs
#multiply data with carrier to obtain BPSK wave
BPSK1=upsampleddata1*np.cos(2*np.pi*Fc*t)
BPSK2=upsampleddata2*np.sin(2*np.pi*Fc*t)
#QPSK=Bpsk1+1j*Bpsk2
QPSK=BPSK1-BPSK2

plt.figure(figsize =(10, 8))

# Plot time-domain BPSK signals
plt.subplot(6, 1, 1)
plt.title("BPSK1 Time-Domain Signal (Odd)")
plt.plot(t, BPSK1)

plt.subplot(6, 1, 2)
plt.title("BPSK2 Time-Domain Signal (Even)")
plt.plot(t, BPSK2)

# Plot data
plt.subplot(6, 1, 3)
plt.title("Upsampled Data Odd")
plt.plot(t,upsampleddata1)

plt.subplot(6, 1, 4)
plt.title("Upsampled Data Even")
plt.plot(t,upsampleddata2)

# Plot time-domain QPSK signal
plt.subplot(6, 1, 5)
plt.title("QPSK Time-Domain Signal")
plt.plot(t, QPSK)

# Plot Constellation Diagram
plt.subplot(6, 1, 6)
plt.title("QPSK Constellation Diagram")
plt.scatter(nrz1, nrz2, color='magenta')  # Scatter plot of symbols in I-Q plane
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")

[ax.grid(True) for ax in plt.gcf().get_axes()]
plt.tight_layout()
plt.show()
