import numpy as np
import matplotlib.pyplot as plt
N = 10 # Number of symbols to transmit ,to be more than 500 to observe proper BERcurve
L = 16  # oversampling factor,L=Tb/Ts(Tb=bit period,Ts=sampling period)
# if a carrier is used, use L = Fs/Fc, where Fs >> 2xFc
Fc = 40  # carrier frequency
Fs = L*Fc  # sampling frequency
ak = np.random.randint(0,2,N)#generate random binary data
nrz=2*ak-1 #NRZ encoded data
upsampleddata=np.repeat(nrz,L)#upsampled NRZ data
t = np.arange(0, len(ak)*L)
y = np.cos(2*np.pi*Fc*t/Fs)
bpsk=upsampleddata*y
plt.subplot(4,1,1)
plt.title("Carrier Signal")
plt.plot(t,y)
plt.subplot(4,1,2)
plt.title("Data")
plt.plot(t,upsampleddata)
plt.subplot(4,1,3)
plt.title("BPSK wave")
plt.plot(t,bpsk)
plt.subplot(4,1,4)
plt.title("Constellation Diagram")
plt.plot(np.real(upsampleddata), np.imag(upsampleddata), 'mo')
plt.tight_layout()
plt.show()
