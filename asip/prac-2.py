import numpy as np
import matplotlib.pyplot as plt
A=1
T=2
fs=1000
t=np.linspace(0,3*T,3*T*fs)
triangle_signal=A*(2*np.abs(2*(t/T - np.floor(t/T + 0.5))) - 1)
signal1=triangle_signal[:2000]
shift=200
shiftedSignal=np.roll(signal1,shift)
correlation=np.correlate(signal1,shiftedSignal,mode="full")
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.title("Original Triangle Signal")
plt.plot(signal1)
plt.subplot(3,1,2)
plt.title("Shifted Triangle Signal")
plt.plot(shiftedSignal)
plt.subplot(3,1,3)
plt.title("Correlation of Original and Shifted Signal")
plt.plot(correlation)
plt.tight_layout()
plt.show()