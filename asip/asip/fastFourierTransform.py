import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import wavfile
fs,speech=wavfile.read("C:\\Users\\abhay\\asip\\sampleWav.wav")
signal_segment=speech[:1028]
fft_values=np.fft.fft(signal_segment)
frequencies=np.fft.fftfreq(len(signal_segment),1/fs)
magnitude=np.abs(fft_values)
plt.figure(figsize=(10,10))
plt.plot(frequencies[:1024],magnitude[:1024])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum of the Signal Segment")
plt.show()