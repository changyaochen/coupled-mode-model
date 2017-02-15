"""
    This is to test fft function
"""

import os;
import numpy as np;
from scipy.stats import mode;
import matplotlib.pyplot as plt; 

__all__ = ['myFFT']
             
def myFFT(time, y):
    """ 
    This is to take two 1D arrays, one is time, in second, the other is 
    data-series data, and then to perform FFT.
    The returns are FFT freq, and FFT amplitude.
    The unit for FFT freq is rad
    The unit for FFT amplitude is a.u.
    As of Aug. 2016, I haven't figured out how to convert the 
    FFT amplitude unit to PSD yet
    """
    #plt.close("all");
    #omega = 1; 
    #time = np.linspace(0.0, 10.0, 10000+1);
    sample_time_interval = mode(np.diff(time))[0][0];
    #y = np.sin(2*np.pi*omega*time)
    
    #plt.figure(1);
    #plt.plot(time, y);
    #plt.show();
    
    spectrum = np.fft.rfft(y);
    spectrum_freq = np.fft.rfftfreq(y.size, sample_time_interval);
    #plt.figure(2);
    #plt.plot(spectrum_freq, spectrum);
    #plt.show();
    #plt.xscale('log')
    
    return spectrum_freq, abs(spectrum);
    
    
    
    
if __name__ == '__main__':
    main();