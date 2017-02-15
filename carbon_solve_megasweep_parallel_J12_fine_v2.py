#!/bin/env python
#PBS -l nodes=1:ppn=8
#PBS -l walltime=20:00:00
#PBS -N megasweep_J12_fine_v2
#PBS -A staff
#PBS -j oe
#PBS -m ea
#PBS -l opsys=el6
# processed by submit filter

import os
if os.environ.has_key('PBS_O_WORKDIR'):
    os.chdir(os.environ['PBS_O_WORKDIR'])

"""
    This is my attempt to solve for the time-domain solution for
    PLL-condition self-sustaining oscillation
"""

import os
import numpy as np
import scipy as sp
import math
#import scipy.io as sio
import matplotlib as mp
import matplotlib.pyplot as plt; 
#from myFFT import *
from scipy.stats import mode;  
from joblib import Parallel, delayed
import multiprocessing
import shutil as st
     

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
    #plt.close("all")
    #omega = 1; 
    #time = np.linspace(0.0, 10.0, 10000+1)
    sample_time_interval = mode(np.diff(time))[0][0]
    #y = np.sin(2*np.pi*omega*time)
    
    #plt.figure(1)
    #plt.plot(time, y)
    #plt.show()
    
    spectrum = np.fft.rfft(y)
    spectrum_freq = np.fft.rfftfreq(y.size, sample_time_interval)
    #plt.figure(2)
    #plt.plot(spectrum_freq, spectrum)
    #plt.show()
    #plt.xscale('log')
    
    return spectrum_freq, abs(spectrum)
    
def solvr_linear(Y, t): 
    '''
    Let me start from linear resonator case such that
    x'' + (1+x^2)x'/Q + x == f0*cos(omega*t)
    omega will set to be resonance, i.e. 1# define the ODE for linear oscillator
    '''
    Q = 100.0
    omega = 1.0
    f0 = 1.0/Q
    return [
            Y[1], 
            -(Y[0] + Y[1]/Q) + f0*np.cos(omega*t),               
            ]; # Y[0] is displacement, Y[1] is velocity
            
def solvr_linear_osc(Y, t): 
    '''
    Now let me move on to the self-sustaining oscillator case
    x'' + (1+x^2)x'/Q + x == clip(gain*x)
    omega will set to be resonance, i.e. 1# define the ODE for linear oscillator
    the clipping function np.clip(input, lower_bound, upper_bound)
    will prevent the divergence
    '''
    Q = 100.0
    gain = 10.0
    bound = 0.01
    return [
            Y[1], 
            -(Y[0] + Y[1]/Q) + np.clip(gain*Y[1], -1.0*bound, 1.0*bound),               
            ]
def solvr_nonlinear_osc(Y, t, bound = 0.01):  # default value for bound is 0.01
    '''
    Now let me move on to the self-sustaining oscillator, nonlinear case
    x'' + (1+x^2)x'/Q + x + beta * x**3== clip(gain*x)
    omega will set to be resonance, i.e. 1# define the ODE for linear oscillator
    the clipping function np.clip(input, lower_bound, upper_bound)
    will prevent the divergence
    '''
    Q = 100.0
    gain = 10.0
    beta = 1
    return [
            Y[1], 
            -(Y[0] + Y[1]/Q + beta*Y[0]**3) + np.clip(gain*Y[1], -1.0*bound, 1.0*bound),               
            ]
            
def solvr_nonlinear_osc_no_clip(Y, t): 
    '''
    Now let me move on to the self-sustaining oscillator, nonlinear case
    x'' + (1+x^2)x'/Q + x + beta * x**3== clip(gain*x)
    omega will set to be resonance, i.e. 1# define the ODE for linear oscillator
    the clipping function np.clip(input, lower_bound, upper_bound)
    will prevent the divergence
    '''
    Q = 100.0
    gain = 200.0
    beta = 1
    return [
            Y[1], 
            -(Y[0] + Y[1]/Q + beta*Y[0]**3) + gain*Y[0],               
            ]

def solvr_nonlinear_osc_2modes(Y, t, Q1=100.0, Q2=100.0, omega2=1.2, beta=1.0, gain=10.0, bound=0.01, J21=1.0, J12=1.0): 
    '''
    
    x'' + (1+x^2)x'/Q + x + beta * x**3== clip(gain*x)
    omega will set to be resonance, i.e. 1# define the ODE for linear oscillator
    the clipping function np.clip(input, lower_bound, upper_bound)
    will prevent the divergence
    '''
    return [
            Y[1], 
            -(Y[0] + Y[1]/Q1 + beta*Y[0]**3) + np.clip(gain*Y[1], -1.0*bound, 1.0*bound) + J21*Y[2],   
            Y[3],
            -((omega2**2) * Y[2] + Y[3]/Q2) + J12*Y[0]            
            ]; 

def process_input(para):
    #dir_prefix = os.getcwd()
    #os.chdir(dir_prefix + '/Box Sync/python_garage')
    t_start_reading = os.times()[-1]
    plt.close("all")
    print('\n\n\n\n\n')
    
    # ==============================================================
    # ============= parameter initialization =======================
    # ==============================================================
    
    Q1, Q2, bound = 100000, 100000, 0.0002
    beta = 0.005
    gain = 1000.0
    J21, J12 = 0.0001, 0.001; # reference  J21, J12 = 0.0001, 0.001
    reference = int(max(Q1, Q2))
    time = np.linspace(0, 20*reference, 500*reference+1)
    #time = np.linspace(0, 7000, 1000+1);  # for debugging purpose
    save_file = 0
    save_file_FFT = 1
    #para_list = np.logspace(-4,-2,50)
    #para_list = [0.0001]

    # ============ resonator case ============ 
    #xInt, vInt = 0.1, 0
    #Assol = sp.integrate.odeint(solvr_linear, [xInt, vInt], time); #   solve the equations!
    
    # ============ oscillator case ============ 
    #xInt, vInt = 0.1, 0
    #Assol = sp.integrate.odeint(solvr_nonlinear_osc_2modes, [xInt, vInt], time); #   solve the equations!
    
    # ============ coupled oscillator case ============ 
    

    print('\n\n\n\n\n')
    print 'Starting with parameter = {}'.format(para); 
    # ==============================================================
    # ============= parameter setting! =============================
    # ==============================================================
    J12 = para
    temp_folder = 'PythonParallelTemp_J12_fine_v2'
    # check whether the file already exist
    if os.path.isfile(os.path.join(os.getcwd(),temp_folder) + '/' + str(para) + '.txt'):
        return
    # ==============================================================
    # ============================================================== 
    x1Int, v1Int, x2Int, v2Int = 0.1, 0, 0, 0
    t_start_ss = os.times()[-1]
    Assol = sp.integrate.odeint(solvr_nonlinear_osc_2modes, [x1Int, v1Int, x2Int, v2Int], time,
                                args = (Q1, Q2, 1.1, beta, gain, bound, J21, J12)); #   solve the equations!
    AsFinal = Assol[-1]
    
    #plt.figure(1)
    #plt.subplot(211)
    #plt.plot(time, Assol[:,0], 'b-')
    #plt.xlabel('time'); plt.ylabel('displacement'); plt.title('mode 1')
    #plt.subplot(212); 
    #plt.plot(time, Assol[:,2], 'g-')
    #plt.xlabel('time'); plt.ylabel('displacement'); plt.title('mode 2')
    #plt.show()
    
    
    freq1, fft_out1 = myFFT(time[-int(0.4*len(time)):], Assol[-int(0.4*len(time)):,0]); 
    # only last 20% of the time-domain data, which assuming to have reached steady-state
    
    freq2, fft_out2 = myFFT(time[-int(0.4*len(time)):], Assol[-int(0.4*len(time)):,2]); 
    # only last 20% of the time-domain data, which assuming to have reached steady-state
        
        
    #plt.figure(11)
    ##plt.plot(2*np.pi*freq1, abs(fft_out1),'b-')
    #plt.plot(2*np.pi*freq1, fft_out1.real,'bo-', 2*np.pi*freq2, fft_out2.real,'go-')
    #plt.xlim([0.8, 2])
    #plt.grid(b=True, which='both', color='0.65',linestyle='--')
    #plt.title('Steady-state')
    #plt.xlabel('Frequency (Hz)'); 
    #plt.ylabel('FFT amplitude (a.u.)');    plt.yscale('log')
    #plt.show()
    
    print; print 'The FFT resolution for steady-state is {} Hz'.format(2*np.pi*mode(np.diff(freq2))[0][0])
    print; print 'The maximum in FFT for steady-state happens at {} Hz'.format(2*np.pi*freq2[fft_out2.argmax()]); 
            
    if save_file == 1:
        np.savetxt('test.dat',np.hstack([time[:,np.newaxis], Assol]))
    if save_file_FFT == 1:
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        idx_low, idx_high = np.argmax(2*np.pi*freq1 > 0.5), np.argmin(2*np.pi*freq1 < 1.7)
        freq1, fft_out1, fft_out2 = freq1[idx_low:idx_high], fft_out1[idx_low:idx_high], fft_out2[idx_low:idx_high]
        to_save = np.vstack([2*np.pi*freq1, np.log(fft_out1.real), np.log(fft_out2.real), para*np.ones(freq1.shape[0])])
        to_save = np.transpose(to_save)
        with open(os.path.join(os.getcwd(),temp_folder) + '/' + str(para) + '.txt','w') as FFT_file:
            np.savetxt(FFT_file, to_save, fmt='%10.6f')
        print; print 'Saved data with parameter = {}'.format(para)
        #return to_save
    
    print; print 'elasped time for calculating single steady-state is {} seconds'.format(os.times()[-1] - t_start_ss)
    

    
    
    # ===================================================================
    # ================= Below I will start the ringdown =================
    # ===================================================================
    
    #time = np.linspace(0,5001,10*5001)
    #Ringdown = sp.integrate.odeint(solvr_nonlinear_osc_2modes, AsFinal, time,
    #                            args = (Q1, Q2, 1.1, beta, gain, 0.0, J21, J12)); #   solve the ringdown where bound is set to zero
    #
    #plt.figure(3)
    #plt.subplot(211)
    #plt.plot(time, Ringdown[:,0], 'b-', time, max(Ringdown[:100,0])*np.exp(-time/2/Q1),'r--')
    #plt.xlabel('time'); plt.ylabel('displacement'); plt.title('Ringdown, mode 1')
    ##plt.yscale('log')
    #plt.subplot(212); 
    #plt.plot(time, Ringdown[:,2], 'g-',time, max(Ringdown[:100,2])*np.exp(-time/2/Q2),'r--')
    #plt.xlabel('time'); plt.ylabel('displacement'); plt.title('Ringdown, mode 2')
    ##plt.yscale('log')
    #plt.show()
    #
    #idx_temp = np.argmax(time>5000)
    ##idx_temp = time.shape[0]
    #freq1, fft_out1 = myFFT(time[:idx_temp], Ringdown[:idx_temp,0]); 
    ## only the first 5000 seconds
    #
    #freq2, fft_out2 = myFFT(time[:idx_temp], Ringdown[:idx_temp,2])
    ## only the first 5000 seconds
    #
    #plt.figure(31)
    ##plt.plot(2*np.pi*freq1, abs(fft_out1),'b-')
    #plt.plot(2*np.pi*freq1, fft_out1.real,'bo-', 2*np.pi*freq2, fft_out2.real,'go-')
    #plt.xlim([0.8, 2])
    #plt.grid(b=True, which='both', color='0.65',linestyle='--')
    #plt.title('Ringdown...')
    #plt.xlabel('Frequency (Hz)'); 
    #plt.ylabel('FFT amplitude (a.u.)');plt.yscale('log')
    #
    #plt.show()
    #print; print 'The FFT resolution for ringdown is {} Hz'.format(2*np.pi*mode(np.diff(freq2))[0][0])
    #
    #pass
    #    
    ##plt.figure(2)
    ##plt.plot(time, Assol[:,2])
    ##plt.xlabel('time'); plt.ylabel('displacement'); plt.title('mode 2'); 
    ##plt.show()
    #
    #if save_file == 1:
    #    np.savetxt('test_ringdown.dat',np.hstack([time[:,np.newaxis], Ringdown]))
    #print; print 'elasped time for total is {} seconds'.format(os.times()[-1] - t_start_reading); 

def main():
    para_list = np.logspace(-3.5, -2.5, 200);   

    #num_cores = multiprocessing.cpu_count()	# BUG: finds all in /proc, should count cpuset instead [stern]
    num_cores = int(os.environ['PBS_NUM_PPN'])	# will work for nodes=1:....

    Parallel(n_jobs=num_cores)(delayed(process_input)(i) for i in para_list)

    #raw_input("Press Enter to continue...")
    with open('combined_J12_fine_v2.txt', 'a') as outfile:
        for para in para_list:
            with open(os.path.join(os.getcwd(),'PythonParallelTemp_J12_fine_v2') + '/' +str(para)+'.txt', 'r') as infile:
                outfile.write(infile.read())
    #st.rmtree(os.path.join(os.getcwd(),'PythonParallelTemp_J12_fine'))

if __name__ == '__main__':
    print "cpu_count:\t", multiprocessing.cpu_count()
    if os.environ.has_key('PBS_NUM_PPN'):
	    print "num_ppn:\t", os.environ['PBS_NUM_PPN']

    main()
