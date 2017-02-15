"""
    This is my attempt to solve for the time-domain solution for
    PLL-condition self-sustaining oscillation
"""

import os 
import numpy as np 
import scipy as sp 
#import scipy.io as sio 
import matplotlib.pyplot as plt   
from scipy.stats import mode  
from ccylib import * 
    
    
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
            ]  # Y[0] is displacement, Y[1] is velocity
            
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
            ]  

def main():
    #dir_prefix = os.getcwd() 
    #os.chdir(dir_prefix + '/Box Sync/python_garage') 
    #t_start_reading = os.times()[-1] 
    plt.close("all") 
    #print('\n\n\n\n\n') 
    
    # ==============================================================
    # ============= parameter initialization =======================
    # ==============================================================
    
    Q1, Q2, bound = 100000, 100000, 0.0002 
    beta = 0.005 
    gain = 1000.0 
    J21, J12 = 0.0001, 0.001  # reference  J21, J12 = 0.0001, 0.001
    
    save_file = 0 
    save_file_FFT = 0 

    # ============ resonator case ============ 
    #xInt, vInt = 0.1, 0 
    #Assol = sp.integrate.odeint(solvr_linear, [xInt, vInt], time)  #   solve the equations!
    
    # ============ oscillator case ============ 
    #xInt, vInt = 0.1, 0 
    #Assol = sp.integrate.odeint(solvr_nonlinear_osc_2modes, [xInt, vInt], time)  #   solve the equations!
    
    # ============ coupled oscillator case ============ 
    x1Int, v1Int, x2Int, v2Int = 0.1, 0, 0, 0 
    t_start_ss = os.times()[-1] 
    
    # temporarily override the paramter values
    J21, J12 = 10.0*J21, 10.0*J12
    Q2 = 1000000
    
    
    reference = int(max(Q1, Q2)) 
    time = np.linspace(0, 1*reference, 10*reference+1) 
    #time = np.linspace(0, 7000, 1000+1)   # for debugging purpose
    
    Assol = sp.integrate.odeint(solvr_nonlinear_osc_2modes, [x1Int, v1Int, x2Int, v2Int], time,
                                args = (Q1, Q2, 1.1, beta, gain, bound, J21, J12))  #   solve the equations!
    AsFinal = Assol[-1] 
    
    plt.figure(1) 
    plt.subplot(211) 
    plt.plot(time[::100], Assol[::100,0], 'b-') 
    prettify_figure('displacement', ' ', 'steady-state, mode 1, resampled')
    plt.subplot(212)  
    plt.plot(time[::100], Assol[::100,2], 'g-') 
    prettify_figure('displacement', 'time', 'steady-state, mode 2, resampled')
    plt.show() 
    
    
    freq1, fft_out1 = myFFT(time[-int(0.4*len(time)):], Assol[-int(0.4*len(time)):,0])  
    # only last 40% of the time-domain data, which assuming to have reached steady-state
    
    freq2, fft_out2 = myFFT(time[-int(0.4*len(time)):], Assol[-int(0.4*len(time)):,2])  
    # only last 40% of the time-domain data, which assuming to have reached steady-state
        
        
    plt.figure(11) 
    #plt.plot(2*np.pi*freq1, abs(fft_out1),'b-') 
    plt.plot(2*np.pi*freq1, fft_out1.real,'bo-', 2*np.pi*freq2, fft_out2.real,'go-') 
    plt.xlim([0.8, 2]) 
    prettify_figure('FFT amplitude (a.u.)', 'Frequency (Hz)', 'Steady-states', grid = 'on')
    plt.yscale('log') 
    plt.show() 
    
    print;  print 'The FFT resolution (mode1) for steady-state is {} Hz'.format(2*np.pi*mode(np.diff(freq1))[0][0]) 
    print;  print 'The maximum in FFT (mode1) for steady-state happens at {} Hz'.format(2*np.pi*freq1[fft_out1.argmax()])  
            
    if save_file == 1:
        np.savetxt('test.dat',np.hstack([time[:,np.newaxis], Assol])) 
    if save_file_FFT == 1:
        idx_low, idx_high = np.argmax(2*np.pi*freq1 > 0.5), np.argmin(2*np.pi*freq1 < 1.7) 
        freq1, fft_out1, fft_out2 = freq1[idx_low:idx_high], fft_out1[idx_low:idx_high], fft_out2[idx_low:idx_high]
        to_save = np.vstack([2*np.pi*freq1, np.log(fft_out1.real), np.log(fft_out2.real), para*np.ones(freq1.shape[0])]) 
        to_save = np.transpose(to_save) 
        with open('test_mega_steady_state_!!CHANGE_HERE!!.txt','a') as FFT_file:
            np.savetxt(FFT_file, to_save,fmt='%10.6f',) 
        print;  print 'Saved data!' 
    
    print;  print 'elasped time for calculating single steady-state is {} seconds'.format(os.times()[-1] - t_start_ss)  
    
    t_start_temp = os.times()[-1]
    RTSA(time, Assol[:,0], 100)
    prettify_figure('Frequency', 'time', 'real-time spectrum, steady-state, mode 1')
    plt.ylim([0.8, 2])
    #plt.savefig('RTSA_steady_state.png', dpi = 400)
    
    RTSA(time, Assol[:,2], 100)
    prettify_figure('Frequency', 'time', 'real-time spectrum, steady-state, mode 2')
    plt.ylim([0.8, 2])

    print;  print 'elasped time for RTSA for steady-state is {} seconds'.format(os.times()[-1] - t_start_temp) 
       
   
    # ===================================================================
    # ================= Below I will start the ringdown =================
    # ===================================================================
    
    #time = np.linspace(0,5001,10*5001)
    time = np.linspace(0, 3*reference, 100*reference+1) 
    t_start_ringdown = os.times()[-1]
    Ringdown = sp.integrate.odeint(solvr_nonlinear_osc_2modes, AsFinal, time,
                                args = (Q1, Q2, 1.1, beta, gain, 0.0, J21, J12))  #   solve the ringdown where bound is set to zero
    
    plt.figure(3) 
    plt.subplot(211) 
    plt.plot(time[::100], Ringdown[::100,0], 'b-', time[::100], max(Ringdown[:100,0])*np.exp(-time[::100]/2/Q1),'r--') 
    prettify_figure('displacement', ' ', 'ringdown, mode 1, resampled') 
    #plt.yscale('log') 
    plt.subplot(212)  
    plt.plot(time[::100], Ringdown[::100,2], 'g-',time[::100], max(Ringdown[:100,2])*np.exp(-time[::100]/2/Q2),'r--') 
    prettify_figure('displacement', 'time', 'ringdown, mode 2, resampled') 
    #plt.yscale('log') 
    plt.show() 
    
    #idx_temp = np.argmax(time>5000) 
    ##idx_temp = time.shape[0] 
    #freq1, fft_out1 = myFFT(time[:idx_temp], Ringdown[:idx_temp,0])  
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
    #plt.xlabel('Frequency (Hz)')  
    #plt.ylabel('FFT amplitude (a.u.)'); plt.yscale('log') 
    #
    #plt.show() 
    #print;  print 'The FFT resolution for ringdown is {} Hz'.format(2*np.pi*mode(np.diff(freq2))[0][0]) 

        
    #plt.figure(2) 
    #plt.plot(time, Assol[:,2]) 
    #plt.xlabel('time')  plt.ylabel('displacement')  plt.title('mode 2')  
    #plt.show() 
    
    if save_file == 1:
        np.savetxt('test_ringdown.dat',np.hstack([time[:,np.newaxis], Ringdown])) 
    print;  print 'elasped time for ringdown is {} seconds'.format(os.times()[-1] - t_start_ringdown)  
    
    t_start_temp = os.times()[-1]
    mode1 = RTSA(time, Ringdown[:,0], 100)
    prettify_figure('Frequency', 'time', 'real-time spectrum, ringdown, mode 1')
    plt.ylim([0.8, 2])
    #plt.savefig('RTSA_ringdown.png', dpi = 400)

    mode2 = RTSA(time, Ringdown[:,2], 200)
    prettify_figure('Frequency', 'time', 'real-time spectrum, ringdown, mode 2')
    plt.ylim([0.8, 2])    
    
    print;  print 'elasped time for RTSA for ringdown is {} seconds'.format(os.times()[-1] - t_start_temp) 
    print;  print 'The FFT resolution (mode1) for ringdown is {} Hz'.format(mode(np.diff(mode1[1:, 0]))[0][0]) 
    
    return mode1, mode2, Ringdown
    
    
if __name__ == '__main__':
    plt.close('all')
    print('\nStaring running the main program...\n')
    mode1, mode2, Ringdown = main() 
    
