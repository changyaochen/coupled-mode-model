"""
    This is my attempt to solve for the time-domain solution for
    PLL-condition self-sustaining oscillation
    Specifically for 1:3 condition
    This is trying to respond the reviewer #2's request
    
"""

import os 
import numpy as np 
import scipy as sp 
#import scipy.io as sio 
import matplotlib.pyplot as plt   
from scipy.stats import mode  
import ccylib 
    

def main():
    #dir_prefix = os.getcwd() 
    #os.chdir(dir_prefix + '/Box Sync/python_garage') 
    #t_start_reading = os.times()[-1] 
    plt.close("all") 
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    # ==============================================================
    # ============= parameter initialization =======================
    # ==============================================================
    
    Q1, Q2, bound = 100000, 100000, 5.0e-5 
    beta = 5.0e-3 
    gain = 1000.0 
    J21, J12 = 2.0e-4, 2.0e-4  # reference  J21, J12 = 0.0001, 0.001
    omega2 = 1.1    # here is the true 1:3 model!
    
    save_file = 0 
    save_file_FFT = 0 
    plot_RTSA = 0
  
    # ============ coupled oscillator case ============ 
    x1Int, v1Int, x2Int, v2Int = 0.1, 0, 0, 0 
    t_start_ss = os.times()[-1] 
    
    # temporarily override the paramter values
#    Q1, Q2 = 1.1*122680.0, 86505.0
#    beta   = 5e-3
    omega2 = 1.057 * 3    # here is the 1:3 model!
#    bound  = 2e-4    # this is 'optimized' value, given Q1, Q2, and omega2
    J21    = 2.0e-3  # this is 'optimized' value, given Q1, Q2, and omega2, bound
    #J21    = 1.0e-3  # this is 'test' value
    #J12    = 1.0e-3  # this is 'optimized' value, given Q1, Q2, and omega2, bound, J21  
    J12    = 2.0e-3    # this is 'test' value
    para = 0
    
    
    reference = int(max(Q1, Q2)) 
    time = np.linspace(0, 6*reference, 60*reference+1) 
    #time = np.linspace(0, 7000, 1000+1)   # for debugging purpose
    
    Assol = sp.integrate.odeint(ccylib.solvr_nonlinear_osc_2modes, [x1Int, v1Int, x2Int, v2Int], time,
                                args = (Q1, Q2, omega2, beta, gain, bound, J21, J12))  #   solve the equations!
    AsFinal = Assol[-1] 
    
    plt.figure(1) 
    plt.subplot(211) 
    plt.plot(time[::100], Assol[::100,0], 'b-') 
    ccylib.prettify_figure('displacement', ' ', 'steady-state, mode 1, resampled')
    plt.subplot(212)  
    plt.plot(time[::100], Assol[::100,2], 'g-') 
    ccylib.prettify_figure('displacement', 'time', 'steady-state, mode 2, resampled')
    plt.show() 
    
    
    freq1, fft_out1 = ccylib.myFFT(time[-int(0.1*len(time)):], Assol[-int(0.1*len(time)):,0])  
    # only last 10% of the time-domain data, which assuming to have reached steady-state
    
    freq2, fft_out2 = ccylib.myFFT(time[-int(0.1*len(time)):], Assol[-int(0.1*len(time)):,2])  
    # only last 10% of the time-domain data, which assuming to have reached steady-state
        
        
    plt.figure(11) 
    plt.plot(2*np.pi*freq1, fft_out1.real,'bo-') 
    #plt.plot(2*np.pi*freq1, fft_out1.real,'bo-', 2*np.pi*freq2, fft_out2.real,'go-') 
    plt.xlim([1.0, 1.0 + 2*(omega2 - 1)]) 
    ccylib.prettify_figure('FFT amplitude (a.u.)', 'Frequency (rad/s)', 'Steady-states', grid = 'on')
    plt.yscale('log') 
    plt.show() 
    
    
    print;  print('The FFT resolution (mode1) for steady-state is {} rad/s'.format(2*np.pi*mode(np.diff(freq1))[0][0])) 
    print;  print('The maximum in FFT (mode1) for steady-state happens at {} rad/s'.format(2*np.pi*freq1[fft_out1.argmax()]))  
            
    if save_file == 1:
        np.savetxt('solve_single_case_to_fit_data_ss_1_to_3.dat',np.hstack([time[:,np.newaxis], Assol])) 
    if save_file_FFT == 1:
        idx_low, idx_high = np.argmax(2*np.pi*freq1 > 0.5), np.argmin(2*np.pi*freq1 < 1.7) 
        freq1, fft_out1, fft_out2 = freq1[idx_low:idx_high], fft_out1[idx_low:idx_high], fft_out2[idx_low:idx_high]
        to_save = np.vstack([2*np.pi*freq1, np.log(fft_out1.real), np.log(fft_out2.real), para*np.ones(freq1.shape[0])]) 
        to_save = np.transpose(to_save) 
        with open('solve_single_case_to_fit_data_FFT_1_to_3.txt','a') as FFT_file:
            np.savetxt(FFT_file, to_save,fmt='%10.6f',) 
        print;  print('Saved data!') 
    
    print;  print('elasped time for calculating single steady-state is {} seconds'.format(os.times()[-1] - t_start_ss))  
    
    
    if plot_RTSA == 1:
        t_start_temp = os.times()[-1]
        ccylib.RTSA(time, Assol[:,0], 100)
        ccylib.prettify_figure('Frequency', 'time', 'real-time spectrum, steady-state, mode 1')
        plt.ylim([0.8, 2])
        #plt.savefig('RTSA_steady_state.png', dpi = 400)
        
        ccylib.RTSA(time, Assol[:,2], 100)
        ccylib.prettify_figure('Frequency', 'time', 'real-time spectrum, steady-state, mode 2')
        plt.ylim([0.8, 2])
    
        print;  print('elasped time for RTSA for steady-state is {} seconds'.format(os.times()[-1] - t_start_temp))   
   
    # ===================================================================
    # ================= Below I will start the ringdown =================
    # ===================================================================
    
    #time = np.linspace(0,5001,10*5001)
    time = np.linspace(0, 1*reference, 50*reference+1)
    t_start_ringdown = os.times()[-1]
    Ringdown = sp.integrate.odeint(ccylib.solvr_nonlinear_osc_2modes, AsFinal, time,
                                args = (Q1, Q2, omega2, beta, gain, 0.0, J21, J12))  #   solve the ringdown where bound is set to zero
    
    plt.figure(3) 
    plt.subplot(211) 
    plt.plot(time[::100], Ringdown[::100,0], 'b-', time[::100], max(Ringdown[:100,0])*np.exp(-time[::100]/2/Q1),'r--') 
    ccylib.prettify_figure('displacement', ' ', 'ringdown, mode 1, resampled') 
    #plt.yscale('log') 
    plt.subplot(212)  
    plt.plot(time[::100], Ringdown[::100,2], 'g-',time[::100], max(Ringdown[:100,2])*np.exp(-time[::100]/2/Q2),'r--') 
    ccylib.prettify_figure('displacement', 'time', 'ringdown, mode 2, resampled') 
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
        np.savetxt('solve_single_case_to_fit_data_rd_1_to_3.dat',np.hstack([time[:,np.newaxis], Ringdown])) 
    print;  print('elasped time for ringdown is {} seconds'.format(os.times()[-1] - t_start_ringdown))  
    
    if plot_RTSA == 1:
        t_start_temp = os.times()[-1]
        mode1 = ccylib.RTSA(time, Ringdown[:,0], 100)
        ccylib.prettify_figure('Frequency', 'time', 'real-time spectrum, ringdown, mode 1')
        plt.ylim([0.8, 2])
        #plt.savefig('RTSA_ringdown.png', dpi = 400)
    
        mode2 = ccylib.RTSA(time, Ringdown[:,2], 200)
        ccylib.prettify_figure('Frequency', 'time', 'real-time spectrum, ringdown, mode 2')
        plt.ylim([0.98, 1.10])    
        
        print;  print('elasped time for RTSA for ringdown is {} seconds'.format(os.times()[-1] - t_start_temp)) 
        print;  print('The FFT resolution (mode1) for ringdown is {} Hz'.format(mode(np.diff(mode1[1:, 0]))[0][0])) 
    
    return Ringdown
    
    
if __name__ == '__main__':
    plt.close('all')
    print('\nStaring running the main program...\n')
    rd = main() 
    
