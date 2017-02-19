#!/bin/env python
#PBS -l nodes=1:ppn=8
#PBS -l walltime=3:00:00
#PBS -N megasweep
#PBS -A staff
#PBS -j oe
#PBS -m ea
#PBS -l opsys=el6
# processed by submit filter

"""change in v5: parameters are chosen to mimic data
    I will also just save time-domain data as images, and examine later to find the best conditions
    no FFT and RTSA
"""
# below is for Python 2
import os
if os.environ.has_key('PBS_O_WORKDIR'):
    os.chdir(os.environ['PBS_O_WORKDIR'])
#    
## below is for Python 3
#if 'PBS_O_WORKDIR' in os.environ:
#  os.chdir(os.environ['PBS_O_WORKDIR'])

"""
    This is my attempt to solve for the time-domain solution for
    PLL-condition self-sustaining oscillation
"""

import os
import numpy as np
import scipy as sp
import scipy.integrate
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import mode;
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt;
#from ccylib import myFFT, RTSA
     
def myFFT(time, y):
    """ myFFT(time, y), perform real FFT on data y(time)
    
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

def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    # -0.0001 added for numerical stability as suggested in:
    # http://stackoverflow.com/questions/23840756
    return np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,a,b],
                    [0,0,-np.finfo(float).eps,zback]])
             
def RTSA(time, data, num=100): # num is number of slice, default to 100
    """ RTSA(time, data, num=100), returns matrix M
    
    This is the same as the same Matlab RTSA() function that I wrote
    It will require myFFT() function
    RTSA stands for real time spectrum (fake) analyzer
    """
    
    num_each_slice = len(time)//num    # data points in each segment
    for i in range(0, int(num)):
        freq, fft_out = myFFT(
                    time[num_each_slice * i:num_each_slice * i + num_each_slice -1], 
                    data[num_each_slice * i:num_each_slice * i + num_each_slice -1]) 
        if i == 0:    # initialize output matrix
            M = np.zeros((len(freq)+1, num+1))
            M[1:, 0] = 2 * np.pi * freq
        #M[0, i+1]  = np.mean(time[num_each_slice * i:num_each_slice * i + num_each_slice -1])
        M[0, i+1]  = time[num_each_slice* i]  # take the first time instance as index
        M[1:, i+1] = np.log10(fft_out)
    
    mega_plot(M)
    
    ## ===== below is old code, now I will use mega_plot() ======
    #fig = plt.figure()
    ##ax = Axes3D(fig)
    ##ax.view_init(elev = -10000.0, azim = 0.0)
    #X = M[0, 1:]
    #Y = M[1:, 0]
    #X, Y = np.meshgrid(X, Y)
    #Z = M[1:, 1:]
    #plt.imshow(Z, cmap=plt.cm.jet, aspect = 'equal', interpolation = 
    #           'nearest', )
    ##surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
    ##                   linewidth=0, antialiased=False)
    ##fig.colorbar(surf, shrink=0.5, aspect=5)
    ##
    ###for angle in range(0, 360):
    ###    ax.view_init(30, angle)
    ###    plt.draw()
    ##
    ##ax.view_init(90.0-0.0001, -90.0-0.0001)
    ##ax.persp_transformation = orthogonal_proj
    #plt.show()
    ## ============================================================
    pass
    
    return M

 
def sigmoid(x):
    return 1/(1 + math.exp(-x)) 
    

def prettify_figure(ylabel = 'y label', xlable = 'x label', title = 'title', grid = 'off'):
    pass
    plt.ylabel(ylabel, size = 'x-large')
    plt.xlabel(xlable, size = 'x-large')
    plt.title(title, size = 'x-large')
    
    fig = plt.gcf()
    fig.set_facecolor('white')   
    fig.set_size_inches(12.0, 9.0, forward=True)
    
    if grid == 'on':
        plt.grid(b=True, which='both', color='0.65',linestyle='--')

def mega_plot(M, ylabel_text = 'ylabel', xlabel_text = 'xlabel', smoothing = 'none'): 
    """ takes a matrix as input (including row and column header), and plot!
    
    This is the same as the same Matlab megaPlot() function 
    """
    X = M[0, 1:]
    Y = M[1:, 0]
    Z = M[1:, 1:]
    plt.figure()
    ax = plt.gca()
    plt.imshow(Z, cmap = plt.cm.jet, extent = [X.min(), X.max(), Y.min(), Y.max()], 
             aspect = 'auto', origin = 'lower', interpolation = smoothing)  # make the color plot
    
    # attach the axis labels
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    
    # check whether the axis needs to be plotte in log scale
    # the default option is linear scale
    if len(np.unique(np.diff(X))) > 10:    # the threshold should be 1, but I use 10 to consider possible missing points
        plt.xscale('log')
    if len(np.unique(np.diff(Y))) > 10:    # the threshold should be 1, but I use 10 to consider possible missing points
        plt.yscale('log')
    
    #plt.show()  
    
def mega2matrix(raw_data):
    """ takes in n x 3 np.array, and covert it to correct p x q matrix
    
    This is similar to Matlab mega2matrix function that I wrote before.
    This function is much less fault-tolrant since it assume prefect
    input data, assuming equal spaced, no-missing data. Suitable for
    processing simulation data.
    """
    
    steps = np.unique(raw_data[:,-1])
    num_step = len(steps)
    data_point = max(np.shape(raw_data))//num_step
    M = np.zeros((data_point+1, num_step+1))
    M[1:, 0] = raw_data[0:data_point, 0]
    for i in range(0, num_step):
        M[0,  i+1] = steps[i]
        M[1:, i+1] = raw_data[i*(data_point):(i+1)*(data_point), 1] 
        
    return M

    
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


# ==============================================================================
# ==============================================================================
# ============================== Main ==========================================
# ==============================================================================
# ==============================================================================
def process_input(para):
    #dir_prefix = os.getcwd()
    #os.chdir(dir_prefix + '/Box Sync/python_garage')
    t_start_reading = os.times()[-1]
    plt.close("all")
    print('='*50)
    plt.ioff()
    
    # ==============================================================
    # ============= parameter initialization =======================
    # ==============================================================
    
    Q1, Q2, bound, beta, J21, J12 = 100000.0, 100000.0, 2e-4, 5e-3, 0.0001, 0.001   # default values
    gain, omega2 = 1e3, 1.1    # default values
    Q1, Q2= 122680.0, 86505.0
    omega2 = 1.0539 * 3  # here is to the 1:3 model
    bound  = 1e-4    # this is 'optimized' value, given Q1, Q2, and omega2
    J21    = 7.6e-4  # this is 'optimized' value, given Q1, Q2, and omega2, bound
    
    save_file = 0
    save_file_FFT = 0
    doRTSA = 0
    doPlot = 1
    #para_list = np.logspace(-4,-2,50)
    #para_list = [0.0001]

    reference = int(max(Q1, Q2))
    time = np.linspace(0, 5*reference, 50*reference+1)
    #time = np.linspace(0, 1000, 10000+1);  # for debugging purpose

    
    # ============ coupled oscillator case ============ 
    

#    print('='*50)
    print('Starting with parameter = {}'.format(para)); 
    # ==============================================================
    # ============= parameter setting! =============================
    # ==============================================================
    J12 = para
    J21 = para
    temp_folder = 'PythonParallelTemp_1_to_3_Js'
    if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
    temp_file = os.path.join(os.getcwd(),temp_folder) + '/ss_Js_' + str(para) + '.png' 
    # ==============================================================
    # ============================================================== 
    x1Int, v1Int, x2Int, v2Int = 0.1, 0, 0, 0
    t_start_ss = os.times()[-1]
    Assol = sp.integrate.odeint(solvr_nonlinear_osc_2modes, [x1Int, v1Int, x2Int, v2Int], time,
                                args = (Q1, Q2, omega2, beta, gain, bound, J21, J12)); #   solve the equations!
    AsFinal = Assol[-1]
    
    if doPlot:
      plt.figure(1)
      plt.subplot(211)
      plt.plot(time[::100], Assol[::100,0], 'b-')
      plt.xlabel('time'); plt.ylabel('displacement'); plt.title('mode 1')
      plt.subplot(212); 
      plt.plot(time[::100], Assol[::100,2], 'g-')
      plt.xlabel('time'); plt.ylabel('displacement'); plt.title('mode 2')
      #plt.show()
      plt.savefig(temp_file)
      plt.close(1)
    
    
    freq1, fft_out1 = myFFT(time[-int(0.1*len(time)):], Assol[-int(0.4*len(time)):,0]); 
    # only last 10% of the time-domain data, which assuming to have reached steady-state
    
    freq2, fft_out2 = myFFT(time[-int(0.1*len(time)):], Assol[-int(0.4*len(time)):,2]); 
    # only last 10% of the time-domain data, which assuming to have reached steady-state
        
    if doPlot:    
      plt.figure(11)
      #plt.plot(2*np.pi*freq1, abs(fft_out1),'b-')
      plt.plot(2*np.pi*freq1, fft_out1.real,'bo-', 2*np.pi*freq2, fft_out2.real,'go-')
      plt.xlim([1.0, 1.1])
      plt.grid(b=True, which='both', color='0.65',linestyle='--')
      plt.title('Steady-state')
      plt.xlabel('Frequency (Hz)'); 
      plt.ylabel('FFT amplitude (a.u.)');    plt.yscale('log')
      #plt.show()
      plt.savefig(os.path.join(os.getcwd(),temp_folder) + '/ss_FFT_Js_' + str(para) + '.png')
      
      #print; print 'The FFT resolution for steady-state is {} Hz'.format(2*np.pi*mode(np.diff(freq2))[0][0])
      #print; print 'The maximum in FFT for steady-state happens at {} Hz'.format(2*np.pi*freq2[fft_out2.argmax()]); 
            
    if save_file == 1:
        np.savetxt(os.path.join(os.getcwd(),temp_folder) + '/ss_Js_' + str(para) + '.txt',
                   np.hstack([time[:,np.newaxis], Assol]))
    #if save_file_FFT == 1:
    #    if not os.path.exists(temp_folder):
    #        os.makedirs(temp_folder)
    #    idx_low, idx_high = np.argmax(2*np.pi*freq1 > 0.5), np.argmin(2*np.pi*freq1 < 1.7)
    #    freq1, fft_out1, fft_out2 = freq1[idx_low:idx_high], fft_out1[idx_low:idx_high], fft_out2[idx_low:idx_high]
    #    to_save = np.vstack([2*np.pi*freq1, np.log(fft_out1.real), np.log(fft_out2.real), para*np.ones(freq1.shape[0])])
    #    to_save = np.transpose(to_save)
    #    with open(os.path.join(os.getcwd(),temp_folder) + '/' + str(para) + '.txt','w') as FFT_file:
    #        np.savetxt(FFT_file, to_save, fmt='%10.6f')
    #    print; print 'Saved data with parameter = {}'.format(para)
    #    #return to_save
    
    print; print('elasped time for calculating single steady-state is {} seconds'.format(os.times()[-1] - t_start_ss)); 
    

    
    
    # ===================================================================
    # ================= Below I will start the ringdown =================
    # ===================================================================
    t_start_ss = os.times()[-1]
    #time = np.linspace(0,5001,10*5001)
    Ringdown = sp.integrate.odeint(solvr_nonlinear_osc_2modes, AsFinal, time,
                                args = (Q1, Q2, omega2, beta, gain, 0.0, J21, J12)); #   solve the ringdown where bound is set to zero
    
    if doPlot:
      plt.figure(3)
      plt.subplot(211)
      plt.plot(time[::100], Ringdown[::100,0], 'b-', time[::100], max(Ringdown[:100,0])*np.exp(-time[::100]/2/Q1),'r--')
      plt.xlabel('time'); plt.ylabel('displacement'); plt.title('Ringdown, mode 1')
      #plt.yscale('log')
      plt.subplot(212); 
      plt.plot(time[::100], Ringdown[::100,2], 'g-',time[::100], max(Ringdown[:100,2])*np.exp(-time[::100]/2/Q2),'r--')
      plt.xlabel('time'); plt.ylabel('displacement'); plt.title('Ringdown, mode 2')
      #plt.yscale('log')
      #plt.show()
      
      temp_file_2 = os.path.join(os.getcwd(),temp_folder) + '/rd_Js_' + str(para) + '.png' 
      plt.savefig(temp_file_2)
      plt.close(3)

    
    
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
        
    #plt.figure(2)
    #plt.plot(time, Assol[:,2])
    #plt.xlabel('time'); plt.ylabel('displacement'); plt.title('mode 2'); 
    #plt.show()
    
    if save_file == 1:
        np.savetxt(os.path.join(os.getcwd(),temp_folder) + '/rd_Js_' + str(para) + '.txt'
                   ,np.hstack([time[:,np.newaxis], Ringdown]))
    #print; print 'elasped time for total is {} seconds'.format(os.times()[-1] - t_start_reading); 
    
    if doRTSA:
      RTSA(time, Ringdown[:,0], 100)
      prettify_figure('Frequency', 'time', 'real-time spectrum, ringdown, mode 1')
      plt.ylim([1.0, 1.1])
      plt.yscale('linear')
      plt.savefig(os.path.join(os.getcwd(),temp_folder) + '/RTSA_mode1_Js_' + str(para) + '.png', dpi = 400)
      
      RTSA(time, Ringdown[:,2], 200)
      prettify_figure('Frequency', 'time', 'real-time spectrum, ringdown, mode 2')
      plt.ylim([1.0, 1.1])  
      plt.yscale('linear')  
      plt.savefig(os.path.join(os.getcwd(),temp_folder) + '/RTSA_mode2_Js_' + str(para) + '.png', dpi = 400)
    print('elasped time for calculating single ringdown is {} seconds'.format(os.times()[-1] - t_start_ss));

def main():
    
    para_list = np.logspace(-5, -3, 11);   
#    num_cores = multiprocessing.cpu_count()	# BUG: finds all in /proc, should count cpuset instead [stern]
    num_cores = int(os.environ['PBS_NUM_PPN'])	# will work for nodes=1:....
    
    print('Number of cores: {}'.format(num_cores))
    Parallel(n_jobs=num_cores)(delayed(process_input)(i) for i in para_list)

    #raw_input("Press Enter to continue...")
    #with open('combined_J21_v5.txt', 'a') as outfile:
    #    for para in para_list:
    #        with open(os.path.join(os.getcwd(),'PythonParallelTemp_J21_v5') + '/' +str(para)+'.txt', 'r') as infile:
    #            outfile.write(infile.read())
    
#    # below is for no parallelization
#    for para in para_list:
#        process_input(para)

if __name__ == '__main__':
    #print "cpu_count:\t", multiprocessing.cpu_count()
    #if os.environ.has_key('PBS_NUM_PPN'):
	   # print "num_ppn:\t", os.environ['PBS_NUM_PPN']
       
#    # below is for debugging on local machine
#    dname = os.path.dirname(os.path.abspath(__file__))
#    os.chdir(dname)
#    # end
    main()
