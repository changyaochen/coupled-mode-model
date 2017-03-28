"""
    This is my function library
    keep expanding!
    Latest date: 9/16/2016
"""

import numpy as np;
import matplotlib.pyplot as plt; 
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib import cm
from scipy.stats import mode;
import math

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
    

def prettify_figure(ylabel = 'y label', xlable = 'x label', 
                    title = 'title', grid = 'off', font_size = 18):
    pass
    plt.ylabel(ylabel, size = 'x-large')
    plt.xlabel(xlable, size = 'x-large')
    plt.title(title, size = 'x-large')
    
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
      item.set_fontsize(font_size)
    
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
    
    plt.show()  
    
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

def find_pks_not_working(x, y):
    """ To find peaks in the data y(x)
    
    In Matlab, there is already existing function from 
    singal processing toolbox for this purpose. 
    Here I just need to reinvent the wheel....
    reference from this post
    https://stackoverflow.com/questions/24656367/find-peaks-location-in-a-spectrum-numpy?answertab=votes#tab-top
    """    
    
    from scipy.signal import convolve
    #Obtaining derivative
    kernel = [1, 0, -1]
    dy = convolve(y, kernel, 'valid') 
    
    #Checking for sign-flipping
    S = np.sign(dy)
    ddS = convolve(S, kernel, 'valid')
    
    #These candidates are basically all negative slope positions
    #Add one since using 'valid' shrinks the arrays
    candidates = np.where(dy < 0)[0] + (len(kernel) - 1)
    
    #Here they are filtered on actually being the final such position in a run of
    #negative slopes
    peaks = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 1))
    
    #If you need a simple filter on peak size you could use:
    alpha = -0.025
    peaks = np.array(peaks)[y[peaks] < alpha]

    plt.scatter(peaks, y[peaks], marker='x', color='g', s=40)

def find_pks(x, y):
    """ To find peaks in the data y(x)
    
    In Matlab, there is already existing function from 
    singal processing toolbox for this purpose. 
    Here I just need to reinvent the wheel....
    
    returns locations (indexs) of the peaks, and corresponding values
    with format list
    """    
    from scipy.signal import argrelmax
    locs = argrelmax(y, order = 5) # comparing 5 points from each side
    
    # change locs from tuple to array
    locs = locs[0]
    
    pks  = y[locs]
    
    
    return locs.tolist(), pks.tolist()

    
    
    
    
