"""
    This is my attempt to solve for the time-domain solution for
    PLL-condition self-sustaining oscillation
"""

import os 
import numpy as np     
from ccylib import * 
import matplotlib.pyplot as plt

def main():
    filepath = os.path.join(os.getcwd(), 'Box Sync/python_garage/coupled mode model/') 
    filename = filepath + 'combined_bound_v4.txt'
    
    t_start_temp = os.times()[-1]
    raw_data = np.loadtxt(filename)    # this takes time.... about for 136 MB file, it takes ~ 25 seconds
    #print np.shape(raw_data)

    print 'elasped time for reading txt is {} seconds'.format(os.times()[-1] - t_start_temp)
    mode1 = mega2matrix(raw_data[:,[0,1,-1]])
    mega_plot(mode1)
    prettify_figure('Frequency', 'bound', 'mode 1, steady-state')
    
    mode2 = mega2matrix(raw_data[:,[0,2,-1]])
    mega_plot(mode2)
    prettify_figure('Frequency', 'bound', 'mode 2, steady-state')
    
    para = np.atleast_2d(mode1[0, 1:]).T
    sidebands = np.zeros((np.shape(mode1)[1] -1, 1))
    
    # find sidebands at each parameter setting
    freqs = mode1[1:, 0]
    for i in range(1, np.shape(mode1)[1]):
        locs, pks = find_pks(freqs, mode1[1:, i])
        pks_result = sorted(zip(pks, freqs[locs]))
        if len(pks_result) == 1:
            sidebands[i-1] = 0
        else:
            sidebands[i-1] = abs(pks_result[-1][-1] - pks_result[-2][-1])

    plt.figure()
    plt.scatter(para, sidebands)
    prettify_figure('side-band frequency', 'bound')
    plt.show()
    
    return mode1, mode2
    

    
if __name__ == '__main__':
    plt.close('all')
    t_start = os.times()[-1]
    print '\n start running the main program...\n'
    mode1, mode2 = main()
    print 'total run time is {} seconds'.format(os.times()[-1] - t_start)

        

    
