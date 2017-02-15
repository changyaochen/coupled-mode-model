"""
    This is the same as the same Matlab RTSA() function that I wrote
    It will require myFFT() function
"""

import numpy as np;
import matplotlib.pyplot as plt; 
from myFFT import myFFT;
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mega_plot import *

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
    pass;
    
    num_each_slice = len(time)//num    # data points in each segment
    for i in range(0, int(num)):
        freq, fft_out = myFFT(
                    time[num_each_slice * i:num_each_slice * i + num_each_slice -1], 
                    data[num_each_slice * i:num_each_slice * i + num_each_slice -1]) 
        if i == 0:    # initialize output matrix
            M = np.zeros((len(freq)+1, num+1))
            M[1:, 0] = 2 * np.pi * freq
        M[0, i+1]  = np.mean(time[num_each_slice * i:num_each_slice * i + num_each_slice -1])
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
    
    
    
if __name__ == '__main__':
    #plt.close('all')
    time = np.linspace(0,100,1001)
    data = np.sin(2*np.pi*1*time)
    RTSA(time, data, 5);