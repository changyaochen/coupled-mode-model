
import numpy as np;
import matplotlib.pyplot as plt; 


             
def mega_plot(M, ylabel_text = 'ylabel', xlabel_text = 'xlabel'): 
    """ takes a matrix as input (including row and column header), and plot!
    
    This is the same as the same Matlab megaPlot() function 
    """
    X = M[0, 1:]
    Y = M[1:, 0]
    Z = M[1:, 1:]
    plt.figure()
    ax = plt.gca()
    plt.imshow(Z, cmap = plt.cm.jet, aspect = 'auto', origin = 'lower')  # make the color plot
    
    # next I will replace the tick labels
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(1, len(xlabels)-1):
        xlabels[i] = unicode(round(min(X) + (i-1)*(max(X) - min(X))/(len(xlabels)-3), 1))
    #print xlabels
    ax.set_xticklabels(xlabels)
    
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    for i in range(1, len(ylabels)-1):
        ylabels[i] = unicode(round(min(Y) + (i-1)*(max(Y) - min(Y))/(len(ylabels)-3), 1))
    ax.set_yticklabels(ylabels)
    
    # attach the axis labels
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    
    plt.show()
    
    
    
    
if __name__ == '__main__':
    plt.close('all')
    x = np.arange(-2.5, 2.6, 0.1)
    y = np.arange(-2.5, 2.6, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    M = np.vstack((x, Z))
    temp = np.atleast_2d(np.insert(y, 0, np.nan)).T
    M = np.hstack((temp, M))    
    mega_plot(M);