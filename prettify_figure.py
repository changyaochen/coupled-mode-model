"""
    This is to make cosmetic change to a single figure
"""
import matplotlib.pyplot as plt;

def prettify_figure(ylabel = 'y label', xlable = 'x label', title = 'title'):
    pass
    plt.ylabel(ylabel, size = 'x-large')
    plt.xlabel(xlable, size = 'x-large')
    
    fig = plt.gcf()
    fig.set_facecolor('white')   
    fig.set_size_inches(12.0, 9.0, forward=True)


    
    
