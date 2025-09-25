import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def Plot3D(X, Y, V, filename='lyapunov_function.png', xlabel='x', ylabel='y'):     

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, V, rstride=1, cstride=1, alpha=0.7, cmap=cm.coolwarm)
    # ax.contour(X, Y, V, 10, zdir='z', offset=0, cmap=cm.coolwarm)
    # ax.scatter([0], [0], [0], color='red', s=100, marker='*', label='Equilibrium')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('V(x)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()