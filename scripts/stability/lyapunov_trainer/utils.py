import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def Plot3D(X, Y, V, filename='lyapunov_function.png', xlabel='$\Theta$', ylabel='$\dot{\Theta}$'):     

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, V, rstride=1, cstride=1, alpha=0.7, cmap=cm.coolwarm)
    # ax.contour(X, Y, V, 10, zdir='z', offset=0, cmap=cm.coolwarm)
    # ax.scatter([0], [0], [0], color='red', s=100, marker='*', label='Equilibrium')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('V(x)')
    # plt.title('Lyapunov Function')
    plt.savefig(filename)
    plt.close()

def Plotflow(Xd, Yd, t, f):
    # Plot phase plane 
    DX, DY = f([Xd, Yd],t)
    DX=DX/np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
    DY=DY/np.linalg.norm(DY, ord=2, axis=1, keepdims=True)
    plt.streamplot(Xd,Yd,DX,DY, color=('gray'), linewidth=0.5,
                  density=0.5, arrowstyle='-|>', arrowsize=1.5)

def plot_2D_roa_lie_overlay(X, Y, V_nn_est, f=None, r=None, filename='2d_roa.png',
                            positive_examples=None, negative_examples=None, borderline_examples=None):
    '''
    Plot Region of attraction for systems with 2 state variables
    '''
    fig = plt.figure(figsize=(8,6))

    ax = plt.gca()
    legend_list = []
    label_list = []
    # Plot positive and negative lie derivatives samples
    inside_contour = V_nn_est.reshape(10000) < 0
    # Plot values less than 0 in green
    x1, x2 = np.meshgrid(X, Y)
    x = np.vstack([x1.flatten(), x2.flatten()]).transpose(1, 0)
    theta = x[:, 0][inside_contour]
    theta_dot = x[:, 1][inside_contour]
    positive_examples = positive_examples[inside_contour]
    negative_examples = negative_examples[inside_contour]
    borderline_examples = borderline_examples[inside_contour]

    ax.scatter(theta[positive_examples], theta_dot[positive_examples], color='green', label=r'$L_V < 0$', s=5)
    legend_list.append(plt.Rectangle((0,0),1,2,color='green',fill=False,linewidth = 2))
    label_list.append(r'$L_V < 0$')

    # Plot values greater or equal to 0 in red
    ax.scatter(theta[negative_examples], theta_dot[negative_examples], color='red', label=r'$L_V > \epsilon$', s=5)
    legend_list.append(plt.Rectangle((0,0),1,2,color='red',fill=False,linewidth = 2))
    label_list.append(r'$L_V > \epsilon$')

    # Plot values greater or equal to 0 in red
    ax.scatter(theta[borderline_examples], theta_dot[borderline_examples], color='yellow', label=r'$0 \leq L_V \leq \epsilon$', s=5)
    legend_list.append(plt.Rectangle((0,0),1,2,color='yellow',fill=False,linewidth = 2))
    label_list.append(r'$0 \leq L_V \leq \epsilon$')
    # Vaild Region
    C = plt.Circle((0, 0), r, color='grey', linewidth=1.5, fill=False, linestyle='--')
    ax.add_artist(C)

    # plot direction field
    xd = np.linspace(-r, r, 10) 
    yd = np.linspace(-r, r, 10)
    Xd, Yd = np.meshgrid(xd,yd)
    t = np.linspace(0, 2, 100)
    Plotflow(Xd, Yd, t, f) 

    # plot contour of estimated lyapunov
    ax.contour(X, Y, V_nn_est, levels=0, linewidths=2, colors='k')
    legend_list.append(plt.Rectangle((0,0),1,2,color='k',fill=False,linewidth = 2))
    label_list.append('NN, True Loss')

    legend_list.append(C)
    label_list.append('Valid Region')

    plt.title('Region of Attraction')
    plt.legend(legend_list, label_list, loc='upper right')
    plt.xlabel(r'Angle, $\theta$ (rad)')
    plt.ylabel(r'Angular velocity $\dot{\theta}$')
    plt.savefig(filename)
    plt.close()

def plot_2D_roa(X, Y, V_nn_est, f=None, r=None, filename='2d_roa.png'):
    '''
    Plot Region of attraction for systems with 2 state variables
    '''
    fig = plt.figure(figsize=(8,6))

    ax = plt.gca()
    # Vaild Region
    C = plt.Circle((0, 0), r, color='grey', linewidth=1.5, fill=False, linestyle='--')
    ax.add_artist(C)

    legend_list = []
    label_list = []

    ax.contour(X, Y, V_nn_est, levels=0, linewidths=2, colors='green')
    legend_list.append(plt.Rectangle((0,0),1,2,color='green',fill=False,linewidth = 2))
    label_list.append('NN, Appx Lie Derivative Loss')

    legend_list.append(C)
    label_list.append('Valid Region')

    plt.title('Region of Attraction')
    plt.legend(legend_list, label_list, loc='upper right')
    plt.xlabel(r'Angle, $\theta$ (rad)')
    plt.ylabel(r'Angular velocity $\dot{\theta}$')
    plt.savefig(filename)
    plt.close()