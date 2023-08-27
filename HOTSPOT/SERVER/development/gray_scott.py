# Adapted from:
# https://github.com/benmaier/reaction-diffusion

# import necessary libraries
import numpy as np
import matplotlib.pyplot as pl

# for the animation
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# ============ define relevant functions =============

def interp(a, b, x):
    return a + (b - a) * x

def xy_to_kf(x, y):
    ''' 
    Implemented map function from Karl Sims RD Tool 
    https://karlsims.com/rdtool.html
    '''
    y1 = y * 0.5 + 0.5
    f = interp(0.002, 0.12, y1)
    s = np.sqrt(f) * 0.5 - f
    x1 = x * interp(1, (y - 0.32) * (y - 0.32), 0.6) * 0.5 + 0.5
    k0 = interp(-0.003, 0.0115, x1)
    k1 = interp(-0.0048, -0.0031, x1)
    k = s + interp(k0, k1, y1)
    return k, f

# an efficient function to compute a mean over neighboring cells
def apply_laplacian(mat):
    """This function applies a discretized Laplacian
    in periodic boundary conditions to a matrix
    For more information see 
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    """

    # the cell appears 4 times in the formula to compute
    # the total difference
    neigh_mat = -4*mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [ 
                    ( 1.0,  (-1, 0) ),
                    ( 1.0,  ( 0,-1) ),
                    ( 1.0,  ( 0, 1) ),
                    ( 1.0,  ( 1, 0) ),
                    # ( 0.05,  (-1, 1) ),
                    # ( 0.05,  (-1, -1) ),
                    # ( 0.05,  (1, 1) ),
                    # ( 0.05,  (1, -1) ),
                ]

    # shift matrix according to demanded neighbors                    # (-1.0,  ( 0, 0) ),

    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (1,0))

    return neigh_mat


def update(A, B, DA, DB, f, k, delta_t):
    """Apply the Gray-Scott update formula"""

    # compute the diffusion part of the update
    diff_A = DA * apply_laplacian(A)
    diff_B = DB * apply_laplacian(B)
    
    # Apply chemical reaction
    reaction = A*B**2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1-A)
    diff_B -= (k+f) * B

    A += diff_A * delta_t
    B += diff_B * delta_t

    return A, B

def get_initial_A_and_B(N, random_influence = 0.1):
    """get the initial chemical concentrations"""

    # get initial homogeneous concentrations
    A = (1-random_influence) * np.ones((N,N))
    B = np.zeros((N,N))

    # put some noise on there
    A += random_influence * np.random.random((N,N))
    B += random_influence * np.random.random((N,N))

    # get center and radius for initial disturbance
    N2, r = N//2, 50

    # apply initial disturbance
    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25

    return A, B

def get_initial_artists(A, B):
    """return the matplotlib artists for animation"""
    fig, ax = pl.subplots(1,2,figsize=(5.65,3))
    imA = ax[0].imshow(A, animated=True,vmin=0,cmap='Greys')
    imB = ax[1].imshow(B, animated=True,vmax=1,cmap='Greys')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('A')
    ax[1].set_title('B')

    return fig, imA, imB

def updatefig(frame_id,updates_per_frame,*args):
    """Takes care of the matplotlib-artist update in the animation"""

    # update x times before updating the frame
    for u in range(updates_per_frame):
        A, B = update(*args)

    # update the frame
    imA.set_array(A)
    imB.set_array(B)

    # # renormalize the colors
    imA.set_norm(Normalize(vmin=np.amin(A),vmax=np.amax(A)))
    imB.set_norm(Normalize(vmin=np.amin(B),vmax=np.amax(B)))

    # return the updated matplotlib objects
    return imA, imB

# =========== define model parameters ==========

# update in time
delta_t = 1.0

# Diffusion coefficients
DA = 0.16
DB = 0.08

# define birth/death rates
f = 0.01215
k = 0.0412

# grid size
N = 200

# intialize the figures
A, B = get_initial_A_and_B(N)
fig, imA, imB = get_initial_artists(A, B)

# initialise the map
kM = np.zeros_like(A)
fM = np.zeros_like(B)

for i, y in enumerate(np.linspace(-1, 1, A.shape[0])):
    for j, x in enumerate(np.linspace(-1, 1, A.shape[0])):
        k_, f_ = xy_to_kf(x, y)
        kM[i, j] = k_
        fM[i, j] = f_

# how many updates should be computed before a new frame is drawn
updates_per_frame = 10

# these are the arguments which have to passed to the update function
animation_arguments = (updates_per_frame, A, B, DA, DB, fM, kM, delta_t)

# start the animation
ani = animation.FuncAnimation(fig, #matplotlib figure
                              updatefig, # function that takes care of the update
                              fargs=animation_arguments, # arguments to pass to this function
                              interval=1, # update every `interval` milliseconds
                              blit=True, # optimize the drawing update 
                              )

# show the animation
pl.show()
