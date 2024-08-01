# 2d Ising model
# for Q2 of project
# based heavily off of 1d model

import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import time


"""
2d Ising model
"""


k = 1  # Boltzmann constant -- currently trying to use "natural units"
N = 30  # number of dipoles in the system
epsilon = 1  # energy contribution factor??
initial_T = 2.5  # temperature
spin_probability = 0.5  # when initialising if random.random >= spin_probability, set spin to 1, else -1
total_steps = 1000 * N * N + 1 # 1000 * N^2 + 1 total steps
timesteps = np.linspace(0, total_steps, total_steps) # for plotting later
Us = np.linspace(0, total_steps, total_steps) # list of Us


# visualise the data in a heatmap
def visualise(state, temp, string):
    seaborn.heatmap(state, cmap="Greys", cbar=False, xticklabels=False,
                    yticklabels=False, square=True)
    plt.title(string + " spin state at temperature: " + str(temp))
    plt.show()


# attempt to speed up program, would remove need for copy and running solve_U twice
def next_state_U(state, row, col): # solves next states U value correctly,
    # quicker then deep copying the old state, and solving U then subtracting difference
    U = 0
    for i in range(N):
        for j in range(N):
            # check if randomly selected dipole is right ((j+1)%N), below ((i+1)%N) or this cell(i, j)
            if i == row and j == col:
                U += state[i][j] * (state[(i + 1) % N][j] + state[i][(j + 1) % N]) * -1
                continue
            if (i + 1) % N == row and j == col:
                U += state[i][j] * (state[(i + 1) % N][j] * -1 + state[i][(j + 1) % N])
                continue
            if (j + 1) % N == col and i == row:
                U += state[i][j] * (state[(i + 1) % N][j] + state[i][(j + 1) % N] * -1)
                continue
            U += state[i][j] * (state[(i + 1) % N][j] + state[i][(j + 1) % N])
    return U * -epsilon  # multiply by negative eps and return


# metropolis algorithm
# inputs:
    # state - some microstate
    # beta - defined as 1/kT, used to determine a probability

# returns:
    #   - next state -- may have one flipped dipole compared to state
def metropolis(state, beta, last_U):
    # choose random dipole
    rrow, rcol = np.random.randint(0, N, (1, 2))[0]
    next_U = next_state_U(state, rrow, rcol) # works
    del_U = next_U - last_U

    if del_U <= 0:
        state[rrow][rcol] *= -1  # flip dipole
        return state, 1, next_U # number indicates dipole has been flipped
    else:
        flip = np.random.random()
        if flip <= np.exp(-beta*del_U):
            state[rrow][rcol] *= -1  # flip dipole
            return state, 1, next_U
        else:
            return state, 0, last_U


def model(): # one dimensional Ising Model
    # variables
    temp = initial_T
    beta = 1/(k*temp)
    # initialise spins
    spins = np.random.random((N, N))
    for i in range(N):
        for j in range(N):
            if spins[i][j] >= spin_probability:
                spins[i][j] = 1
            else:
                spins[i][j] = -1
    #visualise(spins, temp, "Initial") # visualise intial state
    Us[0] = solveU(spins) # define U at time = 0
    start = time.time()
    flips = np.zeros(total_steps)
    total_flips = 0
    for i in range(total_steps):
        if i + 1 == total_steps:
            flips[i] = total_flips # to make array sizes line up for clean plots
            continue
        spins, flipped, next_U = metropolis(spins, beta, Us[i])
        Us[i + 1] = next_U
        total_flips += flipped
        flips[i] = total_flips
        if i % 10000 == 0:
            print(i)
            end = time.time()
            print(end - start)
            start = end
    visualise(spins, temp, "Final") # visualise final state
    plt.plot(timesteps, flips)
    plt.show()
    plt.plot(timesteps, Us)
    plt.show()

    # plot u, f, S, c per time step???


model()
