# model for changing temperature
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn
from numba import njit # makes code miles faster


"""
2-dimensional Ising model
"""

k = 1  # Boltzmann constant - using natural units
N = 100
P = N * N  # true number of dipoles
epsilon = 1  # energy contribution factor??
spin_probability = 0.5  # when initialising if random.random >= spin_probability, set spin to 1, else -1
total_steps = 1000 * P  # 1000 * N^2 + 1 total steps

# visualise the data in a heatmap
def visualise(state, temp, time):
    seaborn.heatmap(state, cmap="Greys", cbar=False, xticklabels=False,
                    yticklabels=False, square=True)
    plt.title("state after " + str(time) + " timesteps and temp: " + str(temp) + ", with " + str(N) + \
              "$^{2}$ dipoles")
    plt.show()


# attempt to speed up program, would remove need for copy and running solve_U twice
# solve next U value given a state and a flipped dipole at (row, col)
@njit
def next_state_U(state, row, col, lastU, N): # solves next states U value correctly,
    del_U = 0
    right = (row + 1) % N
    below = (col + 1) % N

    # subtract influence of unflipped dipole
    del_U -= state[row][col] * state[right][col]
    del_U -= state[(row - 1) % N][col] * state[row][col]
    del_U -= state[row][col] * state[row][below]
    del_U -= state[row][(col - 1) % N] * state[row][col]
    # add change from flipping dipole
    del_U += state[row][col] * -1 * state[right][col]
    del_U += state[(row - 1) % N][col] * state[row][col] * -1
    del_U += state[row][below] * state[row][col] * -1
    del_U += state[row][(col - 1) % N] * state[row][col] * -1

    del_U *= -epsilon
    del_U += lastU

    return del_U



# solve U for initial state
@njit
def solveU(state, N):
    U = 0
    for i in range(N):  # same as 1d, 1->i+1 and j to j+1, then multiply by 2 if necessary
        for j in range(N):
            below = (i + 1) % N  # to account for periodic boundaries
            right = (j + 1) % N
            # add right and below neighbour, should cover all cells, do i need account for both ways
            # ie. multiply by 2, (adding above and left neightbours). realistically,
            # only changes U by 2, so del_U will be twice as big too. ask tutor
            U += state[i][j]*(state[below][j] + state[i][right])
    return U * -epsilon  # multiply by negative eps and return


# metropolis algorithm
# inputs:
    # state - some microstate
    # beta - defined as 1/kT, used to determine a probability

# returns:
    #   - next state -- may have one flipped dipole compared to state
@njit
def metropolis(state, beta, last_U, N):
    # choose random dipole at (rrow, rcol
    rrow, rcol = np.random.randint(0, N, (1, 2))[0]
    dipole = state[rrow][rcol]
    next_U = next_state_U(state, rrow, rcol, last_U, N) # works

    del_U = next_U - last_U

    if del_U <= 0:
        state[rrow][rcol] *= -1  # flip dipole
        return state, next_U, 1, dipole # number indicates dipole has been flipped
    else:
        flip = np.random.random()
        if flip <= np.exp(-beta*del_U):
            state[rrow][rcol] *= -1  # flip dipole
            return state, next_U, 1, dipole
        else:
            return state, last_U, 0, dipole

@njit
def count_spins(state):
    up = 0
    for i in range(N):
        for j in range(N):
            if state[i][j] > 0:
                up += 1
    return up


@njit
def model(): # specifically for changing temp during run of model
    # variables
    temp_change_durations = [1, 3, 3]  # in terms of total steps
    temps_eq = np.ones(temp_change_durations[0]*total_steps)
    temps_inc = np.linspace(1,3, temp_change_durations[1]*total_steps)
    temps_dec = np.linspace(3,1,temp_change_durations[2]*total_steps)
    temps = np.concatenate((temps_eq, temps_inc, temps_dec))
    print(temps.shape)
    Us = np.zeros_like(temps)  # list of Us
    ms = np.zeros_like(temps)
    time_steps = np.linspace(0, np.shape(temps)[0], np.shape(temps)[0])
    # initialise spins
    spins = np.random.random((N, N))
    states = np.zeros((3, N, N))
    for i in range(N):
        for j in range(N):
            if spins[i][j] >= spin_probability:
                spins[i][j] = 1
            else:
                spins[i][j] = -1
    #visualise(spins, temp, "Initial") # visualise intial state
    Us[0] = solveU(spins, N) # define U at time = 0
    ms[0] = count_spins(spins)
    for i in range(len(temps)):
        temp = temps[i]
        beta = 1 / (k * temp)
        if i == temp_change_durations[0]*total_steps:
            for i in range(N):
                for j in range(N):
                    states[0][i][j] = spins[i][j]
        if i == (temp_change_durations[0]+temp_change_durations[1])*total_steps:
            for i in range(N):
                for j in range(N):
                    states[1][i][j] = spins[i][j]
        if i == len(temps) - 1:
            for i in range(N):
                for j in range(N):
                    states[2][i][j] = spins[i][j]

        spins, next_U, flipped, dipole = metropolis(spins, beta, Us[i], N)
        Us[i + 1] = next_U # update Us
        if dipole > 0 and flipped == 0:
            ms[i+1] = ms[i]
        if dipole > 0 and flipped == 1:
            ms[i + 1] = ms[i] - 1
        if dipole < 0 and flipped == 1:
            ms[i + 1] = ms[i] + 1


    return states, time_steps, ms

# make suret
temp_change_durations_ = [1, 3, 3] # in terms of total steps

start = time.time()
states, steps, ms = model() # run model
print(time.time() - start)

plt.plot(steps, ms)
plt.show()
plt.clf()

eq = states[0]
inc = states[1]
dec = states[2]

visualise(eq, 1, temp_change_durations_[0]*total_steps)
visualise(inc, 3, (temp_change_durations_[0]+temp_change_durations_[1])*total_steps)
visualise(dec, 1, (temp_change_durations_[0]+temp_change_durations_[1]+
                   temp_change_durations_[2])*total_steps)



