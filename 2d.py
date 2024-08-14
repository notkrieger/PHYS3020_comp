# 2d Ising model
# for Q2 of project
# based heavily off of 1d model

import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import time

"""
2-dimensional Ising model
"""

k = 1  # Boltzmann constant -- currently trying to use "natural units"
N = 100  # number of dipoles in the system
epsilon = 1  # energy contribution factor??
initial_T = 1  # temperature
spin_probability = 0.5  # when initialising if random.random >= spin_probability, set spin to 1, else -1
total_steps = 1000 * N * N + 1 # 1000 * N^2 + 1 total steps
timesteps = np.linspace(0, total_steps, total_steps) # for plotting later
Us = np.linspace(0, total_steps, total_steps) # list of Us

print(total_steps)

# visualise the data in a heatmap
def visualise(state, temp, string):
    seaborn.heatmap(state, cmap="Greys", cbar=False, xticklabels=False,
                    yticklabels=False, square=True)
    if string == "Initial":
        plt.title(string + " spin state")
    else:
        plt.title(string + " spin state at temperature: " + str(temp))
    plt.show()


# attempt to speed up program, would remove need for copy and running solve_U twice
# solve next U value given a state and a flipped dipole at (row, col)
def next_state_U(state, row, col, lastU): # solves next states U value correctly,
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
def solveU(state):
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
def metropolis(state, beta, last_U):
    # choose random dipole at (rrow, rcol
    rrow, rcol = np.random.randint(0, N, (1, 2))[0]
    next_U = next_state_U(state, rrow, rcol, last_U) # works

    del_U = next_U - last_U

    if del_U <= 0:
        state[rrow][rcol] *= -1  # flip dipole
        return state, next_U, 1 # number indicates dipole has been flipped
    else:
        flip = np.random.random()
        if flip <= np.exp(-beta*del_U):
            state[rrow][rcol] *= -1  # flip dipole
            return state, next_U, 1
        else:
            return state, last_U, 0


def model(temp): # one dimensional Ising Model
    # variables
    beta = 1/(k*temp)
    # initialise spins
    spins = np.random.random((N, N))
    flips = np.zeros(total_steps)
    total_flips = 0
    for i in range(N):
        for j in range(N):
            if spins[i][j] >= spin_probability:
                spins[i][j] = 1
            else:
                spins[i][j] = -1
    #visualise(spins, temp, "Initial") # visualise intial state
    Us[0] = solveU(spins) # define U at time = 0
    start = time.time()
    totalU = 0
    totalU2 = 0
    for i in range(total_steps):
        if i + 1 == total_steps:
            flips[i] = total_flips  # to make array sizes line up for clean plots
            continue
        spins, next_U, flipped = metropolis(spins, beta, Us[i])
        if i+1 == total_steps:
            continue
        Us[i + 1] = next_U
        total_flips += flipped
        flips[i] = total_flips

        if i % 1000000 == 0 and i > 0:
            print(str(round(i/total_steps*100, 1)) + " % complete")
            end = time.time()
            print(end - start)
            start = end

    #visualise(spins, temp, "Final") # visualise final state

    return flips

# for plots
num_trials = 1
low_t = 1.5
high_t = 2.5
num_sim_temps = 3
#temperatures_sim = np.linspace(low_t, high_t, num_sim_temps)
temperatures_sim = [1,2,3]

flips_sim = np.zeros((num_sim_temps, total_steps))


# setting up plot arrays
us = []
us_sim = np.zeros((num_trials, num_sim_temps))
fs = []
fs_sim = np.zeros((num_trials, num_sim_temps))
Ss = []
Ss_sim = np.zeros((num_trials, num_sim_temps))
cs = []
cs_sim = np.zeros((num_trials, num_sim_temps))
ms_sim = np.zeros(num_sim_temps)

# very unclean code, shouldn't affect speed tho cos at end
def ave(sim_values):
    ave = np.zeros(num_sim_temps)
    for j in range(num_trials):
        for i in range(num_sim_temps):
            ave[i] += sim_values[j][i]
    return ave/num_trials


# calculate error
def error(sim_values):
    average = ave(sim_values)
    err = np.zeros(num_sim_temps)

    for j in range(num_trials):
        for i in range(num_sim_temps):
            err[i] += (sim_values[j][i] - average[i])**2
    for i in range(num_sim_temps):
        err[i] /= (num_trials - 1)
        err[i] = err[i]**0.5
    return err

for temp in temperatures_sim:
    results = model(temp)
    for i in range(total_steps):
        flips_sim[temp-1][i] = results[i]

flips_sim = np.log(flips_sim)

plt.plot(timesteps, flips_sim[0], label="1")
plt.plot(timesteps, flips_sim[1], label="2")
plt.plot(timesteps, flips_sim[2], label="3")
plt.legend()
plt.xlabel("time steps")
plt.ylabel("log cumulative flips")
plt.title("flips through time")
plt.show()
