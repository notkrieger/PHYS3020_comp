import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn


"""
1-dimensional model only, 2D will be in another file
"""


k = 1  # Boltzmann constant -- currently trying to use "natural units"
N = 100  # number of dipoles in the system
epsilon = 1  # energy contribution factor??
initial_T = 2  # temperature
spin_probability = 0.5  # when initialising if random.random >= spin_probability, set spin to 1, else -1


# visualise the data in a heatmap
def visualise(state, temp):
    plt.figure(figsize=(10, 2))
    seaborn.heatmap([state]*2, cmap="Greys", cbar=False, xticklabels=False,
                    yticklabels=False)  # had make it 2d for heatmap to work :(
    plt.title("Final spin state at temperature: " + str(temp))
    plt.show()


# solve U for some given state
def solveU(state):
    U = 0
    for i in range(N):  # just count interaction between dipole i and i + 1 --- do i need to x2???
        right = (i + 1) % N  # to account for periodic boundaries
        U += state[i]*state[right]  # add to U
    return U * -epsilon  # multiply by negative eps and return


# metropolis algorithm
# inputs:
    # state - some microstate
    # beta - defined as 1/kT, used to determine a probability

    # returns:
    #   - next state -- may have one flipped dipole compared to state
def metropolis(state, beta):

    # choose random dipole
    random_dipole = np.random.randint(0, N, 1)[0]
    next_state = copy.deepcopy(state) # make copy
    next_state[random_dipole] *= -1 # flip dipole
    U = solveU(state)
    next_U = solveU(next_state)
    del_U = next_U - U
    if del_U <= 0:
        return next_state
    else:
        flip = np.random.random()
        if flip <= np.exp(-beta*del_U):
            return next_state
        else:
            return state


def model(): # one dimensional Ising Model
    temp = initial_T
    beta = 1/(k*temp)
    spins = np.random.random(N) # initialise spins
    for i in range(N):
        if spins[i] >= spin_probability:
            spins[i] = 1
        else:
            spins[i] = -1
    # run metropolis algorithm such that each dipole is given about 1000 chances to flip
    # for each dipole there is a 1/N chance of being selected. therefore, need about 1000N iterations
    total_steps = 1000 * N
    for i in range(total_steps):
        spins = metropolis(spins, beta)
    visualise(spins, temp) # one d visualisation

model()
