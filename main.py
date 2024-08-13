import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from math import comb


"""
1-dimensional model only, 2D will be in another file
"""


k = 1  # Boltzmann constant -- currently trying to use "natural units"
N = 100  # number of dipoles in the system
epsilon = 1  # energy contribution factor??
initial_T = 2  # temperature
spin_probability = 0.5  # when initialising if random.random >= spin_probability, set spin to 1, else -1


# visualise the data in a heatmap
def visualise(state, temp, string):
    plt.figure(figsize=(10, 2))
    seaborn.heatmap([state]*2, cmap="Greys", cbar=False, xticklabels=False,
                    yticklabels=False)  # had make it 2d for heatmap to work :(
    plt.title(string + " spin state at temperature: " + str(temp))
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
def metropolis(state, beta): # did not have to speed up as much because N << N^2 (for N = 100)
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

Z = 0  # partition function

def find_multiplicity(state):
    up_count = 0
    for dipole in state:
        if dipole > 0:
            up_count += 1
    return float(comb(len(state), up_count))


def model(temp): # one dimensional Ising Model
    # variables
    beta = 1/(k*temp)
    # initialise spins
    spins = np.random.random(N)
    for i in range(N):
        if spins[i] >= spin_probability:
            spins[i] = 1
        else:
            spins[i] = -1
    #visualise(spins, temp, "Initial") # visualise intial state

    # run metropolis algorithm such that each dipole is given about 1000 chances to flip
    # for each dipole there is a 1/N chance of being selected. therefore, need about 1000N iterations
    # should I round up to maybe 1250ish??? i think 1000N should be fine
    total_steps = 1000 * N
    for i in range(total_steps):
        spins = metropolis(spins, beta)

    # calculate stuff for plots
    #
    U = solveU(spins) # internal energy due to nearest-neighbour interactions
    S = k * np.log(find_multiplicity(spins)) # entropy S = k ln(g)
    f = U - temp * S # free energy F = U - TS
    c = (U**2 - (-N*epsilon*np.tanh(beta*epsilon))**2)/(k*temp**2)
    m = np.mean(spins) # average spin m = M/mu*N = s_bar

    #visualise(spins, temp, "Final") # visualise final state
    return U/N, f/N, S/N, c/N, m

# for plots
num_trials = 10
low_t = 0.1
high_t = 3
num_sim_temps = 15
temperatures_exact = np.linspace(low_t, high_t, 1000)
temperatures_sim = np.linspace(low_t, high_t, num_sim_temps)

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


## very unclean code, shouldnt affect speed tho cos at end
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


def make_plots(temps_sim, temps_exact):
    # theoretical plots
    for trial in range(num_trials):
        print(trial)
        for i in range(num_sim_temps):
            temp = temps_sim[i]
            print(temp)
            results = model(temp)
            us_sim[trial][i] = results[0]
            fs_sim[trial][i] = results[1]
            Ss_sim[trial][i] = results[2]
            cs_sim[trial][i] = results[3]
            ms_sim[i] = results[4] # dont need to average or calc uncertainty for m
    print(us_sim)
    print(ave(us_sim))
    print(error(us_sim))
    # exact solutions
    for temp in temps_exact:
        beta = 1 / (k * temp)
        us.append(-epsilon*np.tanh(beta*epsilon))
        fs.append(-epsilon-k*temp*np.log(1 + np.exp(-2*epsilon*beta)))
        Ss.append(epsilon/temp*(1 - np.tanh(beta*epsilon)) + k*np.log(1 + np.exp(-2*epsilon*beta)))
        cs.append(epsilon**2*beta/(temp * np.cosh(beta*epsilon)**2))

    # plot u
    plt.plot(temps_exact, us, label="exact")
    plt.errorbar(temps_sim, ave(us_sim), yerr=error(us_sim), ecolor="k", label="sim")
    plt.legend()
    plt.title("plot of u against T")
    plt.ylabel('u')
    plt.xlabel("temperature")
    plt.show()
    #plot f
    plt.plot(temps_exact, fs, label="exact")
    plt.errorbar(temps_sim, ave(fs_sim), yerr=error(fs_sim), ecolor="k", label="sim")
    plt.legend()
    plt.title("plot of f against T")
    plt.ylabel('f')
    plt.xlabel("temperature")
    plt.show()
    #plot S
    plt.plot(temps_exact, Ss, label="exact")
    plt.errorbar(temps_sim, ave(Ss_sim), yerr=error(Ss_sim), ecolor="k", label="sim")
    plt.legend()
    plt.title("plot of S against T")
    plt.ylabel('S')
    plt.xlabel("temperature")
    plt.show()
    #plot c
    plt.plot(temps_exact, cs, label="exact")
    plt.errorbar(temps_sim, ave(cs_sim), yerr=error(cs_sim), ecolor="k",  label="sim")
    plt.legend()
    plt.title("plot of c against T")
    plt.ylabel('c')
    plt.xlabel("temperature")
    plt.show()
    # plot m
    plt.errorbar(temps_sim, ms_sim)
    plt.title("plot of m against T")
    plt.ylabel('m')
    plt.xlabel("temperature")
    plt.show()


make_plots(temperatures_sim, temperatures_exact)