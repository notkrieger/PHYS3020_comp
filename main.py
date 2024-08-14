import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from math import comb
import time


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
    plt.title(string + " spin state")
    plt.show()


def next_state_U(state, index, lastU): # solves next states U value correctly,
    # quicker then deep copying the old state, and solving U then subtracting difference


    del_U = 0

    right = (index + 1) % N
    # subtract influence of unflipped dipole
    del_U -= state[index] * state[right]
    del_U -= state[(index - 1) % N] * state[index]
    # add change from flipping dipole
    del_U += state[index] * -1 * state[right]
    del_U += state[(index - 1) % N] * state[index] * -1
    del_U *= -epsilon
    del_U += lastU

    return del_U

# solve U for some given state
def solveU(state):
    U = 0
    for i in range(N):  # just count interaction between dipole i and i + 1 --- do i need to x2???
        right = (i + 1) % N  # to account for periodic boundaries
        U += state[i]*state[right]  # add to U
    return U * -epsilon  # multiply by negative eps and return


def solve_U2(state):
    U2 = 0
    for i in range(N):  # just count interaction between dipole i and i + 1 --- do i need to x2???
        right = (i + 1) % N  # to account for periodic boundaries
        U2 += (state[i] * state[right]) ** 2  # add to U
    return U2   # multiply by negative eps and return

# metropolis algorithm
# inputs:
    # state - some microstate
    # beta - defined as 1/kT, used to determine a probability

# returns:
    #   - next state -- may have one flipped dipole compared to state
def metropolis(state, beta, lastU): # did not have to speed up as much because N << N^2 (for N = 100)
    # choose random dipole
    random_dipole = np.random.randint(0, N, 1)[0]
    next_U = next_state_U(state, random_dipole, lastU)
    del_U = next_U - lastU
    if del_U <= 0: # flip
        state[random_dipole] *= -1
        return state, next_U
    else:
        flip = np.random.random()
        if flip <= np.exp(-beta*del_U): # flip
            state[random_dipole] *= -1
            return state, next_U
        else: # dont flip
            return state, lastU

def find_multiplicity(state):
    up_count = 0
    for dipole in state:
        if dipole > 0:
            up_count += 1
    return float(comb(len(state), up_count))


def model(temp): # one dimensional Ising Model
    # variables
    total_steps = 1000 * N
    Us = np.linspace(0, total_steps, total_steps)
    Us2 = np.linspace(0, total_steps, total_steps)
    beta = 1/(k*temp)
    totalU = 0
    totalU2 = 0
    totalm = 0
    # initialise spins
    spins = np.random.random(N)
    for i in range(N):
        if spins[i] >= spin_probability:
            spins[i] = 1
        else:
            spins[i] = -1
    #visualise(spins, temp, "Initial") # visualise intial state

    Us[0] = solveU(spins)
    Us2[0] = solve_U2(spins)
    for i in range(total_steps):
        #print(Us[i])
        spins, next_U = metropolis(spins, beta, Us[i])
        totalU += next_U

        if i + 1 >= total_steps:
            continue

        Us2[i + 1] = solve_U2(spins)
        Us[i+1] = next_U
        totalm += np.mean(spins)

    # calculate stuff for plots
    #
    U_ave = np.mean(Us) / N # time average value of U : ⟨U⟩
    U2_ave = np.mean(Us2) / N # ⟨U^2⟩
    m_ave = totalm / total_steps

    U = solveU(spins)
    S = k * np.log(find_multiplicity(spins)) # entropy S = k ln(g)
    f = U_ave - temp * S # free energy F = U - TS
    c = (U2_ave - U_ave**2)/(k*temp**2) # ⟨U^2⟩ - ⟨U⟩^2 / kT^2
    if temp != 0.5:
        print("⟨U^2⟩: " + str(U2_ave))
        print("⟨U⟩^2: " + str(U_ave ** 2))
        print("sim c  : " + str(c))
        print("exact c: " + str(1/(temp**2 * np.cosh(beta)**2)))
    m = np.mean(spins) # average spin m = M/mu*N = s_bar

    #visualise(spins, temp, "Final") # visualise final state
    return U_ave/N, f/N, S/N, c/N, m

# for plots
num_trials = 3
low_t = 0.1
high_t = 3
num_sim_temps = 10
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

# for histogram of m's
m_aves = np.zeros((num_sim_temps, num_trials))


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


def make_plots(temps_sim, temps_exact):
    # simulate solutions

    for trial in range(num_trials):
        start = time.time()
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
            m_aves[i][trial] = results[4]
        end = time.time()
        print(end - start)

    # calculate exact solutions
    for temp in temps_exact:
        beta = 1 / (k * temp)
        us.append(-epsilon*np.tanh(beta*epsilon))
        fs.append(-epsilon-k*temp*np.log(1 + np.exp(-2*epsilon*beta)))
        Ss.append(epsilon/temp*(1 - np.tanh(beta*epsilon)) + k*np.log(1 + np.exp(-2*epsilon*beta)))
        cs.append(epsilon**2*beta/(temp * np.cosh(beta*epsilon)**2))
    """
    for i in range(num_sim_temps):
        plt.hist(m_aves[i], range=[-1, 1], bins=50)
        plt.title("m values at " + str(temperatures_sim[i]) + " ε/k, with " + str(N) + " dipoles")
        plt.ylabel("frequency")
        plt.xlabel("m")
        plt.show()

    
    # plot u
    plt.plot(temps_exact, us, label="exact")
    plt.errorbar(temps_sim, ave(us_sim), ls="--", yerr=error(us_sim), ecolor="k", label="sim")
    plt.legend()
    plt.title("plot of u against T")
    plt.ylabel('u')
    plt.xlabel("temperature")
    plt.show()
    
    #plot f
    plt.plot(temps_exact, fs, label="exact")
    plt.errorbar(temps_sim, ave(fs_sim), ls="--", yerr=error(fs_sim), ecolor="k", label="sim")
    plt.legend()
    plt.title("plot of f against T")
    plt.ylabel('f')
    plt.xlabel("temperature")
    plt.show()
    #plot S
    plt.plot(temps_exact, Ss, label="exact")
    plt.errorbar(temps_sim, ave(Ss_sim), ls="--", yerr=error(Ss_sim), ecolor="k", label="sim")
    plt.legend()
    plt.title("plot of S against T")
    plt.ylabel('S')
    plt.xlabel("temperature")
    plt.show()
    """
    #plot c
    plt.plot(temps_exact, cs, label="exact")
    plt.errorbar(temps_sim, ave(cs_sim), ls="--", yerr=error(cs_sim), ecolor="k",  label="sim")
    #plt.plot(temps_sim, ave(cs_sim), label="sim")
    plt.legend()
    plt.title("plot of c against T")
    plt.ylabel('c')
    plt.xlabel("temperature")
    plt.show()
    """
    # plot m
    plt.errorbar(temps_sim, ms_sim)
    plt.title("plot of m against T")
    plt.ylabel('m')
    plt.xlabel("temperature")
    #plt.show()
    """


make_plots(temperatures_sim, temperatures_exact)