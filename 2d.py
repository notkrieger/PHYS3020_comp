# 2d Ising model
# for Q2 of project
# based heavily off of 1d model

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import time
from numba import jit, njit # makes code miles faster


"""
2-dimensional Ising model
"""

k = 1  # Boltzmann constant -- currently trying to use "natural units"
N = 100  # Nxn dipoles in the system
P = N*N # true number of dipoles
epsilon = 1  # energy contribution factor??
initial_T = 1  # temperature
spin_probability = 0.5  # when initialising if random.random >= spin_probability, set spin to 1, else -1
total_steps = 1000 * P + 1 # 1000 * N^2 + 1 total steps
timesteps = np.linspace(0, total_steps, total_steps) # for plotting later
animation_rate = 10000 # updates animation every X time steps

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

#animate cos it looks cool
def animate(state, time):
    plt.imshow(state)
    plt.title(str(round(time/(total_steps - 1) * 100, 3)) + " %")
    plt.pause(0.00001)
    plt.clf()

# attempt to speed up program, would remove need for copy and running solve_U twice
# solve next U value given a state and a flipped dipole at (row, col)
@jit
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
@jit
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
@jit
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

@jit
def log_multiplicity(state):
    ## use Stirlings approximation
    up = 0
    for i in range(N):
        for j in range(N):
            if state[i][j] == 1:
                up += 1
    down = P - up
    #print("up percent: " + str(up/P*100))
    if up == 0 or down == 0:
        return np.log(1) # to avoid divide by 0 error
    else: # use stirling approximation
        return P * np.log(P) - up * np.log(up) - down * np.log(down)

@jit
def model(temp): # one dimensional Ising Model
    # variables
    beta = 1/(k*temp)
    Us = np.linspace(0, total_steps, total_steps)  # list of Us
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
    Ss = np.linspace(0, total_steps, total_steps)
    for i in range(total_steps):
        if i >= total_steps/10 - 1000:
            Ss[i] = log_multiplicity(spins)
        # animate the process, slows down the simulaiton massively, but looks cool
        if i % animation_rate == 0:
            #animate(spins, i)
            continue

        spins, next_U, flipped = metropolis(spins, beta, Us[i])
        if i+1 > total_steps:
            continue
        Us[i + 1] = next_U

    # calculate stuff for plots
    U_ave = np.mean(Us[-int(total_steps / 10):])  # time average value of U : ⟨U⟩

    S = np.mean(Ss[-int(total_steps / 10):])  # entropy S = k ln(g)
    #print("S: " + str(S))

    f = U_ave - temp * S  # free energy F = U - TS
    # Var(U) = ⟨U^2⟩ - ⟨U⟩^2
    # only use last 10% of time steps for accurate results
    # this just worked idk
    c = np.var(Us[-int(total_steps / 10):]) / (temp ** 2)

    m = abs(np.mean(spins))  # average spin m = M/mu*N = s_bar

    return U_ave/P, f/P, S/P, c/P, m

# for plots
num_trials = 3
low_t = 0.1
high_t = 3
num_sim_temps = 10
temperatures_sim = np.linspace(low_t, high_t, num_sim_temps)
#temperatures_sim = [1,2,3]

all_spins = []


# very unclean code, shouldn't affect speed tho cos at end
@jit
def ave(sim_values):
    ave = np.zeros(num_sim_temps)
    for j in range(num_trials):
        for i in range(num_sim_temps):
            ave[i] += sim_values[j][i]
    return ave/num_trials

# calculate error
@jit
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

def make_plots(temps_sim):
    # setting up plot arrays
    us_sim = np.zeros((num_trials, num_sim_temps))
    fs_sim = np.zeros((num_trials, num_sim_temps))
    Ss_sim = np.zeros((num_trials, num_sim_temps))
    cs_sim = np.zeros((num_trials, num_sim_temps))
    ms_sim = np.zeros(num_sim_temps)

    # simulate solutions
    for trial in range(num_trials):
        print(trial)
        start_total = time.time()
        for i in range(num_sim_temps):
            start = time.time()
            temp = temps_sim[i]
            print('starting simulation at: ' + str(temp) + " C")
            results = model(temp)
            us_sim[trial][i] = results[0]
            fs_sim[trial][i] = results[1]
            Ss_sim[trial][i] = results[2]
            cs_sim[trial][i] = results[3]
            ms_sim[i] = results[4]  # dont need to average or calc uncertainty for m
            end = time.time()
            print("sim time: " + str(end - start))
        end_total = time.time()
        print("1 temp range sim time: " + str(end_total - start_total))

    # plot u
    plt.errorbar(temps_sim, ave(us_sim), ls="--", yerr=error(us_sim), ecolor="k")
    plt.title("plot of u against T")
    plt.ylabel('u')
    plt.xlabel("temperature")
    plt.show()

    #plot f
    plt.errorbar(temps_sim, ave(fs_sim), ls="--", yerr=error(fs_sim), ecolor="k")
    plt.title("plot of f against T")
    plt.ylabel('f')
    plt.xlabel("temperature")
    plt.show()

    # plot S

    plt.errorbar(temps_sim, ave(Ss_sim), ls="--", yerr=error(Ss_sim), ecolor="k")
    plt.title("plot of S against T")
    plt.ylabel('S')
    plt.xlabel("temperature")
    plt.show()

    #plot c
    plt.errorbar(temps_sim, ave(cs_sim), ls="--", yerr=error(cs_sim), ecolor="k")
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



make_plots(temperatures_sim)


