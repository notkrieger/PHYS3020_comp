# 2d Ising model
# for Q2 of project
# based heavily off of 1d model

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import time
from numba import njit # makes code miles faster


"""
2-dimensional Ising model
"""

k = 1  # Boltzmann constant -- currently trying to use "natural units"


epsilon = 1  # energy contribution factor??

spin_probability = 0.5  # when initialising if random.random >= spin_probability, set spin to 1, else -1

animation_rate = 10000 # updates animation every X time steps


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
    plt.title(str(time))
    plt.pause(0.00001)
    plt.clf()

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
    next_U = next_state_U(state, rrow, rcol, last_U, N) # works

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

@njit
def log_multiplicity(state, N):
    P = N*N
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

@njit
def model(temp, N): # 2 dimensional Ising Model
    # variables
    P = N * N  # true number of dipoles
    total_steps = 1000 * P + 1  # 1000 * N^2 + 1 total steps
    beta = 1/(k*temp)
    Us = np.linspace(0, total_steps, total_steps)  # list of Us
    # initialise spins
    spins = np.random.random((N, N))
    lastX = total_steps/100 # 1% of total steps
    for i in range(N):
        for j in range(N):
            if spins[i][j] >= spin_probability:
                spins[i][j] = 1
            else:
                spins[i][j] = -1
    #visualise(spins, temp, "Initial") # visualise intial state
    Us[0] = solveU(spins, N) # define U at time = 0
    Ss = np.linspace(0, total_steps, total_steps)
    for i in range(total_steps):
        if i >= total_steps - lastX:
            Ss[i] = log_multiplicity(spins, N)
        # animate the process, slows down the simulaiton massively, but looks cool
        #if i % animation_rate == 0:
            #animate(spins, i)
            #continue

        spins, next_U, flipped = metropolis(spins, beta, Us[i], N)
        Us[i + 1] = next_U # update Us

    # calculate stuff for plots
    U_ave = np.mean(Us[-int(lastX):])  # time average value of U : ⟨U⟩

    #S = np.mean(Ss[-int(lastX):])  # entropy S = k ln(g)
    S = Ss[-1]
    #print("S: " + str(S))

    f = U_ave - temp * S  # free energy F = U - TS
    # Var(U) = ⟨U^2⟩ - ⟨U⟩^2
    # only use last 10% of time steps for accurate results
    # this just worked idk
    c = np.var(Us[-int(lastX):]) / (temp ** 2)

    m_pos = 0  # average spin m = M/mu*N = s_bar
    m_neg = 0
    for i in range(N):
        for j in range(N):
            if spins[i][j] > 0:
                m_pos += 1
            else:
                m_neg += 1

    return U_ave/P, f/P, S/P, c/P, m_pos/P, m_neg/P

# for plots
num_trials = 100
low_t = 1.8
high_t = 3.5
num_sim_temps = 15
temperatures_sim = np.linspace(low_t, high_t, num_sim_temps)
num_Ns = 3
Ns_ = [5, 10, 50]

@njit
def ave(sim_values):
    ave = np.zeros(num_sim_temps)
    for j in range(num_trials):
        for i in range(num_sim_temps):
            ave[i] += sim_values[j][i]
    return ave/num_trials

# calculate error
@njit
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

@njit
def simulate(temps_sim):
    Ns = [5, 10, 50]
    # setting up plot arrays
    us_sim = np.zeros((num_Ns, num_trials, num_sim_temps))
    fs_sim = np.zeros((num_Ns, num_trials, num_sim_temps))
    Ss_sim = np.zeros((num_Ns, num_trials, num_sim_temps))
    cs_sim = np.zeros((num_Ns, num_trials, num_sim_temps))
    ms_sim = np.zeros((num_Ns, num_sim_temps))
    ms_p_ave = np.zeros((num_Ns, num_trials, num_sim_temps))
    ms_n_ave = np.zeros((num_Ns, num_trials, num_sim_temps))

    # simulate solutions
    for n in range(num_Ns):
        N_ = Ns[n]
        print('schmoving')
        for trial in range(num_trials):
            for i in range(num_sim_temps):
                temp = temps_sim[i]
                results = model(temp, N_)
                us_sim[n][trial][i] = results[0]
                fs_sim[n][trial][i] = results[1]
                Ss_sim[n][trial][i] = results[2]
                cs_sim[n][trial][i] = results[3]
                ms_p_ave[n][trial][i] = results[4]
                ms_n_ave[n][trial][i] = results[5]
    return us_sim, fs_sim, Ss_sim, cs_sim, ms_p_ave, ms_n_ave


def plot(us_sim, fs_sim, Ss_sim, cs_sim, ms_p, ms_n):
    for n in range(num_Ns):
        N_ = Ns_[n]
        # plot u
        plt.errorbar(temperatures_sim, ave(us_sim[n]), ls="--", yerr=error(us_sim[n]),
                     label=str(N_)+"$^{2}$", capsize=3)
        plt.title("plot of u against T")
        plt.legend()
        plt.ylabel('u')
        plt.xlabel("temperature")
    #plt.show()
    plt.clf()
    for n in range(num_Ns):
        N_ = Ns_[n]
        #plot f
        plt.errorbar(temperatures_sim, ave(fs_sim[n]), ls="--", yerr=error(fs_sim[n]),
                     label=str(N_)+"$^{2}$", capsize=3)
        plt.title("plot of f against T")
        plt.legend()
        plt.ylabel('f')
        plt.xlabel("temperature")
    #plt.show()
    plt.clf()

    # plot S
    for n in range(num_Ns):
        N_ = Ns_[n]
        plt.errorbar(temperatures_sim, ave(Ss_sim[n]), ls="--", yerr=error(Ss_sim[n]),
                     label=str(N_)+"$^{2}$", capsize=3)
        plt.title("plot of S against T")
        plt.legend()
        plt.ylabel('S')
        plt.xlabel("temperature")
    #plt.show()
    plt.clf()

    #plot c
    for n in range(num_Ns):
        N_ = Ns_[n]
        plt.errorbar(temperatures_sim, ave(cs_sim[n]), ls="--", yerr=error(cs_sim[n]),
                     label=str(N_)+"$^{2}$", capsize=3)
        plt.title("plot of c against T")
        plt.ylabel('c')
        plt.xlabel("temperature")
        plt.legend()
    #plt.show()
    plt.clf()

    # plot m_pos/m_neg
    for n in range(num_Ns):
        N_ = Ns_[n]
        plt.errorbar(temperatures_sim, ave(ms_p[n]), ls="--", yerr=error(ms_p[n]),
                     label="positive", capsize=3)
        plt.errorbar(temperatures_sim, ave(ms_n[n]), ls="--", yerr=error(ms_n[n]),
                     label="negative", capsize=3, elinewidth=1.5)
        plt.title("mean positive/negative against T for " + str(N_) +"$^{2}$ dipoles")
        plt.legend()
        plt.ylabel('|m|')
        plt.xlabel("temperature")
        plt.show()


us_sim, fs_sim, Ss_sim, cs_sim, ms_p, ms_n = simulate(temperatures_sim)
plot(us_sim, fs_sim, Ss_sim, cs_sim, ms_p, ms_n)


