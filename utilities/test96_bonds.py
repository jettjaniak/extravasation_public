import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from wrappers.simulation_engine import SimulationState, Parameters
from examples import DEFAULT_PARAMETERS, DEFAULT_SETTINGS

settings = DEFAULT_SETTINGS.copy()
settings['test96'] = True
p = Parameters(**DEFAULT_PARAMETERS, **settings)

N_STEPS_FALLING = 1e5
N_TRIALS_FALLING = 50
N_SAMPLES = 100


def iteration(_):
    ss = SimulationState(p, 0.1 / p.k_on_0, 0.0225)
    ss.simulate_without_history(N_STEPS_FALLING)  # falling
    return ss.n_bonds


if __name__ == '__main__':
    print("Starting...")
    means = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        with multiprocessing.Pool() as pl:
            n_bonds = np.array(pl.map(iteration, N_TRIALS_FALLING * [0]), dtype='int')
        means[i] = n_bonds.mean()

    plt.hist(means, density=True)
    plt.title(f"Distribution of the average number of bonds in {N_TRIALS_FALLING} trials")
    plt.xlabel("average number of bonds")
    plt.ylabel(f"empirical density ({N_SAMPLES} samples)")
    plt.savefig("../results/test96_bonds.png")

    with open("../results/test96_bonds.txt", 'w') as file:
        file.write('\n'.join(map(str, means)))