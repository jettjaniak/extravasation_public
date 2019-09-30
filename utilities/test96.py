import multiprocessing
import numpy as np

from wrappers.simulation_engine import SimulationState, Parameters
from examples import DEFAULT_PARAMETERS, DEFAULT_SETTINGS

settings = DEFAULT_SETTINGS.copy()
settings['test96'] = True
p = Parameters(**DEFAULT_PARAMETERS, **settings)

N_STEPS_FALLING = 1e5
N_STEPS_TEST = 5e6


def generic_iteration(**kwargs):
    ss = SimulationState(p, 0.1 / p.k_on_0, 0.0225, kwargs['seed'])
    del kwargs['seed']
    ss.simulate_without_history(N_STEPS_FALLING)  # falling
    ss.simulate_without_history(N_STEPS_TEST, stop_if_no_bonds=True, **kwargs)
    if ss.n_bonds == 0:
        return 1
    else:
        return 0


def tangential_iteration(tangential_force, seed):
    return generic_iteration(tangential=tangential_force, seed=seed)


def shear_iteration(shear_force, seed):
    return generic_iteration(shear=shear_force, seed=seed)


def normal_iteration(normal_force, seed):
    return generic_iteration(normal=normal_force, seed=seed)


TANGENTIAL_COEF = 1e-5 * 6 * np.pi * p.mu * p.k_on_0 * p.r_c**2
tangential_forces = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]

SHEAR_COEF = 1e-5 * p.k_on_0
shear_forces = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]

NORMAL_COEF = 1e-3 * 6 * np.pi * p.mu * p.k_on_0 * p.r_c**2
normal_forces = [0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.4, 3.8]

N_TRIALS = 48


def run_test(name, forces, iteration_func, coef):
    uint_info = np.iinfo(np.uint)
    with open("../results/test96_2.txt", 'a') as file:
        print(f"{name.upper()} FORCE SIMULATIONS")
        detached_list = []
        for f in forces:
            print(f"Force of magnitude {f}.", end=' ')
            seeds = np.random.randint(uint_info.max, size=N_TRIALS, dtype='uint')
            args = tuple(zip(N_TRIALS * [f * coef], seeds))
            with multiprocessing.Pool() as pl:
                detached = sum(pl.starmap(iteration_func, args))
            detached_list.append(str(detached))
            print(f"{detached} detached.")
        file.write(f"{name} ({N_TRIALS}): {', '.join(detached_list)}\n")


if __name__ == '__main__':
    run_test("normal", normal_forces, normal_iteration, NORMAL_COEF)
    run_test("tangential", tangential_forces, tangential_iteration, TANGENTIAL_COEF)
    run_test("shear", shear_forces, shear_iteration, SHEAR_COEF)
