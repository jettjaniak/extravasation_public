import multiprocessing
import pickle
import os
import numpy as np
from collections import Counter
from datetime import datetime

from wrappers.simulation_engine import (
    CellParameters, ReceptorParameters, SimulationSettings, Parameters, SimulationState
)

from params import DEFAULT_CELL_P, DEFAULT_SIM_P, DEFAULT_REC_P

cell_p = CellParameters(**DEFAULT_CELL_P)

sim_p_dict = DEFAULT_SIM_P.copy()
sim_p_dict['test96'] = True
sim_p = SimulationSettings(**sim_p_dict)

rec_p_dict = DEFAULT_REC_P.copy()
rec_p_dict['n_r'] = 3333
rec1_p = ReceptorParameters(**rec_p_dict)
rec2_p = ReceptorParameters(**rec_p_dict)
rec3_p = ReceptorParameters(**rec_p_dict)

p = Parameters(cell_p, sim_p)
p.add_receptor(rec1_p)
p.add_receptor(rec2_p)
p.add_receptor(rec3_p)

N_STEPS_FALLING = 1e5
N_STEPS_TEST = 5e6

FORCES = dict(
    tangential=[2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
    normal=[1, 1.4, 1.8, 2.2, 2.6, 3, 3.4, 3.8]
)
FORCES['shear'] = FORCES['tangential']

COEFFICIENTS = dict(
    tangential=1e-5 * 6 * np.pi * p.cell.mu * p.rec[0].k_on_0 * p.cell.r_c**2,
    normal=1e-3 * 6 * np.pi * p.cell.mu * p.rec[0].k_on_0 * p.cell.r_c**2,
    shear=1e-5 * p.rec[0].k_on_0
)


def iteration(name, test_nr, force, coef, seed):
    ss = SimulationState(p, 0.1 / p.rec[0].k_on_0, 0.0225, seed)
    ss.simulate_without_history(N_STEPS_FALLING)  # falling

    forces_dict = {name: force * coef}
    ss.simulate_without_history(N_STEPS_TEST, stop_if_no_bonds=True, **forces_dict)

    detached = False if ss.n_bonds else True
    return name, test_nr, force, detached, seed


def iteration_callback(result):
    name, test_nr, force, detached, seed = result
    # print(f"name: {name}, test nr {test_nr}, force {force}, seed {seed}: ", end="")
    if detached:
        # print("detached.")
        test_results[name][0][test_nr][force] += 1
    # else:
    #     print("not detached.")


def iteration_error_callback(error):
    print("error callback called!".upper())


def one_test(name, test_nr, n_trials, pool):
    forces = FORCES[name]
    coefficient = COEFFICIENTS[name]
    seeds = np.random.randint(uint_info.max, size=n_trials, dtype='uint')
    # seeds = np.arange(n_trials) + 100
    for force in forces:
        # each seed represents one trial
        for seed in seeds:
            pool.apply_async(iteration, (name, test_nr, force, coefficient, seed),
                             callback=iteration_callback, error_callback=iteration_error_callback)


def many_tests(name, n_tests, n_trials, pool):
    if name not in test_results:
        test_results[name] = ([Counter() for _ in range(n_tests)], n_trials)

    for i in range(n_tests):
        one_test(name, i, n_trials, pool)


if __name__ == '__main__':
    N_TRIALS = 50
    uint_info = np.iinfo(np.uint)
    test_results = dict()

    pool = multiprocessing.Pool()

    many_tests('normal', 15, N_TRIALS, pool)
    many_tests('shear', 15, N_TRIALS, pool)

    pool.close()
    pool.join()

    directory = f'../results/test96_same_cell/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    date_str = datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss")
    with open(f'{directory}/{date_str}.pickle', 'wb') as file:
        pickle.dump(test_results, file)

