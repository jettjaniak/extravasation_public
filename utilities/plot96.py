import numpy as np
import matplotlib.pyplot as plt


FORCE_VALUES_96 = dict(
    normal=[1, 1.4, 1.8, 2.2, 2.6, 3, 3.4, 3.8],
    tangential=[2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
)
FORCE_VALUES_96['shear'] = FORCE_VALUES_96['tangential']

FORCE_VALUES = dict(
    normal=[0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.4, 3.8],
    tangential=[2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
    shear=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
)

TRIALS_96 = 50
DATA_96 = dict(
    normal=(TRIALS_96, np.array([0, 12, 32, 60, 84, 92, 98, 100]) / 2),
    tangential=(TRIALS_96, np.array([0, 0, 2, 22, 44, 70, 72, 80, 86, 90, 92, 98, 100]) / 2),
    shear=(TRIALS_96, np.array([2, 20, 46, 70, 80, 90, 94, 98, 100, 100, 100, 100, 100]) / 2)
)

data_our = {}
with open('../results/test96.txt') as file:
    for line in file.readlines():
        force_name_and_trials, result_str = line.split(': ')
        result = np.array(result_str.strip().split(', '), dtype='int')
        force_name, trials_str = force_name_and_trials.split(' ')
        n_of_trials = int(trials_str[1:-1])
        data_our[force_name] = (n_of_trials, result)


def confidence(trials, successes, z=1):
    """Wilson score interval"""
    if trials == 0:
        return 0

    p_hat = successes / trials
    denominator = 1 + z**2 / trials
    mean = (p_hat + z**2 / (2 * trials)) / denominator
    error = (z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials)) / denominator
    return mean, error


def prob_est(trials, successes):
    return successes / trials


def sd_est(p_est, trials):
    return np.sqrt((p_est * (1 - p_est)) / trials)


def plot_generic(name):
    force_val_96 = FORCE_VALUES_96[name]
    force_val = FORCE_VALUES[name]
    trials_our = data_our[name][0]

    plt.figure()
    plt.title(f"{name.title()} force")
    plt.xlabel("scaled force")
    plt.ylabel("% detached")
    plt.xticks(force_val)

    mean_96, error_96 = confidence(*DATA_96[name])
    plt.errorbar(force_val_96, mean_96 * 100, error_96 * 100, label=f"1996 paper, {TRIALS_96} trials")

    mean_our, error_our = confidence(*data_our[name])
    plt.errorbar(force_val, mean_our * 100, error_our * 100, label=f"our experiment, {trials_our} trials")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_generic("normal")
    plot_generic("tangential")
    plot_generic("shear")
