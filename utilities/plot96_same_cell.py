import numpy as np
import pickle
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({
    'errorbar.capsize': 4,
    'font.size': 15
})


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


def plot_our(name, mean_96, error_96):
    force_val = FORCE_VALUES_96[name]
    n_of_tests = len(joined_results[name][0])
    trials = joined_results[name][1]

    all_prob_estimates = np.empty((n_of_tests, len(force_val)))
    for i in range(n_of_tests):
        all_prob_estimates[i] = collect_our_single(name, i)
    errors = np.sum(np.square((all_prob_estimates - mean_96) / error_96), -1)
    # największe błędy na początku
    all_prob_estimates = all_prob_estimates[np.argsort(errors)[::-1]]

    experiments_color = (0.85, 0.85, 0.85, 1)
    for i in range(n_of_tests - 2):
        plt.plot(force_val, all_prob_estimates[i] * 100, color=experiments_color, zorder=5)
    plt.plot(force_val, all_prob_estimates[-2] * 100, color=experiments_color, zorder=5,
             label=f"our {n_of_tests} experiments")

    mean_color = (0.3, 0.3, 0.3, 1)
    plt.plot(force_val, all_prob_estimates.mean(axis=0) * 100, color=mean_color, zorder=9,
             label="mean from our experiments")

    best_color = "red"
    best_mean, best_error = confidence(trials, all_prob_estimates[-1] * trials)
    plt.errorbar(force_val, best_mean * 100, best_error * 100, color=best_color,
                 linewidth=2, elinewidth=2, capthick=2,
                 label=f"our experiment most similar to the paper's one", zorder=10)


def collect_our_single(name, experiment_nr):
    detached = joined_results[name][0][experiment_nr]
    trials = joined_results[name][1]
    force_val = FORCE_VALUES_96[name]
    prob_estimates = np.empty(len(force_val))
    for i, f in enumerate(force_val):
        prob_estimates[i] = prob_est(trials, detached.get(f, 0))

    return prob_estimates


def plot_generic(name):
    force_val_96 = FORCE_VALUES_96[name]

    plt.figure()
    n_of_cells = joined_results[name][1]
    plt.title(f"{name.title()} force, experiments with {n_of_cells} cells")
    plt.xlabel("force (scaled)")
    plt.ylabel("% detached")
    plt.xticks(force_val_96)

    mean_96, error_96 = confidence(*DATA_96[name])
    plt.errorbar(force_val_96, mean_96 * 100, error_96 * 100, zorder=10,
                 linewidth=2, elinewidth=2, capthick=2,
                 label=f"experiment from 1996 paper")
    plot_our(name, mean_96, error_96)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    filenames = [
        '16-08-2019_10h30m17s',
        '16-08-2019_16h06m34s',
        '16-08-2019_19h31m33s',
        '17-08-2019_03h31m35s',
        '17-08-2019_13h17m22s',
        '17-08-2019_18h42m49s',
        '18-08-2019_18h37m37s'
    ]

    joined_results = dict()
    for filename in filenames:
        with open(f"../results/test96_same_cell/{filename}.pickle", 'rb') as file:
            test_results = pickle.load(file)
        for name in test_results:
            if name not in joined_results:
                joined_results[name] = [[], test_results[name][1]]
            # join only counters with the same number of trials
            assert joined_results[name][1] == test_results[name][1]
            # joining list of counters
            joined_results[name][0] += test_results[name][0]

    for name in joined_results:
        plot_generic(name)
