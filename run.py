"""
run.py

Command-line utility to run PCP solvers in bulk.
"""
import os
import sys
import time
import traceback
import importlib
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import src.correct_solver as correct_solver
import numpy as np
import pandas as pd

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2,
      'axes.labelsize': 12,
      'axes.titlesize': 14,
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 10

# The name of the solver script.
# Only directories that have this exact file will be considered.
SOLVER = 'solver'
SOLVER_FILE = SOLVER + '.py'

# Solver function name.
SOLVER_FUNC = 'solve'

# The alphabet.
ALPHABET = ['0', '1']

def _get_valid_solvers(root):
    '''
    Helper function to fetch a dictionary of paths to valid solvers.
    The argument to this function is the root directory to start the search.
    '''
    valid_solvers = {}

    # Walk through directories and only consider those that don't have
    # any subdirectories.
    for r, dirs, files in os.walk(root):
        # Then, this folder is a valid only if it contains the solver file.
        if SOLVER_FILE in files:
            name = os.path.basename(r)
            valid_solvers[name] = r

    return valid_solvers

def _random_target(n):
    '''
    Helper function to generate a random target of length n by randomly
    sampling, with replacement, from the list ALPHABET.
    '''
    return ''.join(random.choices(ALPHABET, k=n))

def _generate_targets(max_n, n):
    '''
    Generate n targets from length 5 to max_n (inclusive) each, for a
    total of (max_n - 5 + 1) * n targets.

    Also calculates the correct number of deduplications using the solve
    function in the correct_solver script.

    This helper function returns a pandas DataFrame.
    '''
    rows = []

    for l in range(5, max_n + 1):
        for _ in range(n):
            target = _random_target(l)
            row = [target, l, correct_solver.solve(target)]
            rows.append(row)

    return pd.DataFrame(rows, columns=['target', 'length', 'steps'])

def _import_solver(path):
    '''
    Helper function to import the solver at the given path.
    Returns the actual solve function, which takes the target string and
    returns the calculated number of duplications.
    '''
    solver = importlib.import_module(os.path.join(path, SOLVER).replace('/', '.'))
    func = getattr(solver, SOLVER_FUNC)

    return func

def _run_sim(name, func, targets, t):
    '''
    Given a solver function and a pandas DataFrame of targets, runs
    the solver for every target in the dataframe and outputs statistics in
    the out directory.
    '''
    # Copy the targets dataframe.
    copy = targets.copy(deep=True)

    calc_steps = []
    calc_times = []
    calc_corrects = []
    for _, row in targets.iterrows():
        target = row['target']
        length = row['length']
        steps = row['steps']

        # Run function and time it.
        start = time.time()
        calc_step = func(target)
        calc_time = time.time() - start
        calc_correct = int(steps == calc_step)

        # Add these to the list.
        calc_steps.append(calc_step)
        calc_times.append(calc_time)
        calc_corrects.append(calc_correct)

        if calc_time > t:
            print('.Calc time exceeded', end='', flush=True)
            break

        print('.', end='', flush=True)

    # Add the columns to the dataframe.
    missing = len(targets) - len(calc_steps)
    copy[name + '_steps'] = np.append(calc_steps, np.repeat(np.nan, missing))
    copy[name + '_time'] = np.append(calc_times, np.repeat(np.nan, missing))
    copy[name + '_correct'] = np.append(calc_corrects, np.repeat(np.nan, missing))

    return copy

def _average(df, name):
    '''
    Given a pandas DataFrame containing run info and the name of this solver,
    returns useful averages.
    '''
    # Plot runtime vs target length.
    length = df[['length', name + '_time']].groupby(['length'], as_index=False).mean()

    # Plot runtime vs duplication distance.
    dist = df[['steps', name + '_time']].groupby(['steps'], as_index=False).mean()

    return length, dist

def _plot(df, x, y, label, t):
    '''
    Generates a plot and returns the object.
    '''
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x=x, y=y, ax=ax, marker='o', label=label)
    if df[y].max() > t:
        ax.axhline(t, linestyle='--', color='r', linewidth=0.5,
                   label='cutoff ({})'.format(t))
        ax.legend()

    return fig, ax

def run(name, path, targets, t):
    '''
    Given a solver name and the path to that solver, runs all analysis on
    the targets.
    '''
    # DF that contains all run info.
    sys.path.append(path)
    df = _run_sim(name, _import_solver(path), targets, t)
    sys.path.remove(path)

    # Save the df to the folder.
    df.to_csv(os.path.join(path, 'run.csv'), index=False)

    # Averages.
    avg_length, avg_dist = _average(df, name)

    # Save these dfs.
    avg_length.to_csv(os.path.join(path, 'avg_time_length.csv'), index=False)
    avg_dist.to_csv(os.path.join(path, 'avg_time_dist.csv'), index=False)

    # Generate plots.
    plot_length, _ = _plot(avg_length, 'length', name + '_time', name, t)
    plot_length.savefig(os.path.join(path, 'avg_time_length.png'),
                                     bbox_inches='tight', dpi=300)
    plot_dist, _ = _plot(avg_dist, 'steps', name + '_time', name, t)
    plot_dist.savefig(os.path.join(path, 'avg_time_dist.png'),
                                   bbox_inches='tight', dpi=300)

    return df, avg_length, avg_dist

def run_all(solvers, targets, root, t):
    '''
    Run all simulations on the given solvers on the given targets.
    '''
    dfs = []
    dfs_length = []
    dfs_dist = []

    error = []
    for name, path in solvers.items():
        try:
            print('Running solver for {}...'.format(name), end='', flush=True)
            df, avg_length, avg_dist = run(name, path, targets, t)
            dfs.append(df)
            dfs_length.append(avg_length)
            dfs_dist.append(avg_dist)

            print('.Done', flush=True)
        except Exception as e:
            error.append(name)

            # Output stacktrace file.
            with open(os.path.join(path, 'error.txt'), 'w') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
            print('.Error written to error.txt', flush=True)

    # Concatenate the dataframes.
    df = pd.concat(dfs, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df_length = pd.concat(dfs_length, axis=1)
    df_length = df_length.loc[:, ~df_length.columns.duplicated()]
    df_dist = pd.concat(dfs_dist, axis=1)
    df_dist = df_dist.loc[:, ~df_dist.columns.duplicated()]

    # Names that didn't have errors
    names = list(set(solvers.keys()) - set(error))

    # Calculate accuracies.
    cols = [name + '_correct' for name in names]
    accuracies = df[cols].mean()
    accuracies.to_csv(os.path.join(root, 'accuracies.csv'), header=False)

    # Save csvs.
    df.to_csv(os.path.join(root, 'run_all.csv'), index=False)
    df_length.to_csv(os.path.join(root, 'avg_time_length_all.csv'), index=False)
    df_dist.to_csv(os.path.join(root, 'avg_time_dist_all.csv'), index=False)

    # Remove _time suffix from column names for plotting reasons.
    df_length.columns = df_length.columns.str.replace('_time', '')
    df_dist.columns = df_dist.columns.str.replace('_time', '')

    melted_length = pd.melt(df_length, id_vars=['length'], value_vars=names,
                            var_name='name', value_name='time')
    melted_dist = pd.melt(df_dist, id_vars=['steps'], value_vars=names,
                            var_name='name', value_name='time')

    # Generate plots.
    _plot_all(melted_length, 'length', 'time', 'name', t,
              os.path.join(root, 'avg_time_length_all.png'))
    _plot_all(melted_dist, 'steps', 'time', 'name', t,
              os.path.join(root, 'avg_time_dist_all.png'))

def _plot_all(df, x, y, hue, t, path):
    '''
    Helper function to plot all the simulation results.
    '''
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, marker='o', alpha=0.5)
    if df[y].max() > t:
        ax.axhline(t, linestyle='--', color='r', linewidth=0.5,
                   label='cutoff ({})'.format(t))
        ax.legend()
    fig.savefig(path, bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    import math
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str,
                        help=('Root directory that contains subdirectories ')
                              + 'containing {}'.format(SOLVER_FILE))
    parser.add_argument('max_n', type=int,
                        help=('Maximum length of target to simulate. '
                              + 'Has to be greater than 5.'))
    parser.add_argument('-i', type=int, default=3,
                        help='Number of strings of each length to generate.')
    parser.add_argument('-t', type=float, default=math.inf,
                        help=('The maximum number of seconds allowed for a '
                              + 'single calculation. Any calculations that '
                              + 'exceed this are terminated.'))
    args = parser.parse_args()

    if args.max_n < 5:
        raise Exception('max_n must be greater than 5!')

    solvers = _get_valid_solvers(args.root)
    print('Found {} valid solvers.'.format(len(solvers)))

    targets = _generate_targets(args.max_n, args.i)
    print('Generated {} targets'.format(len(targets)))

    run_all(solvers, targets, args.root, args.t)
