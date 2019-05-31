# Caltech IST4 PCP Analysis
Command-line tool to help bulk execution and graph plotting for Caltech IST4 Programming Challenge.

## Prerequisites
- Python 3
- Numpy
- Pandas
- Matplotlib
- Seaborn
All of these are automatically installed with Anaconda.

## Usage
```
usage: run.py [-h] [-i I] [-t T] root max_n

positional arguments:
  root        Root directory that contains subdirectories containing solver.py
  max_n       Maximum length of target to simulate. Has to be greater than 5.

optional arguments:
  -h, --help  show this help message and exit
  -i I        Number of strings of each length to generate.
  -t T        The maximum number of seconds allowed for a single calculation.
              Any calculations that exceed this are terminated.
```

## Instructions
### Directory structure
An example directory structure is contained in the `solvers` directory.
Since every student will submit their own `solver.py`, and calling the function
`solve` will solve the target, each student's `solver.py` must be put in its
own subdirectory. It will be beneficial to name the subdirectory as the
student's name, but without spaces. The example directory contains four
students, each labeled with an integer. They all have their own `solver.py`
submissions, which are contained within their respective folders.

### Batch Execution and Plotting
#### Basic command
From the command-line, run `run.py` from its directory with the appropriate
options. You must provide the root directory that contains student-specific
subdirectories and the maximum target length to generate. Beware that lengths
greater than 30 may take considerable time to run. For our example solvers, to
run each submission for targets with lengths 5 to 20, you would run the command
```
python run.py solvers/example 20
```
Note that the minimum target length is set to 5 because any length less than
that would not make any difference.

Once the script finishes running all the `solver.py`s, it will generate `.csv`
files in each directory that contains information about each run, along with
time-vs-length and time-vs-distance plots. Statistics for the entire run are
produced in the root directory. This includes the aggregation of all individual
runs and plots.

#### Optional arguments
By default, the script generates 3 targets of each length. This can be changed
by using the `-i` argument, but be careful with small values, for there may
be much greater variation.

Most of the time, you don't want this script to take forever just because some
submissions are taking much longer than others. For example, there may be a
submission that is highly optimized while there may be another that runs much
slower than the rest of the submissions. You can give a time limit to how long
each `solve` calculation can take by using the `-t` argument. This indicates
the time limit, in seconds, that any `solve` call can take. When a `solve` call
of one submission takes longer than this time limit, the script moves on to
the next submission in favor of saving time. An example command with a time
limit of three seconds would be
```
python run.py solvers/example 30 -t 3
```
