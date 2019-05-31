'''
solver.py
Joseph Min IST4 PCP 2019

Finds the minimum number of duplications from the seed to the given string.
'''

import sys
import time
import collections


def _findDedups(s):
    '''
    Given a string, finds all the possible deduplications as a list of strings.
    This list is sorted by decreasing duplication size, which is equivalent
    to sorting by increasing deduplication string length.
    '''
    l = len(s)
    dedups = set()

    # Iterate through all possible deduplication lengths.
    for dup_length in range(1, int(l / 2) + 1):
        for i in range(l - 2 * dup_length + 1):
            first = s[i:i+dup_length]
            second = s[i+dup_length:i + 2 * dup_length]

            # If there is a duplication, generate the deduplication string.
            if first == second:
                new_s = s[:i] + s[i+dup_length:]
                dedups.add(new_s)

    return dedups

def _findSeed(s):
    '''
    Find the seed by comparing the first and last character.
    '''
    first = s[0]
    last = s[-1]

    if '1' not in s:
        return '0'
    elif '0' not in s:
        return '1'
    elif first == '0' and last == '1':
        return '01'
    elif first == '1' and last == '0':
        return '10'
    elif first == '0' and last == '0':
        return '010'
    elif first == '1' and last == '1':
        return '101'

def _solve(q, found, seed):
    '''
    Internal solve function.
    Given a queue, the worker gets tuples from the queue and generates all the
    deduplications from the string and increments the number of steps, then
    puts new tuples into the queue as long as the deduplicated string is not in
    the found set.
    '''
    while len(q) > 0:
        # Get new element.
        # The first element is the number of steps taken to get to this
        # particular string.
        # The second element is the string itself.
        step, s = q.popleft()

        # Find all deduplications and check if the seed is in it.
        # If it is, then the minimum duplication distance is just step + 1.
        dedups = _findDedups(s)
        if seed in dedups:
            return step + 1

        # Otherwise,check if the deduplication hasn't been found before
        # and add it to the queue.
        for new_s in dedups:
            # Add a new queue item as long as the deduplicated string isn't
            # one that we have already found in a previous step, so as to
            # reduce redundant computations.
            # Also, an important observation is that a string s, that has
            # already been found, is found in some later step, this 'path' of
            # deduplications will never be the shortest because calculation on
            # s has alredy started in a previous step.
            if new_s not in found:
                found.add(new_s)
                q.append((step + 1, new_s))

def solve(s):
    '''
    Solves for the minimum number of deduplications required to get to a
    seed string from the given string s.
    '''
    # First, find the seed we are looking for.
    seed = _findSeed(s)

    # Initialize the queue and add the string to the queue.
    # The queue contains tuples, the first element is the number of
    # deduplications taken to reach the second element.
    q = collections.deque()
    q.append((0, s))

    # Initialize the set of strings found.
    found = set()
    found.add(s)

    return _solve(q, found, seed)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("using this file requires an input string \n python solver.py 1010 \n will run the program to solve for string 1010")
    else:
        s = sys.argv[1]
        start_time = time.time()
        val = solve(s)
        total_time = time.time() - start_time
        print('solved for string {} in {} steps taking {:1.4f} seconds'.format(s, val, time.time() - start_time))
