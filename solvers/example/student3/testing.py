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
