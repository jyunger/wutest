"""Tools for mining sequences"""
import math
import re

import pandas

import digraph
import unclassified
from digraph import Digraph
from unclassified import get_logger

LOGGER = get_logger('ml.sequence')

def apriori(sequences, threshold, universe=None, value_as_frac=False, max_pattern_length=None):
    """Finds subsequences occurring more than threshold times in the set of sequences.

    A tuple x is a subsequence of y if x == tuple(y[i] for i in sorted(set(indices)) for some indices. That is,
    the elements of x appear in the elements of y in order, though not necessarily consecutive.

    Args:
        sequences: list of tuples
        threshold: int > 1 or float between 0 and 1. If int, the pattern must match more than threshold sequences
            If float, then pattern must match more than threshold * len(sequences).
        value_as_frac: bool. If False, returned dict is pattern -> list of sequences. If True, it is
            pattern -> fraction of sequences that match
        max_pattern_length: int or None. If int, the algorithm will not check for patterns longer than this.

    Returns:
        dict of tuple: key is a tuple describing a pattern, value is list of all sequences with key as subsequence
    """
    LOGGER.debug('start')
    if universe is None:
        universe = set()
        for s in sequences:
            universe = universe.union(set(s))
    LOGGER.debug(f"universe size {len(universe)}")

    if threshold < 1:
        threshold = len(sequences) * threshold
        as_pct = True
    else:
        as_pct = False
    LOGGER.debug(f"abs threshold == {threshold}")
    all_patterns = [
        dict([((p,), matches) for p, matches in ((p, [s for s in sequences if p in s]) for p in universe)
              if len(matches) >= threshold])
    ]
    LOGGER.debug(f"first level patterns: {len(all_patterns[0])}")
    def exclude(t, i):
        return tuple(x for j, x in enumerate(t) if j != i)
    def describe(patterns):
        s = pandas.Series({k: len(v) for k, v in patterns.items()}, name='num_matches').sort_values(ascending=False)
        msg = '\n\n'.join([
            s.describe().to_string(),
            s.head(5).to_string()
            ]
        )
        return msg
    while True:
        LOGGER.info(f"finding level {len(all_patterns)+1}")
        patterns = all_patterns[-1]  # get all the longest patterns
        LOGGER.debug(f"stats on number of matches at previous level:\n{describe(patterns)}")
        new_patterns = {}
        possible = set()
        LOGGER.debug(f"combining {len(patterns)} of len {len(all_patterns)} to form new patterns")
        for p in patterns:
            if max_pattern_length is not None and len(p) == max_pattern_length:
                continue
            completions = None
            overlapping_patterns = set(patterns)
            for i in range(len(p)):
                overlapping_patterns = set(q for q in overlapping_patterns if p[:i] == q[:i])
                q = exclude(p, i)
                if completions is None:
                    completions = {p2[-1] for p2 in overlapping_patterns if q == p2[:-1]}
                else:
                    completions.intersection_update({p2[-1] for p2 in overlapping_patterns if q == p2[:-1]})
                if len(completions) == 0:
                    break
            else:
                possible = possible.union({p+(c,) for c in completions})
        LOGGER.debug(f"found {len(possible)} new patterns to test")
        LOGGER.debug("testing new patterns")
        for p in possible:
            seqs = patterns[p[1:]]
            n = len(seqs)
            # find sub-pattern with the smallest number of matching sequences. We only need to check these
            for i in range(1, len(p)):
                seq2 = patterns[exclude(p, i)]
                m = len(seq2)
                if m < n:
                    seqs = seq2
                    n = m
            matches = [s for s in seqs if is_subsequence(p, s)]
            if len(matches) > threshold:
                new_patterns[p] = matches
        LOGGER.debug(f"found {len(new_patterns)} new patterns that meet threshold")
        if not new_patterns:
            break
        all_patterns.append(new_patterns)
    d = dict(sum([list(patterns.items()) for patterns in all_patterns], []))
    if value_as_frac:
        num_sequences = len(sequences)
        d = dict((k, len(v)/num_sequences) for k, v in d.items())
    return d


def prefix_span(sequences, threshold, value_type='frac', max_len=None):
    """Finds common subsequences and their support in a list of sequences, using a depth first search.

    Args:
        sequences: list of tuple
        threshold: int or float. If int, it represents the minimum number of matching sequences to include a pattern.
            If float, 0 <= threshold < 1, it is converted to an int = (len(sequences) * threshold)
        value_type: str. If 'frac', the returned support is as a fraction of the total number of sequences.
            If 'list', the support is the list of matching sequences
            If 'count', the support is the count of matching sequences

    Returns:
        dict pattern -> support where support is the fraction of sequences matching pattern if value_as_frac is True
            else support = list of matching sequences
    Notes:
        More efficient for long sequences than apriori
        May require more memory than apriori
    """
    assert(value_type in 'frac list count trie'.split())
    if threshold < 1:
        threshold = len(sequences) * threshold

    # database of ((seq_id, sequence), search_start_idx). That is, all the sequences we care about, and the first element we
    # haven't looked at yet
    fulldb = list(tuple(s) for s in sequences)

    all_patterns = dict()  # will map pattern -> (subdb, dict(extention -> ...)). This is a trie like structure
    stack = [([(i, 0) for i in range(len(fulldb))], all_patterns)]
    while stack:
        projdb, patterns = stack.pop()
        new_patterns = dict()
        for seq_id, idx in projdb:
            s = fulldb[seq_id]
            for j, e in enumerate(s[idx:]):
                subdb = new_patterns.setdefault(e, [])
                if len(subdb) == 0 or subdb[-1][0] != seq_id:
                    subdb.append((seq_id, idx+j+1))
        for e, subdb in new_patterns.items():
            if len(subdb) >= threshold:
                patterns[e] = (subdb, {})
                stack.append(patterns[e]) # make sure the reference to the dict is in the stack and patterns[e]

    # flatten the trie structure
    dg = Digraph(
        ((), [(i, 0) for i in range(len(fulldb))], all_patterns),  # prefix, matches, extensions
        lambda dg: (dg[0], dg[1]),  # the data at a node is the prefix and the matches for that prefix
        lambda dg: [(dg[0] + (k,), subdb, sup_patterns) for k, (subdb, sup_patterns) in dg[2].items()]
    )
    result = list(x.data for x in dg.breadthfirst_traverse(use_cache=False) if x.data[0] != ())
    if value_type == 'frac':
        num_sequences = len(sequences)
        return dict((p, len(v)/num_sequences) for p, v in result)
    elif value_type == 'list':
        return dict((p, list(fulldb[seq_id] for seq_id, _ in db)) for p, db in result)
    elif value_type == 'count':
        return dict((p, len(v)) for p, v in result)
    else:
        # should warn or something.
        return all_patterns


def is_subsequence(p, s):
    i = 0
    for x in p:
        try:
            i = s.index(x, i) + 1
        except ValueError:
            return False
    return True


def sequence_df(df, groupby, seq_col, order_col=None):
    if order_col is not None:
        df = df.sort_values(unclassified.box(groupby) + [order_col])
    return df.groupby(groupby)[seq_col].agg(lambda s: tuple(s))


def to_pattern(p):
    return tuple(x if isinstance(x, tuple) else (x, 1) for x in p)


def matches(p, s):
    """pattern matching on sequence

    Args
        p = List[(object, int)]. Each element is (x, 0|1). If 0, then assert not x is not found in s, 1 asserts x is found in s.
            So [('a', 1), ('b', 0), ('c', 1)] would match ('a', 'c') and ('a', 'f', 'c') but not ('a', 'b', 'c').
            If the last element is an exclude (x, 0), the exclusion goes to the end of the sequence.
            A singleton x could be used in place of (x, 1), for convenience
        s = tuple, the sequence to be matched
    """
    p = to_pattern(p)
    i = 0
    ub = len(s) + 1
    for x, k in p:
        if k == 1:
            try:
                i = s[:ub].index(x, i) + 1
                ub = len(s) + 1
            except ValueError:
                return False
        else:  # k == 0
            try:
                ub1 = s.index(x, i) + 1
            except ValueError:
                ub1 = len(s) + 1
            ub = min(ub, ub1)
    return ub == len(s) + 1


def sequence_alphabet(sequences):
    alphabet = set()
    for s in sequences:
        alphabet.update(set(s))
    return alphabet


class SequenceRE:

    def __init__(self, alphabet):
        self.alphabet = sorted(set(alphabet))
        self.num_chr = math.ceil(math.log(len(self.alphabet))/math.log(26))
        self.mapping = [
            (t, ''.join(chr(l % 26 + 97) for l in (i//26**d for d in range(self.num_chr)))) for i, t in enumerate(alphabet)
        ]
        self.encoder = dict(self.mapping)
        self.decoder = dict(reversed(p) for p in self.mapping)

    def encode_re(self, p):
        return re.sub("\<([\w\s]+)\>", lambda m: f"(?:{self.encoder.get(m.groups()[0], m.groups()[0])}_)", p)

    def encode_seq(self, s):
        return ''.join([self.encoder[t] + '_' for t in s])

    def decode_seq(self, s):
        return tuple(self.decoder.get(s[i * (self.num_chr+1):(i+1) * (self.num_chr + 1) - 1])
                     for i in range(len(s) // (self.num_chr+1))
                     )


def maximal_seq(seqs):
    for p1 in seqs:
        if not any(is_subsequence(p1, p) for p in d_apriori if p != p1):
            yield p1