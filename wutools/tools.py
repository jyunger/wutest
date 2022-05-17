"""A collection of functions with no theme.  Mostly small utilities."""
import datetime
import gc
import gzip
import itertools
import logging
import math
import os
import pickle
import shelve
import shutil
import sys
import time
from collections import OrderedDict

import numpy
import pandas
import psutil

EMPTY = '__empty__'
BOXED_NONE = '__boxed_none__'


def box(x, typ=list, none=None, atomic=(str,)):
    """Wraps an object in a list (or other iterable type) or returns it unchanged if it is already of the correct type

    This is useful for keeping code simple looking by allowing interfaces to take lists or scalars.  E.g.
        def foo(x):
            for y in box(x):
                print(y)
    Then foo([1]) and foo(1) both print "1". Not having to type the [] doesn't seem like much, but it does really
    add up over time.

    Args:
        x: the object to be "boxed"
        typ: fn that takes an iterable and returns the iterable as a given type. Think list or tuple as functions.
        none: anything. This is the value that is returned if x is None. Except if none == EMPTY or none == BOXED_NONE.
            If EMPTY, then return typ(). If BOXED_NONE, return typ([None])
        atomic: an iterable of types that are considered "atomic" even though they are iterable. The default is str so
            that box('hey') is ['hey'] and not ['h', 'e', 'y']. Pass an empty list if you want the latter behavior.

    Returns:
        typ(x) if x is iterable and not atomic
        typ([x]) if x is not None and not iterable or atomic
        None, typ([]), or typ([None]) if x is None

    >>> box(1)
    [1]
    >>> box([1])
    [1]
    >>> box(tuple([1]))
    [1]
    >>> box(None) is None
    True
    >>> box(None, none=EMPTY)
    []
    >>> box(None, none=BOXED_NONE)
    [None]
    >>> box(None, typ=tuple, none=BOXED_NONE)
    (None,)
    """
    if x is None:
        if none == EMPTY:
            return typ([])
        elif none != BOXED_NONE:
            return none

    if not isinstance(x, tuple(atomic)) and hasattr(x, '__iter__'):
        return typ(x)
    return typ([x])


def as_date(x):
    """Converts object to a datetime

    Args:
        x: datetime.date, datetime.datetime, str, int representation of a date. If a str, then 'yyyy' is interpreted
            as yyyy-01-01, and yyyymm and yyyy-mm are interpreted as yyyy-mm-01. After that processing,
            pandas.to_datetime must recognize the format.

    Returns:
        python datetime.datetime.

    >>> print(as_date(datetime.date(2019, 6, 1)))
    2019-06-01 00:00:00
    >>> print(as_date('20190601'))
    2019-06-01 00:00:00
    >>> print(as_date(20190601))
    2019-06-01 00:00:00
    >>> print(as_date(2019))
    2019-01-01 00:00:00
    >>> print(as_date(201906))
    2019-06-01 00:00:00
    >>> print(as_date('2019'))
    2019-01-01 00:00:00
    >>> print(as_date('201906'))
    2019-06-01 00:00:00
    >>> print(as_date('2019-6'))
    2019-06-01 00:00:00
    >>> print(as_date('2019-06'))
    2019-06-01 00:00:00

    """
    if isinstance(x, datetime.date) or isinstance(x, datetime.datetime):
        return datetime.datetime(*x.timetuple()[:6])
    elif isinstance(x, str):
        if len(x) == 4:  # yyyy
            x = x + '0101'
        elif len(x) == 6 and '-' not in x:  # yyyymm
            x = x + '01'
        elif len(x.split('-')) == 2:  # yyyy-m
            x = x + '-01'
        return pandas.to_datetime(x).to_pydatetime()
    elif isinstance(x, int):
        if x < 10000:  # yyyy
            x = x * 10000 + 101
        elif x < 1000000:  # yyyymm
            x = x * 100 + 1
        return datetime.datetime(x // 10000, (x // 100) % 100, x % 100)
    elif x is None:
        return None
    else:
        raise TypeError("Unrecognized type {!r} for conversion to date".format(type(x).__name__))


def split_by_month(from_dt, upto_dt):
    """a generator of months from from_dt, up to but not including upto_dt

    Args:
        from_dt, upto_dt: datelike (can be cast to datetime by as_date). The dates must be first of the month and
            from_dt < upto_dt

    Returns:
        a generator that iterates through the months from from_dt to upto_dt.

    Notes:
        the end date is exclusive (ie not yielded by the generator).
    """
    from_dt = as_date(from_dt)
    upto_dt = as_date(upto_dt)
    if from_dt >= upto_dt:
        raise ValueError("from_dt {} must be less than upto_dt {}".format(from_dt, upto_dt))
    if from_dt.day != 1 or upto_dt.day != 1:
        raise ValueError("from_dt {} and upto_dt {} must be beginning of the month".format(from_dt, upto_dt))
    start = from_dt.year * 12 + from_dt.month - 1
    end = upto_dt.year * 12 + upto_dt.month - 1

    def ord_to_dt(n): return as_date('{}{:02d}'.format(n // 12, n % 12 + 1))

    def gen():
        dt1 = from_dt
        for n in range(start+1, end+1):
            dt2 = ord_to_dt(n)
            yield dt1, dt2
            dt1 = dt2
    return list(gen())


def date_iterator(s, e=None, step=1):
    """Generator yielding all dates from s up to but not including e.

    Args:
        s: datelike (see as_date)
        e: datelike or None. If None, then iterator only yields a single date: as_date(s)
        step: int, number of days between yieldd dates. E.g. step=1 gives every day, step=7 gives every week.

    Returns:
        generator yielding datetimes from s upto e, with successive dates having timedelta(step)

    >>> list(date_iterator(20200101, 20200103))
    [datetime.datetime(2020, 1, 1, 0, 0), datetime.datetime(2020, 1, 2, 0, 0)]

    >>> list(date_iterator(20200101, 20200115, 7))
    [datetime.datetime(2020, 1, 1, 0, 0), datetime.datetime(2020, 1, 8, 0, 0)]
    """
    step = datetime.timedelta(step)
    dt = as_date(s)
    e = dt + step if e is None else as_date(e)
    while dt < e:
        yield dt
        dt = dt + step


def suggest_categoricals(df, max_distinct=10, frac_distinct=None):
    """make a dataframe of only those columns with a limited number of distinct values"""
    if max_distinct is None:
        max_distinct = len(df) * frac_distinct

    def is_categorical(s):
        if s.dtype.name in {'category', 'int', 'int64', 'float', 'float64'}:
            return False
        try:
            # drop_duplicates is slow, so check on a small set first
            return (len(s.head(math.ceil(max_distinct * 2)).drop_duplicates()) <= max_distinct and
                    len(s.drop_duplicates()) <= max_distinct)
        except SystemError:
            return False
    # candidates = [c for c in df if len(df[c].drop_duplicates()) <= max_distinct and df[c].dtype.name not in {'category', 'int', 'int64', 'float', 'float64'}]
    # return pandas.DataFrame([pandas.Categorical(df[c]) for c in candidates], columns=candidates)
    df = df.copy()
    for c in df:
        if is_categorical(df[c]):
            df[c] = pandas.Categorical(df[c])
    return df

def as_date_range(start, end):
    start = as_date(start)
    end = as_date(end) if end else start + datetime.timedelta(1)
    return start, end


def pandas_to_datetime(df, cols):
    return pandas.DataFrame({c: pandas.to_datetime(df[c]) if c in box(cols) else df[c] for c in df.columns}, columns=df.columns)

def ngrams(l, n, blank='<blank>'):
    x = pad(l, n, blank)
    return list(zip(*[x[i:] for i in range(n)]))


def pad(l, n, blank):
    x = [blank] * (n - 1) + l + [blank] * (n - 1)
    return x


def list_join(sep, iterable):
    sep = box(sep, none=[None])
    iterable = iter(iterable)
    r = next(iterable)
    for x in iterable:
        r.extend(sep)
        r.extend(x)
    return r


def first_event_of_type(df, event_type):
    event_type = box(event_type, typ=set, none=EMPTY)
    return df[df.event_type.isin(event_type)].groupby('distinct_id').head(1)


def append_time_of_first_event(df, event_type, time_col=None):
    if time_col is None:
        time_col = 'time_first_{}'.format(event_type.lower().replace(' ', '_').replace('$', '_mp_'))
    dfe = first_event_of_type(df.sort_values('time'), event_type).groupby('distinct_id').head()['distinct_id time'.split()].rename(columns={'time': time_col})
    return pandas.merge(df, dfe, how='left', on='distinct_id')


def ngrams_to_df(ng, col_root='x', count_root='cnt_', prob_root='cond_prob_', ngram='ngram', entropy_root='entropy_'):
    df = pandas.DataFrame(ng)
    n = len(df.columns)
    col_names = list("{}{:d}".format(col_root, i) for i in range(n))
    df.columns = col_names
    df = df.assign(**{'{}{}'.format(count_root, ngram): 1}).groupby(list(df.columns), as_index=False).count()

    def entropy_df(df, c):
        s = df['{}{}'.format(prob_root, c)]
        entropy_col = '{}{}'.format(entropy_root, c)
        return df.assign(**{entropy_col: -s * numpy.log2(s)})[[c, entropy_col]].groupby(c, as_index=False).sum()

    for c in col_names:
        cnt_col = '{}{}'.format(count_root, c)
        df = pandas.merge(df, df.assign(**{cnt_col: df['{}{}'.format(count_root, ngram)]})[[c, cnt_col]].groupby(c,
                                                                                                                 as_index=False).sum(),
                          how='left', on=c)
        df['{}{}'.format(prob_root, c)] = df['{}{}'.format(count_root, ngram)] / df['{}{}'.format(count_root, c)]
        df = pandas.merge(df, entropy_df(df, c), how='left', on=c)

    return df


def sequence_entropies(df):
    elist = [t for idx in df.groupby('distinct_id').groups.values() for t in
             ['<BEGIN>'] + list(df.loc[list(idx)].event_type) + ['<END>']]
    digrams = [p for p in ngrams(elist, 2) if p != ('<END>', '<BEGIN>') and '<blank>' not in p]
    return ngrams_to_df(digrams)


class ContextTimer:
    def __init__(self, name=None, logger=None, verbose=False):
        self.name = name
        if logger is None and verbose:
            logger = logging.getLogger()
        self.logger = logger

    def __enter__(self):
        if self.logger:
            self.logger.info("start {!r}".format(self.name))
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        if self.logger:
            self.logger.info(self)

    def __str__(self):
        return "{}: {}".format(self.name, self.interval)


class LogMemory(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """
    def filter(self, record):
        record.mem = f"{get_mem_usage():0.2f}"
        return True

def remove_handlers(l):
    for h in list(l.handlers):
        l.removeHandler(h)
    return l


def get_logger(name=None, level=None):
    """get a logger for waking_up namespace. Use name=None for top level"""
    if name is None:
        name = 'waking_up'
    else:
        name = f"waking_up.{name}"
    l = remove_handlers(logging.getLogger(name))
    if name == 'waking_up':
        f = LogMemory()
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(mem)sGB - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
#         l = remove_handlers(l)
        l.addHandler(ch)
        for f in list(l.filters):
            l.removeFilter(f)
        l.addFilter(f)
    if level is not None:
        l.setLevel(level)

    if len(list(l.filters)) == 0:
        l.addFilter(LogMemory())

    return l


class Store:
    def __init__(self, dbfile):
        self.dbfile = dbfile
        d, f = os.path.split(dbfile)
        os.makedirs(d, exist_ok=True)

    def __enter__(self):
        return shelve.open(self.dbfile).__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __getitem__(self, k):
        with self as d:
            return d[k]

    def __setitem__(self, k, v):
        with self as d:
            d[k] = v

    def __contains__(self, k):
        with self as d:
            return k in d

    def __delitem__(self, k):
        with self as d:
            del d[k]

    def __iter__(self):
        with self as d:
            return d.keys().__iter__()

    def items(self):
        with self as d:
            for k, v in d.items():
                yield k, v

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default


def binary_probs(df, col, col_display=None):
    col_display = col if col_display is None else col_display
    x = df.assign(**{c: df[c].fillna(False) if df[c].dtype == 'bool' else ~df[c].isnull() for c in df})
    p_attr = x.mean()
    pand = (x.T & x[col]).T.mean()
    pnand = (~x.T & x[col]).T.mean()
    p_col = p_attr[col]
    p_attr_col = x.groupby(col).mean().T
    p_col_attr = pand / p_attr
    p_col_nattr = pnand / (1-p_attr)
    x = pandas.DataFrame(
        dict(zip([c.format(col_display=col_display) for c in ['attr', 'attr|{col_display}', 'attr|~{col_display}',
                                                 '{col_display}|attr', '{col_display}|~attr']
        ],
                 [p_attr, p_attr_col[True], p_attr_col[False], p_col_attr, p_col_nattr]
                )
            )
    ).assign(**{col_display: p_col})
    x.index.name = 'attr'
    return x


def pickle_load(path, safe=True):
    if not safe and not os.path.exists(path):
        return None
    with (gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb')) as fh:
        return pickle.load(fh)


def pickle_save(data, path):
    if path.endswith('.gz'):
        compress = True
        path = path[:-3]
    else:
        compress = False

    with open(path, 'wb') as fh:
        pickle.dump(data, fh)
    if compress:
        return compress_file(path)
    else:
        return path


def compress_file(path):
    with open(path, 'rb') as fsrc, gzip.    open(path + '.gz', 'wb') as fdest:
        shutil.copyfileobj(fsrc, fdest)
    os.remove(path)
    return path + '.gz'


def chunk(it, n):
    """yields chunks of iterator it of size n"""
    it = iter(it)
    piece = list(itertools.islice(it, n))
    while piece:
        yield piece
        piece = list(itertools.islice(it, n))


def set_digits(d=2, nan_rep='', padding=None, left_padding=None, right_padding=None):
    if padding is None: padding = ''
    if left_padding is None: left_padding = padding
    if right_padding is None: right_padding = padding
    fmt = f"{left_padding}{{:0.{d}f}}{right_padding}".format

    def digits(x):
        return fmt(x) if x==x else nan_rep
    return digits


def str_dotter(n):
    def formatter(s):
        if len(s) > n:
            s = s[:n//2-1] + ('...' if n%2 == 1 else '..') + s[-(n//2-1):]
        return '{{:>{0}.{0}s}}'.format(n).format(s)
    return formatter


def as_predicate(iter_or_callable, inclusion=True):
    """Casts an interable to a predicate by check for inclusion or exclusion.

    Args:
        iter_or_callable: if callable, the arg is returned directly. Otherwise, it is assumed to be an iterable of
            values you want to test against.
        inclusion: True means lambda x: x in iter_or_callable, False means lambda x: x not in iter_or_callable
            If iter_or_callable is callable, this arg has no effect.

    Returns:
        callable that returns bool
    """
    if callable(iter_or_callable):
        return iter_or_callable
    values = box(iter_or_callable, typ=set, none={None})
    def pred(x):
        return x in values if inclusion else (x not in values)
    return pred


### some memory management tools
def _getr(slist, olist, seen):
    for e in slist:
        if id(e) in seen:
            continue
        seen[id(e)] = None
        olist.append(e)
        tl = gc.get_referents(e)
        if tl:
            _getr(tl, olist, seen)


def get_all_objects():
    """Return a list of all live Python
    objects, not including the list itself."""
    gcl = gc.get_objects()
    olist = []
    seen = {}
    # Just in case:
    seen[id(gcl)] = None
    seen[id(olist)] = None
    seen[id(seen)] = None
    # _getr does the real work.
    _getr(gcl, olist, seen)
    return olist


def get_mem_usage(display=False):
    process = psutil.Process(os.getpid())
    x = process.memory_info().rss / 1e9
    if display:
        print(f"mem usage {x:0.3f}G")
    else:
        return x


def better_size(obj, seen=None):
    """looks inside of some containers to get full memory footprint.
    DataFrames are a bit tricky because there is a mgr which shares data
    across dataframes (e.g. if a column is the same, the data is not copied).
    So, this will probably overstate the memory footprint of DFs"""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, (dict, list, tuple, pandas.DataFrame)):
        size = size + sum([better_size(r, seen) for r in gc.get_referents(obj)])
    # print(type(obj), size)
    return size
####################################

def as_projection(proj, exclude_id=True, merge_dict=None):
    """
    Convenience function to create a value for pymongo projection. Mostly, this is useful for the exclude_id arg,
    allowing you to specify a list and conveniently exclude the _id property.

    Args:
        proj: dict (usual projection dict) or list of document properties to include.
        exclude_id: bool. If True, the projection will have as_projection(proj, True)['_id'] == False. Otherwise,
            it will be whatever the dict is.
        merge_dict: dict or None. If not None, merge_dict is a projection dict will be merged into proj. Allows for
            more complicated projections to be constructed in one line. This arg is equivalent to
                dict(as_projection(proj), **merge_dict)
            Note that merge_dict overwrites proj
    Returns:

    """
    if not isinstance(proj, dict):
        proj = {k: 1 for k in box(proj)}
    else:
        proj = dict(proj)  # copy
    proj['_id'] = not exclude_id
    if merge_dict is not None:
        proj = dict(proj, **merge_dict)
    return proj


class LRUCache:
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def __getitem__(self, k):
        v, t = self.cache.pop(k)
        self.cache[k] = (v, datetime.datetime.now())
        return v

    def get(self, k, d):
        try:
            return self.cache[k][0]
        except KeyError:
            return d

    def __setitem__(self, k, v):
        self.cache[k] = (v, datetime.datetime.now())
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear_older_than(self, age_limit):
        dt = datetime.datetime.now() - age_limit
        self.cache = OrderedDict([(k, v) for k, v in self.cache.items() if v[1] > dt])


def no_peek_product(*args):
    """generate a product of iterators without peeking ahead.
    Note, however, that all te values have to end up in memory because the generator is consumed"""
    args, gens = list(zip(*[itertools.tee(a) for a in args]))
    if len(args) > 1:
        for x in gens[0]:
            for v in no_peek_product(*list(zip(*[itertools.tee(a) for a in args[1:]]))[1]):
                yield (x,) + v
    else:
        for x in itertools.tee(args[0])[1]:
            yield (x,)
