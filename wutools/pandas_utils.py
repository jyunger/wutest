"""Utilities for working with pandas dataframes"""
import numpy
import pandas, re

import tools as unclassified
from tools import box


def df_to_ns(df, globalz={}):
    return dict(globalz, **{c: df[c] for c in df})


def eval_col(df, formula, globalz={}):
    """evaluates a formula in context of df and returns a series indexed by df.index"""
    ns = df_to_ns(df, globalz)
    return eval(formula, ns)


def eval_cols(df, formulas, globalz=None):
    """returns a dataframe of evaluated formulas
    formulas is ; separated str or a list of str"""
    if globalz is None: globalz = {}
    formulas = split_formulas(formulas)

    def name_value(f, i):
        r = re.compile(r'^(?:\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*(?:=\s*([^=].*))?\s*|(.+))$')
        m = r.match(f)
        if not m:
            raise SyntaxError(f"Failed to parse {f!r} using {r!r}")
        # m.groups() is one of
        # ('name', None, None)  "my_col"
        # ('name', 'value', None)  "my_col = other_col*2"
        # or (None,  None, 'expr')  "my_col + 1"
        name, value, expr = m.groups()
        if name is None:
            name = f"expr_{i:02d}"
            value = expr
        if value is None:
            value = name
        return name, value

    named_formulas = [name_value(f, i) for i, f in enumerate(formulas)]
    ns = df_to_ns(df, globalz)

    def eval_update(ns, n, f):
        try:
            s = eval(f, ns)
        except:
            raise Exception("Could not evaluate {!r} in {}".format(f, formulas))
        if not isinstance(s, pandas.Series):
            s = pandas.Series(s, index=df.index)
        ns[n] = s
    [eval_update(ns, n, f) for n, f in named_formulas]
    try:
        # this order puts the columns in order in which they appear last in the formulas
        order = dict([(n, i) for i, (n, _) in enumerate(named_formulas)])
        cols = sorted(order, key=lambda k: order[k])
        return pandas.DataFrame({n: ns[n] for n in cols}, columns=cols)
    except:
        raise Exception(
            "Failed to construct df from {}".format('\n'.join(['{}: {}'.format(n, f) for n, f in named_formulas])))


def split_formulas(formulas):
    if isinstance(formulas, str):
        formulas = [f for f in (f.strip() for f in formulas.split(';')) if f]
    return formulas


def assign(df, formulas=None, globalz={}):
    if formulas is None:
        return df
    df_new = eval_cols(df, split_formulas(formulas), globalz=globalz)
    return df.assign(**{c: df_new[c] for c in df_new.columns})


def suggest_categoricals(df, max_distinct=10):
    """make a dataframe of only those columns with a limited number of distinct values"""
    candidates = [c for c in df if len(df[c].drop_duplicates()) <= max_distinct and df[c].dtype.name != 'category']
    return pandas.DataFrame([pandas.Categorical(df[c]) for c in candidates], columns=candidates)


def add_rank(df, groupby, col='rank'):
    return df.assign(**{col: df.assign(**{col: 1}).groupby(groupby)[col].cumsum()})


def append_group_value(df, groupby, agg, agg_cols, agg_formulas=None, names=None, globalz=None, **groupby_kw):
    groupby = box(groupby)
    dfg = group_by(df, groupby, agg, agg_cols, names, globalz=globalz, **groupby_kw)
    if agg_formulas:
        dfg = assign(dfg, agg_formulas)
    idx = df.index
    df = pandas.merge(df, dfg.reset_index(), how='left', on=groupby)
    df.index = idx  # the merge loses the index of df, but by construction the index could be preserved exactly.
    return df


def group_by(df, groupby, agg, agg_cols, names=None, globalz=None, **groupby_kw):
    groupby = box(groupby)
    agg = box(agg)
    agg_cols = box(agg_cols, none= [c for c in df if c not in groupby])
    df_agg_cols = eval_cols(df, ';'.join(agg_cols), globalz=globalz)
    agg_cols_names = list(df_agg_cols)  # use the eval_cols logic to name the columns in case agg_cols is expr
    dfg = pandas.concat([df[groupby], df_agg_cols], axis=1, sort=False).groupby(groupby, **groupby_kw)[agg_cols_names].agg(agg)
    if len(agg) == 1:
        default_names = list(c[0] for c in dfg)
    else:
        default_names = list('_'.join(str(l) for l in box(c)) for c in dfg)
    names = box(names, none=default_names)
    if len(names) != len(dfg.columns):
        raise ValueError(f"names (len={len(names)}) should have same length as "
                         f"len(agg_cols)*len(agg)(={len(agg_cols)}*{len(agg)}={len(agg_cols) * len(agg)}")
    dfg.columns = names
    return dfg


def group_query(df, groupby, agg, query, **groupby_kw):
    groupby = unclassified.box(groupby)
    dfx = group_by(df, groupby, agg, agg_cols=None, **groupby_kw).reset_index()
    try:
        dfx = dfx.query(query)
    except Exception:
        raise ValueError(f"could not evaluate query {query!r}\non columns {list(dfx)}")
    dfx = dfx[[c for c in dfx if c in groupby]]
    df = pandas.merge(df, dfx, how='inner', on=groupby)
    return df


def drop_safe(df, drop_if):
    if not callable(drop_if):
        if isinstance(drop_if, str):
            drop_if = [drop_if]
        if hasattr(drop_if, '__iter__'):
            drop_set = set(drop_if)
            def drop_if(c): return c in drop_set
        else:
            raise TypeError(f"Can't figure out what to do with type {type(drop_if).__name__!r}")
    return df[[c for c in df.columns if not drop_if(c)]]


def keep_safe(df, keep_if, sort_key=None):
    if not callable(keep_if):
        if isinstance(keep_if, str):
            keep_if = [keep_if]
        if hasattr(keep_if, '__iter__'):
            keep_set = set(keep_if)
            def keep_if(c): return c in keep_set
        else:
            raise TypeError(f"Can't figure out what to do with type {type(keep_if).__name__!r}")
    cols = [c for c in df.columns if keep_if(c)]
    if sort_key:
        cols = sorted(cols, key=sort_key)
    return df[cols]


def move_to_front(df, cols):
    cols = unclassified.box(cols, none=unclassified.EMPTY)
    return df[[c for c in cols if c in set(df.columns)] + [c for c in df if c not in set(cols)]]


def expand_dict_col(df, col, prefix=None, replace_col=False):
    if col not in df:
        return df.copy()
    prefix = f'{col}_' if prefix is None else prefix
    dfc = pandas.DataFrame(list(df[col]), index=df.index)
    dfc = dfc.rename(columns={c: f'{prefix}{c}' for c in dfc})
    if replace_col:
        dfc = pandas.concat([df.drop(col, axis=1), dfc], axis=1, sort=False)
    return dfc


def distinct_row_count(df, cols=None, count_name='count', as_index=False):
    cols = unclassified.box(cols, none=list(df))
    df = df[cols]
    df = df.assign(__one__=1).groupby(cols, as_index=as_index)['__one__'].count()
    if as_index:
        df = df.to_frame(count_name)
    return df.rename(columns={'__one__': count_name})


def frac_total(s):
    return s / s.sum()


def append_frac_total(df, col, groupby=None, name=None):
    col = box(col)
    if name is None:
        name = [f"frac_{c}" for c in col]
    if groupby is None:
        frac = lambda c, df: frac_total(df[c])
    else:
        frac = lambda c, df: df.groupby(groupby)[c].transform(frac_total)
    return df.assign(**{n: frac(c, df) for n, c in zip(name, col)})


def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val)
def disp_click(df):
    return (df.style.format({c: make_clickable for c in df if re.match(r'^(.*[_\-])?(url|link|href)([_\-].*)?$', c)}))


def pf(df, title=None, **to_string_kwargs):
    if title is not None:
        print(title)
    print(df.to_string(**to_string_kwargs))
    print()


def fill_fwd(fill_vals, fill_from, null_value=numpy.nan):
    """
    fill_vals is a series or df.
    fill_from is a series with 1/True for each row to fill forward, 0/False for rows to be overwritten
    null_value is the value to use before the first fill_from location. Usually numpy.nan works, but for datetimes,
        you'll want to use np.datetime64("NaT")

    fill_vals and fill_from need to have same index"""
    values = fill_vals.values[fill_from.astype('bool')]
    idx = fill_from.cumsum().values - 1
    fill_vals = fill_vals.copy()
    fill_vals.update(pandas.Series(numpy.where(idx >= 0, values[idx], null_value), index=fill_vals.index))
    return fill_vals


def not_null(df, cols, how='all'):
    cols = unclassified.box(cols, none=None)
    if not cols:
        return df
    fn = getattr(pandas.DataFrame, how)
    return df[fn(~df[cols].isnull(), axis=1)]


def fix_dates(df, cols=None):
    """Convert cols to utc datetimes (naive)

    Args:
        df: DataFrame
        cols: None or list or scalar. The cols of df to convert to datetime64
            If None, convert all cols. Otherwise, cols will be boxed and only those cols will be converted.

    Returns:
        copy of df with cols converted to datetime64, converted to UTC and then made naive
    """
    df = df.copy()
    if cols is None:
        cols = list(df)
    else:
        cols = unclassified.box(cols)
    for c in cols:
        df[c] = pandas.to_datetime(df[c], utc=True).dt.tz_localize(None)
    return df
