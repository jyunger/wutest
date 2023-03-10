import pandas, re

def set_plotly_facet_axis_scale(plotly_fig, scale_x='shared', scale_y='shared'):
    """
    Args:
        scale_x in 'shared', 'free', 'by_col'
        scale_y in 'shared', 'free', 'by_row'

    Returns:
        same instance of plotly_fig, only modified.
    """
    layout = plotly_fig.layout
    ax = axes(layout)
    if set([scale_x, scale_y]).intersection(set('free by_col by_row'.split())):
        grid = deduce_facet_grid(plotly_fig)
    else:
        grid = None

    for axis_name, scale in zip('x y'.split(), [scale_x, scale_y]):
        if scale == 'shared':
            layout['{}{}'.format(axis_name, ax[0])].pop('matches', None)
            for axis in ax[1:]:
                layout['{}{}'.format(axis_name, axis)]['matches'] = '{}{}'.format(axis_name, ax[0][4:])
        elif scale == 'free':
            for axis in ax:
                a = layout['{}{}'.format(axis_name, axis)]
                a.pop('matches', None)
                a.pop('showticklabels', None)
        else:
            g = grid.T if axis_name == 'x' else grid
            for r in g:
                layout['{}{}'.format(axis_name, r[0])].pop('matches', None)
                for axis in r[1:]:
                    layout['{}{}'.format(axis_name, axis)]['matches'] = '{}{}'.format(axis_name, r[0][4:])
    return plotly_fig


def axes(layout):
    n = len([k for k in layout if re.match(r'[xy]axis\d*', k)])
    assert(n%2 == 0)
    return axis_range(n // 2)


def axis_range(n):
    return ['axis{}'.format(i if i > 1 else '') for i in range(1, n+1)]


def deduce_facet_grid(plotly_fig):
    # probably heavy handed (using pandas to unstack)
    return pandas.Series(axes(plotly_fig.layout), index=pandas.MultiIndex.from_tuples([(plotly_fig.layout['x'+a]['domain'][0], plotly_fig.layout['y'+a]['domain'][0]) for a in axes(plotly_fig.layout)], names=['x', 'y'])).unstack('x').values


def fix_plotly():
    import plotly.io as pio
    pio.renderers.default='notebook'