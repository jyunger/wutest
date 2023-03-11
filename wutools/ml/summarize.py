"""Some tools for displaying or summarizing results of models"""

import graphviz

from wutools.ml import sequence


def apriori_to_graph(d_apriori):
    """depict the data from sequence.apriori as a graph"""
    g = graphviz.Digraph(graph_attr={'rankdir': 'LR'})
    for t, frac in d_apriori.items():
        g.node(str(t), ', '.join(t + (f'\n{frac:0.3f}',)))
    for i in d_apriori:
        for j in d_apriori:
            if i == j:
                continue
            if sequence.is_subsequence(i, j):
                # stupid n**3 algo
                for k in d_apriori:
                    if k != i and k != j and sequence.is_subsequence(i, k) and sequence.is_subsequence(k, j):
                        break
                else:
                    g.edge(str(i), str(j))
    return g
