import collections


class Digraph:
    def __init__(self, top, data_fn, children_fn, typename=None):
        self.top = top
        self.data_fn = data_fn
        self.children_fn = children_fn
        self.typename = typename

    @property
    def data(self):
        return self.data_fn(self.top)

    @property
    def children(self):
        return tuple(Digraph(c, self.data_fn, self.children_fn, self.typename) for c in self.children_fn(self.top))

    def breadthfirst_traverse(self, use_cache=True):  # this is really breadth first
        visited = set()  # visited will prevent dups
        stack = [self]
        while stack:
            this = stack.pop(0)
            if not use_cache or this.top not in visited:
                yield this
                if use_cache:
                    visited.add(this.top)
                stack.extend(list(c for c in this.children))

    def depthfirst_traverse(self, order='pre'):
        stack = collections.deque([(self, 'descending')])
        while stack:
            this, direction = stack.pop()
            if direction == 'descending':
                if order == 'pre':
                    stack.extend(reversed([(c, 'descending') for c in this.children]))
                    stack.append((this, 'ascending'))
                else:
                    stack.append((this, 'ascending'))
                    stack.extend(reversed([(c, 'descending') for c in this.children]))
            else:
                yield this

    def prune(self, tree_predicate):
        """

        Args:
            tree_predicate: fn(Digraph)->bool. Is applied to children.

        Returns:

        """
        def pruned_children(t):
            return tuple(c for c in self.children_fn(t) if
                         tree_predicate(Digraph(c, self.data_fn, self.children_fn, self.typename)))

        return Digraph(self.top, self.data_fn, pruned_children, self.typename)

    NODE_NOT_VISITED = object()
    EMPTY_CACHE = object()

    def accumulate(self, fn, cache=EMPTY_CACHE):
        """recursively apply fn. signature of fn:
            fn(Digraph, tuple(whatever fn returns))
        where the second arg is the result of accumulate applied to children

        Does not return a Digraph"""
        if cache is not None:
            if cache is self.EMPTY_CACHE:
                cache = {}
            value = cache.get(self.top, self.NODE_NOT_VISITED)
        else:
            value = self.NODE_NOT_VISITED
        if value is self.NODE_NOT_VISITED:
            return fn(self, tuple(c.accumulate(fn, cache=cache) for c in self.children))
        else:
            return value

    def transform(self, top_fn, derived_data_fn, derived_children_fn, typename=None):
        """derived_data_fn(old_data_fn, old_children_fn, old_typename) returns a new data_fn.
        Similar for derived_children_fn

        transform is like putting a decorator on self.data_fn and self.children_fn"""
        return Digraph(
            top_fn(self.top, self.data_fn, self.children_fn),
            derived_data_fn(self.data_fn, self.children_fn),
            derived_children_fn(self.data_fn, self.children_fn),
            typename=typename
        )


def to_nested_tuples(dg):
    return Digraph(
        dg.accumulate(lambda dg, ch: (dg.data , ch), cache=None),
        lambda p: p[0],
        lambda p: p[1]
    )
