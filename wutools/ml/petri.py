"""An implementation of a Petri net
Mostly just to enhance understanding of Workflow Mining--van der Aalst, Weijters, Maruster
https://drive.google.com/file/d/1EmlYAndiCedmajZIjufkc4-ExK4Cj9pf/view?usp=sharing
"""
import itertools

import unclassified


class PTNet:

    def __init__(self, places=None, transitions=None, flow_relation=None, marking=None):
        """

        Args:
            places: a set of labels for places
            transitions: a set of labels for transitions
            flow_relation: a set flow_relation, edges from (places x transition).union(transition x places)
            marking:
        """
        self.places = unclassified.box(places, typ=set, none=unclassified.EMPTY)
        self.transitions = unclassified.box(transitions, typ=set, none=unclassified.EMPTY)
        self.flow_relation = unclassified.box(flow_relation, typ=set, none=unclassified.EMPTY, atomic=(tuple, str))
        if marking is None:
            self.marking = {}
        else:
            self.marking = dict(marking)

        v = self._violations()
        if v:
            raise ValueError("invalid specification of PTNet", v)

    def place(self, place):
        """Add a new place

        Args:
            place:

        Returns:
            self
        """
        self.places.add(place)
        return self

    def transition(self, transition):
        """Add a new transition

        Args:
            transition:

        Returns:
            self

        """
        self.transitions.add(transition)
        return self

    def arc(self, an_arc):
        fm, to = an_arc
        if (fm in self.places and to in self.transitions) or (fm in self.transitions and to in self.places):
            self.flow_relation.add(an_arc)
        else:
            self.flow_relation.add(an_arc)
            raise ValueError(self._violations())

    def add_marking(self, marking):
        """Add a marking

        Args:
            marking: dict

        Returns:
            self
        """

        for p in self.places:
            v = marking.pop(p, 0)
            if v > 0:
                self.marking[p] = self.marking.get(p, 0) + v
            elif v < 0:
                raise ValueError(f"Marking for {p!r} is {v} < 0")
        return self

    def node_inputs(self, node):
        return set(x for x,y in self.flow_relation if y == node)

    def node_outputs(self, node):
        return set(y for x,y in self.flow_relation if x == node)

    def _violations(self):
        violations = {}
        # places disjoint from transitions
        overlap = self.places.intersection(self.transitions)
        if overlap:
            violations['place-transition overlap'] = overlap

        # check that arcs connect places to transitions or transitions to places
        arc_space = set.union(
            set(itertools.product(self.places, self.transitions)),
            set(itertools.product(self.transitions, self.places))
        )
        invalid_arcs = self.flow_relation.difference(arc_space)
        if invalid_arcs:
            violations['invalid arcs'] = invalid_arcs

        # check that every marking is >= 0
        invalid_marks = [(p, v) for p, v in ((p, self.marking.get(p, 0)) for p in self.places) if not v >= 0]
        if invalid_marks:
            violations['invalid markings'] = invalid_marks

        return violations
