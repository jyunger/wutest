from numpy import random
import unittest

import unclassified
from ml.sequence import prefix_span

class MyTestCase(unittest.TestCase):
    def test_prefix_span(self):
        sequences = 'cat cow dog sheep wolf fox chickens rooster chicken chick woodchuck woodpecker lion mouse grouse'.split()
        d = prefix_span(sequences, 2, value_type='count')
        self.assertEqual(d[tuple('ch')], 4)
        self.assertEqual(d[tuple('wood')], 2)
        self.assertNotIn(tuple('chuck'), d)

        d = prefix_span(sequences, 2, value_type='list')
        self.assertListEqual(d[tuple('wood')], [tuple('woodchuck'), tuple('woodpecker')])

    def test_prefix_span_speed(self):
        random.seed(seed=1)
        alphabet = list(range(120))
        sequences = [random.choice(alphabet, size=random.binomial(100, 0.75, size=1)) for _ in range(20000)]
        with unclassified.ContextTimer('start') as t:
            d = prefix_span(sequences, 0.5)
        self.assertLessEqual(t.interval, 1, msg="prefix_span is taking too long")

if __name__ == '__main__':
    unittest.main()
