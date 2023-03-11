import unittest


from ml.dict_encoder import items_flat, join_key


class MyTestCase(unittest.TestCase):
    def test_items_flat(self):
        d = {'a': 1, 'b': {'x': 'foo'}}
        self.assertListEqual([(('a',), 1), (('b', 'x'), 'foo')], sorted(items_flat(d)))
        d = {'a': 1, 'b': {'x': 'foo'}, 1: 'bar'}
        self.assertListEqual([('1', 'bar'), ('a', 1), ('b.x', 'foo')], sorted(join_key(items_flat(d), '.')))

if __name__ == '__main__':
    unittest.main()
