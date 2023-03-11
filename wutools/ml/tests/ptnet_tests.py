import unittest

from ml import petri


class MyTestCase(unittest.TestCase):
    def test_init(self):
        pt = petri.PTNet()
        self.assert_(True)

        pt = petri.PTNet('start', 'install', ('start', 'install'), [('start', 1)])
        self.assert_(True)

        self.assertRaisesRegex(ValueError, r'.*?((invalid arcs|invalid markings).*){2}', petri.PTNet, 'start', 'install', ('start', 'start'), [('start', -1)])

    def test_add(self):
        pt = petri.PTNet('start', 'install', ('start', 'install'), [('start', 1)])
        pt.place('end')
        pt.arc(('install', 'end'))
        self.assert_(('install', 'end') in pt.flow_relation)

        self.assertRaisesRegex(ValueError, r"{'invalid arcs': {\('start', 'emd'\)}}", pt.arc, ('start', 'emd'))

        pt.add_marking({'start': 1, 'end': 0, 'foo': -1})
        self.assertDictEqual(pt.marking, {'start': 2})

    def test_nodes(self):
        pt = petri.PTNet('start end'.split(), 'install', ('start', 'install'), [('start', 1)])
        self.assertSetEqual(pt.node_inputs('install'), set('start'.split()))
        self.assertSetEqual(pt.node_outputs('start'), set('install'.split()))
        self.assertSetEqual(pt.node_outputs('install'), set())
        pt.arc(('install', 'end'))
        self.assertSetEqual(pt.node_outputs('install'), set('end'.split()))

if __name__ == '__main__':
    unittest.main()
