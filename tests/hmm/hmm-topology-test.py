import random
import unittest

from kaldi.base.io import istringstream, ostringstream
from kaldi.hmm import HmmTopology

def get_default_topology(phones):
    phones = sorted(phones)

    topo_string = """<Topology>
    <TopologyEntry>
    <ForPhones>
    """
    for i in phones:
        topo_string += str(i) + " "

    topo_string += """</ForPhones>
    <State> 0 <PdfClass> 0
    <Transition> 0 0.5
    <Transition> 1 0.5
    </State>
    <State> 1 <PdfClass> 1
    <Transition> 1 0.5
    <Transition> 2 0.5
    </State>
    <State> 2 <PdfClass> 2
    <Transition> 2 0.5
    <Transition> 3 0.5
    </State>
    <State> 3 </State>
    </TopologyEntry>
    </Topology>
    """

    topo_string = ostringstream.from_str(topo_string)

    topo = HmmTopology()
    iss = istringstream.from_str(topo_string.to_str())
    topo.read(iss, False)
    return topo

class TestHMMTopology(unittest.TestCase):

    def testHmmTopology(self):

        input_str = """<Topology>
        <TopologyEntry>
        <ForPhones> 1 2 3 4 5 6 7 8 9 </ForPhones>
        <State> 0 <PdfClass> 0
        <Transition> 0 0.5
        <Transition> 1 0.5
        </State>
        <State> 1 <PdfClass> 1
        <Transition> 1 0.5
        <Transition> 2 0.5
        </State>
        <State> 2 <PdfClass> 2
        <Transition> 2 0.5
        <Transition> 3 0.5
        </State>
        <State> 3 </State>
        </TopologyEntry>
        <TopologyEntry>
        <ForPhones> 10 11 13 </ForPhones>
        <State> 0 <PdfClass> 0
        <Transition> 0 0.5
        <Transition> 1 0.5
        </State>
        <State> 1 <PdfClass> 1
        <Transition> 1 0.5
        <Transition> 2 0.5
        </State>
        <State> 2 </State>
        </TopologyEntry>
        </Topology>"""

        chain_input_str = """<Topology>
        <TopologyEntry>
        <ForPhones> 1 2 3 4 5 6 7 8 9 </ForPhones>
        <State> 0 <ForwardPdfClass> 0 <SelfLoopPdfClass> 1
        <Transition> 0 0.5
        <Transition> 1 0.5
        </State>
        <State> 1 </State>
        </TopologyEntry>
        </Topology>
        """

        for i in range(10):
            binary = random.choice([True, False])

            topo = HmmTopology()

            iss = istringstream.from_str(input_str)
            topo.read(iss, False)
            self.assertEqual(3, topo.min_length(3))
            self.assertEqual(2, topo.min_length(11))

            oss = ostringstream()
            topo.write(oss, binary)

            topo2 = HmmTopology()
            iss2 = istringstream.from_str(oss.to_bytes())
            topo2.read(iss2, binary)

            # Test equality
            oss1 = ostringstream()
            oss2 = ostringstream()
            topo.write(oss1, False)
            topo2.write(oss2, False)
            self.assertEqual(oss1.to_str(), oss2.to_str())

            # Test chain topology
            chain_topo = HmmTopology()
            chain_iss = istringstream.from_str(chain_input_str)
            chain_topo.read(chain_iss, False)
            self.assertEqual(1, chain_topo.min_length(3))

            # make sure get_default_topology doesnt crash
            phones = [1, 2]
            get_default_topology(phones)

if __name__ == '__main__':
    unittest.main()
