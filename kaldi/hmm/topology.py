import _hmm_topology

# Note (VM):
#   Most of _hmm_topology.HmmTopology methods check for an empty phone list
#   through KALDI_ASSERT. In case of failure, the python interpreter 
#   hard-crashes. Our solution in previous cases (e.g., Matrix or Vector) 
#   implemented a python wrapper through inheritance of the clif wrappers.
#
#   However, for reasons completely unknown to me at this moment, inheritance 
#   does not work for HmmTopology (any call fails with ValueError: Value
#   invalidated due to capture by std::unique_ptr). 
#
#   Thus, here we used our next best option: composition.
class HmmTopology():
    """Python wrapper for kaldi::HmmTopology

    This class defines a more Pythonic user facing API for the kaldi HmmTopology through composition. It protects the user from hard-crashes when calling raw methods, with the downside of having to call self.getObj() everytime a kaldi::HmmTopology object is needed.
    """
    def __init__(self):
        self.obj = _hmm_topology.HmmTopology()

    def read(self, istream, binary):
        self.obj.read(istream, binary)

    def write(self, ostream, binary):
        """Safe-version of Kaldi's HmmTopology.write
        Checks for empty _phones."""
        if len(self.obj.get_phones()) == 0:
            raise ValueError("Cannot write empty phone list")
        self.obj._write(ostream, binary)

    # Note (VM):
    # Doesnt hard crash
    # but throws C++ exception with stack trace
    def check(self):
        self.obj.check()

    def is_hmm(self):
        """Safe-version of Kaldi's HmmTopology.is_hmm
        Checks for empty _phones."""
        if len(self.obj.get_phones()) == 0:
            import warnings
            warnings.warn("HmmTopology.is_hmm() called with empty phone list.")
            return False
        return self.obj.is_hmm()

    # Note (VM):
    # Doesnt hard crash
    # but throws C++ exception with stack trace
    def topology_for_phone(self, phone):
        return self.obj.topology_for_phone(phone)

    def num_pdf_classes(self, phone):
        """Safe-version of Kaldi's HmmTopology.num_pdf_classes
        Checks for empty _phones."""
        if len(self.obj.get_phones()) == 0:
            raise Exception("HmmTopology phone list is empty")
        return self.obj.num_pdf_classes(phone)

    def get_phones(self):
        return self.obj.get_phones()

    def get_phone_to_num_pdf_classes(self):
        """Safe-version of Kaldi's HmmTopology.get_phone_to_num_pdf_classes
        Checks for empty _phones."""
        if len(self.obj.get_phones()) == 0:
            raise Exception("HmmTopology phone list is empty")
        return self.obj._get_phone_to_num_pdf_classes() 

    # Note (VM):
    # Doesnt hard crash
    # but throws C++ exception with stack trace
    def min_length(self, phone):
        return self.obj.min_length(phone) 

    def __eq__(self, other):
        return self.obj.__eq__(other)

    def getObj(self):
        """ Returns raw wrapper for kaldi::HmmTopology"""
        return self.obj

################################################################################
# Define Topology Utility Functions
################################################################################

def construct_topology(topology):
    """Construct a new :class:`HmmTopology` instance from the input topology.

    Args:
        topology (:class:`kaldi.hmm._hmm_topology.HmmTopology`): input topology
    
    Returns:
        A new :class:`HmmTopology` instance.
    """
    res = HmmTopology()
    res.obj = topology
    return res


################################################################################

__all__ = [name for name in dir()
           if name[0] != '_'
           and not name.endswith('Base')]
