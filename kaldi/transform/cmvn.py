from . import _cmvn
from .. import matrix
from ..matrix import _kaldi_matrix
from ..matrix import _kaldi_vector
from ..util import io

class Cmvn(object):
    """Cepstral mean variance normalization (CMVN).

    This class is used for accumulating CMVN statistics and applying CMVN using
    accumulated statistics.

    The :attr:`stats` attribute is the accumulated mean and variance statistics
    matrix of size `(2, dim+1)`. `stats[0,:-1]` represents the sum of
    accumulated feature vectors. `stats[1,:-1]` represents the sum of
    element-wise squares of accumulated feature vectors. `stats[0,-1]`
    represents the total count of accumulated feature vectors. `stats[1,-1]` is
    initialized to zero but otherwise is not used.

    Attributes:
        stats (DoubleMatrix or None): Accumulated mean/variance statistics
            matrix of size `(2, dim+1)`. ``None`` if the stats matrix is not
            initialized yet.
    """
    def __init__(self, dim=None):
        """
        Args:
            dim (int): The feature dimension. If specified, it is used for
                initializing the `stats` matrix by calling the init method.
        """
        self.stats = None if dim is None else self.init(dim)

    def accumulate(self, feats, weights=None):
        """Accumulates CMVN statistics.

        Computes the CMVN statistics for the given features and adds them
        to the internal statistics matrix.

        If `feats` is a single feature vector, then `weights` should be a number
        or ``None``. If `feats` is a feature matrix, then `weights` should be a
        vector with size `feat.num_rows` or ``None``.

        Args:
            feats (VectorBase or MatrixBase): The input features.
            weights (float or VectorBase): The frame weights.
                Defaults to``1.0`` for each frame if omitted.
        """
        if not self.stats:
            raise ValueError("CMVN stats matrix is not initialized. It should "
                             "be initialized either by reading from file or by "
                             "calling the init method and accumulating stats.")
        if isinstance(feats, _kaldi_matrix.MatrixBase):
            _cmvn.acc_cmvn_stats(feats, weights, self.stats)
        elif isinstance(feats, _kaldi_matrix.VectorBase):
            if weights is None:
                weights = 1.0
            _cmvn.acc_cmvn_stats_single_frame(feats, weights, self.stats)
        else:
            raise TypeError("Input feature should be a matrix or vector.")

    def apply(self, feats, norm_vars=False, reverse=False):
        """Applies CMVN to the given feature matrix.

        Args:
            feats (Matrix): The feature matrix to normalize.
            norm_vars (bool): Whether to apply variance normalization.
            reverse (bool): Whether to apply CMVN in a reverse sense, so as to
                transform zero-mean, unit-variance features into features with
                the desired mean and variance.
        """
        if not self.stats:
            raise ValueError("CMVN stats matrix is not initialized. It should "
                             "be initialized either by reading from file or by "
                             "calling the init method and accumulating stats.")
        if reverse:
            _cmvn.apply_cmvn_reverse(self.stats, norm_vars, feats)
        else:
            _cmvn.apply_cmvn(self.stats, norm_vars, feats)

    def init(self, dim):
        """Initializes the CMVN statistics matrix.

        The :attr:`stats` matrix is initialized as a `2 x dim+1` matrix of
        zeros.

        This method is called during object initialization if the feature
        dimension is specified when constructing the object. It can be also be
        called at a later time to initialize or reset the internal statistics
        matrix.

        Args:
            dim (int): The feature dimension.
        """
        assert(dim > 0)
        self.stats = matrix.DoubleMatrix(2, dim + 1)

    def read(self, rxfilename, binary=True):
        """Reads CMVN statistics from file.

        Args:
            rxfilename (str): Extended filename for reading CMVN statistics.
            binary (bool): Whether to open the file in binary mode.
        """
        with io.Input(rxfilename, binary=binary) as ki:
            self.stats = matrix.DoubleMatrix().read_(ki.stream(), ki.binary)

    def skip_dims(self, dims):
        """Modifies the stats to skip normalization for given dimensions.

        This is a destructive operation. The stats for given dimensions are
        replaced with fake values that effectively disable normalization in
        those dimensions. This method should only be called after stats
        accumulation is finished since accumulation modifies the stats matrix.

        Args:
            dims(List[int]): Dimensions for which to skip normalization.
        """
        _cmvn.fake_stats_for_some_dims(dims, self.stats)

    def write(self, wxfilename, binary=True):
        """Writes CMVN statistics to file.

        Args:
            wxfilename (str): Extended filename for writing CMVN statistics.
            binary (bool): Whether to open the file in binary mode.
        """
        with io.Output(wxfilename, binary=binary) as ko:
            self.stats.write(ko.stream(), binary)


__all__ = ['Cmvn']
