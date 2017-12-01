from . import _cmvn
from .. import matrix
from ..matrix import _kaldi_matrix
from ..matrix import _kaldi_vector
from ..util import io

class Cmvn(object):
    """Cepstral mean variance normalization (CMVN).

    This class is used for accumulating CMVN statistics and applying CMVN using
    accumulated statistics. Global CMVN can be computed and applied as follows::

        cmvn = Cmvn(40)  # 40 dimensional features

        for key, feats in SequentialMatrixReader("ark:feats.ark"):
            cmvn.accumulate(feats)

        with MatrixWriter("ark:feats_norm.ark") as writer:
            for key, feats in SequentialMatrixReader("ark:feats.ark"):
                cmvn.apply(feats, norm_vars=True)
                writer[key] = feats

    Attributes:
        stats (DoubleMatrix or None): Accumulated mean/variance statistics
            matrix of size `2 x dim+1`. This matrix is initialized when a CMVN
            object is instantiated by specifying a feature dimension or when the
            :meth:`init` method is explicitly called. It is ``None`` otherwise.
            `stats[0,:-1]` represents the sum of accumulated feature vectors.
            `stats[1,:-1]` represents the sum of element-wise squares of
            accumulated feature vectors. `stats[0,-1]` represents the total
            count of accumulated feature vectors. `stats[1,-1]` is initialized
            to zero but otherwise is not used.
    """
    def __init__(self, dim=None):
        """
        Args:
            dim (int): The feature dimension. If specified, :attr:`stats` matrix
                is initialized with the given dimension.
        """
        self.init(dim)

    def accumulate(self, feats, weights=None):
        """Accumulates CMVN statistics.

        Computes the CMVN statistics for given features and adds them to the
        :attr:`stats` matrix.

        Args:
            feats (VectorBase or MatrixBase): The input features.
            weights (float or VectorBase): The frame weights. If `feats` is a
                single feature vector, then `weights` should be a single number.
                If `feats` is a feature matrix, then `weights` should be a
                vector of size `feats.num_rows`. If `weights` is not specified
                or ``None``, then no frame weighting is done.
        """
        if not self.stats:
            raise ValueError("CMVN stats matrix is not initialized. Initialize "
                             "it either by reading it from file or by calling "
                             "the init method to accumulate new statistics or "
                             "by directly setting the stats attribute.")
        if isinstance(feats, _kaldi_matrix.MatrixBase):
            _cmvn.acc_cmvn_stats(feats, weights, self.stats)
        elif isinstance(feats, _kaldi_vector.VectorBase):
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
            raise ValueError("CMVN stats matrix is not initialized. Initialize "
                             "it either by reading it from file or by calling "
                             "the init method and accumulating new statistics "
                             "or by directly setting the stats attribute.")
        if reverse:
            _cmvn.apply_cmvn_reverse(self.stats, norm_vars, feats)
        else:
            _cmvn.apply_cmvn(self.stats, norm_vars, feats)

    def init(self, dim):
        """Initializes the CMVN statistics matrix.

        This method is called during object initialization. It can also be
        called at a later time to initialize or reset the internal statistics
        matrix.

        Args:
            dim (int or None): The feature dimension. If ``None``, then
                :attr:`stats` attribute is set to ``None``. Otherwise, it is
                initialized as a `2 x dim+1` matrix of zeros.
    """
        if dim is None:
            self.stats = None
        else:
            assert(dim > 0)
            self.stats = matrix.DoubleMatrix(2, dim + 1)

    def read_stats(self, rxfilename, binary=True):
        """Reads CMVN statistics from file.

        Args:
            rxfilename (str): Extended filename for reading CMVN statistics.
            binary (bool): Whether to open the file in binary mode.
        """
        with io.Input(rxfilename, binary=binary) as ki:
            self.stats = matrix.DoubleMatrix().read_(ki.stream(), ki.binary)

    def skip_dims(self, dims):
        """Modifies the stats to skip normalization for given dimensions.

        This is a destructive operation. The statistics for given dimensions are
        replaced with fake values that effectively disable normalization in
        those dimensions. This method should only be called after statistics
        accumulation is finished since accumulation modifies the :attr:`stats`
        matrix.

        Args:
            dims(List[int]): Dimensions for which to skip normalization.
        """
        _cmvn.fake_stats_for_some_dims(dims, self.stats)

    def write_stats(self, wxfilename, binary=True):
        """Writes CMVN statistics to file.

        Args:
            wxfilename (str): Extended filename for writing CMVN statistics.
            binary (bool): Whether to open the file in binary mode.
        """
        with io.Output(wxfilename, binary=binary) as ko:
            self.stats.write(ko.stream(), binary)


__all__ = ['Cmvn']
