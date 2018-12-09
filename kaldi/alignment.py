from __future__ import division

from .base import io as _base_io
from . import decoder as _dec
from . import fstext as _fst
from .fstext import utils as _fst_utils
from .gmm import am as _gmm_am
from . import hmm as _hmm
from .lat import align as _lat_align
from .lat import functions as _lat_funcs
from .matrix import _kaldi_matrix
from . import nnet3 as _nnet3
from . import tree as _tree
from .util import io as _util_io


__all__ = ['Aligner', 'MappedAligner', 'GmmAligner', 'NnetAligner']


class Aligner(object):
    """Speech aligner.

    This can be used to align transition-id log-likelihood matrices with
    reference texts.

    Args:
        transition_model (TransitionModel): The transition model.
        tree (ContextDependency): The phonetic decision tree.
        lexicon (StdFst): The lexicon FST.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        disambig_symbols (List[int]): Disambiguation symbols.
        graph_compiler_opts (TrainingGraphCompilerOptions): Configuration
            options for graph compiler.
        beam (float): Decoding beam used in alignment.
        transition_scale (float): The scale on non-self-loop transition
            probabilities.
        self_loop_scale (float): The scale on self-loop transition
            probabilities.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, tree, lexicon, symbols=None,
                 disambig_symbols=None, graph_compiler_opts=None, beam=200.0,
                 transition_scale=1.0, self_loop_scale=1.0, acoustic_scale=0.1):
        self.transition_model = transition_model
        self.symbols = symbols
        if not graph_compiler_opts:
            graph_compiler_opts = _dec.TrainingGraphCompilerOptions()
        self.graph_compiler = _dec.TrainingGraphCompiler(
            transition_model, tree, lexicon,
            disambig_symbols, graph_compiler_opts)
        self.decoder_opts = _dec.FasterDecoderOptions()
        self.decoder_opts.beam = beam
        self.transition_scale = transition_scale
        self.self_loop_scale = self_loop_scale
        self.acoustic_scale = acoustic_scale

    @staticmethod
    def read_tree(tree_rxfilename):
        """Reads phonetic decision tree from an extended filename.

        Returns:
            ContextDependency: Phonetic decision tree.
        """
        tree = _tree.ContextDependency()
        with _util_io.xopen(tree_rxfilename) as ki:
            tree.read(ki.stream(), ki.binary)
        return tree

    @staticmethod
    def read_lexicon(lexicon_rxfilename):
        """Reads lexicon FST from an extended filename.

        Returns:
            StdFst: Lexicon FST.
        """
        return _fst.read_fst_kaldi(lexicon_rxfilename)

    @staticmethod
    def read_symbols(symbols_filename):
        """Reads symbol table from file.

        Returns:
            SymbolTable: Symbol table.
        """
        if symbols_filename is None:
            return None
        else:
            return _fst.SymbolTable.read_text(symbols_filename)

    @staticmethod
    def read_disambig_symbols(disambig_rxfilename):
        """Reads disambiguation symbols from an extended filename.

        Returns:
            List[int]: List of disambiguation symbols.
        """
        if disambig_rxfilename is None:
            return None
        else:
            with _util_io.xopen(disambig_rxfilename, "rt") as ki:
                return [int(line.strip()) for line in ki]

    @staticmethod
    def read_model(model_rxfilename):
        """Reads transition model from an extended filename.

        Returns:
            TransitionModel: Transition model.
        """
        with _util_io.xopen(model_rxfilename) as ki:
            return _hmm.TransitionModel().read(ki.stream(), ki.binary)

    @classmethod
    def from_files(cls, model_rxfilename, tree_rxfilename, lexicon_rxfilename,
                   symbols_filename=None, disambig_rxfilename=None,
                   graph_compiler_opts=None, beam=200.0, transition_scale=1.0,
                   self_loop_scale=1.0, acoustic_scale=0.1):
        """Constructs a new GMM aligner from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the transition
                model.
            tree_rxfilename (str): Extended filename for reading the phonetic
                decision tree.
            lexicon_rxfilename (str): Extended filename for reading the lexicon
                FST.
            symbols_filename (str): The symbols file. If provided, "text" input
                of :meth:`align` should include symbols instead of integer
                indices.
            disambig_rxfilename (str): Extended filename for reading the list
                of disambiguation symbols.
            graph_compiler_opts (TrainingGraphCompilerOptions): Configuration
                options for graph compiler.
            beam (float): Decoding beam used in alignment.
            transition_scale (float): The scale on non-self-loop transition
                probabilities.
            self_loop_scale (float): The scale on self-loop transition
                probabilities.
            acoustic_scale (float): Acoustic score scale.

        Returns:
            A new aligner object.
        """
        transition_model = cls.read_model(model_rxfilename)
        tree = cls.read_tree(tree_rxfilename)
        lexicon = cls.read_lexicon(lexicon_rxfilename)
        symbols = cls.read_symbols(symbols_filename)
        disambig_symbols = cls.read_disambig_symbols(disambig_rxfilename)
        return cls(transition_model, tree, lexicon, symbols,
                   disambig_symbols, graph_compiler_opts, beam,
                   transition_scale, self_loop_scale, acoustic_scale)

    def _make_decodable(self, loglikes):
        """Constructs a new decodable object from input log-likelihoods.

        Args:
            loglikes (object): Input log-likelihoods.

        Returns:
            DecodableMatrixScaled: A decodable object for computing scaled
            log-likelihoods.
        """
        if loglikes.num_rows == 0:
            raise ValueError("Empty loglikes matrix.")
        return _dec.DecodableMatrixScaled(loglikes, self.acoustic_scale)

    def align(self, input, text):
        """Aligns input with text.

        Output is a dictionary with the following `(key, value)` pairs:

        ================ =========================== ===========================
        key              value                       value type
        ================ =========================== ===========================
        "alignment"      Frame-level alignment       `List[int]`
        "best_path"      Best lattice path           `CompactLattice`
        "likelihood"     Log-likelihood of best path `float`
        "weight"         Cost of best path           `LatticeWeight`
        ================ =========================== ===========================

        If :attr:`symbols` is ``None``, the "text" input should be a
        string of space separated integer indices. Otherwise it should be a
        string of space separated symbols. The "weight" output is a lattice
        weight consisting of (graph-score, acoustic-score).

        Args:
            input (object): Input to align.
            text (str): Reference text to align.

        Returns:
            A dictionary representing alignment output.

        Raises:
            RuntimeError: If alignment fails.
        """
        if self.symbols:
            words = _fst.symbols_to_indices(self.symbols, text.split())
        else:
            words = text.split()

        graph = self.graph_compiler.compile_graph_from_text(words)
        _hmm.add_transition_probs(self.transition_model, [],
                                  self.transition_scale, self.self_loop_scale,
                                  graph)
        decoder = _dec.FasterDecoder(graph, self.decoder_opts)
        decoder.decode(self._make_decodable(input))

        if not decoder.reached_final():
            raise RuntimeError("No final state was active on the last frame.")

        try:
            best_path = decoder.get_best_path()
        except RuntimeError:
            raise RuntimeError("Empty alignment output.")

        ali, _, weight = _fst_utils.get_linear_symbol_sequence(best_path)
        likelihood = - (weight.value1 + weight.value2)

        if self.acoustic_scale != 0.0:
            scale = _fst_utils.acoustic_lattice_scale(1.0 / self.acoustic_scale)
            _fst_utils.scale_lattice(scale, best_path)

        best_path = _fst_utils.convert_lattice_to_compact_lattice(best_path)

        return {
            "alignment": ali,
            "best_path": best_path,
            "likelihood": likelihood,
            "weight": weight
        }

    def to_phone_alignment(self, alignment, phones=None):
        """Converts frame-level alignment to phone-level alignment.

        Args:
            alignment (List[int]): Frame-level alignment.
            phones (SymbolTable): The phone symbol table. If provided, output
                includes symbols instead of integer indices.

        Returns:
            List[Tuple[int,int,int]]: A list of triplets representing, for
            each phone in the input, the phone index/symbol, the begin time (in
            frames) and the duration (in frames).
        """
        success, split_ali = _hmm.split_to_phones(self.transition_model,
                                                  alignment)
        if not success:
            raise RuntimeError("Alignment phone split failed.")
        phone_start, phone_alignment = 0, []
        for entry in split_ali:
            phone = self.transition_model.transition_id_to_phone(entry[0])
            if phones:
                phone = phones.find_symbol(phone)
            phone_duration = len(entry)
            phone_alignment.append((phone, phone_start, phone_duration))
            phone_start += phone_duration
        return phone_alignment

    def to_word_alignment(self, best_path, word_boundary_info):
        """Converts best alignment path to word-level alignment.

        Args:
            best_path (CompactLattice): Best alignment path.
            word_boundary_info (WordBoundaryInfo): Word boundary information.

        Returns:
            List[Tuple[int,int,int]]: A list of triplets representing, for each
            word in the input, the word index/symbol, the begin time (in frames)
            and the duration (in frames). The zero/epsilon words correspond to
            optional silences.
        """
        success, best_path = _lat_align.word_align_lattice(
            best_path, self.transition_model, word_boundary_info, 0)
        if not success:
            raise RuntimeError("Lattice word alignment failed.")
        word_alignment = _lat_funcs.compact_lattice_to_word_alignment(best_path)
        if self.symbols:
            mapper = lambda x: (self.symbols.find_symbol(x[0]), x[1], x[2])
        else:
            mapper = lambda x: x
        return list(map(mapper, zip(*word_alignment)))


class MappedAligner(Aligner):
    """Mapped speech aligner.

    This can be used to align phone-id log-likelihood matrices with reference
    texts.

    Args:
        transition_model (TransitionModel): The transition model.
        tree (ContextDependency): The phonetic decision tree.
        lexicon (StdFst): The lexicon FST.
        symbols (SymbolTable): The symbol table. If provided, "text" output of
            :meth:`decode` includes symbols instead of integer indices.
        disambig_symbols (List[int]): Disambiguation symbols.
        graph_compiler_opts (TrainingGraphCompilerOptions): Configuration
            options for graph compiler.
        beam (float): Decoding beam used in alignment.
        transition_scale (float): The scale on non-self-loop transition
            probabilities.
        self_loop_scale (float): The scale on self-loop transition
            probabilities.
        acoustic_scale (float): Acoustic score scale.
    """

    def _make_decodable(self, loglikes):
        """Constructs a new decodable object from input log-likelihoods.

        Args:
            loglikes (object): Input log-likelihoods.

        Returns:
            DecodableMatrixScaledMapped: A decodable object for computing scaled
            log-likelihoods.
        """
        if loglikes.num_rows == 0:
            raise ValueError("Empty loglikes matrix.")
        return _dec.DecodableMatrixScaledMapped(self.transition_model, loglikes,
                                                self.acoustic_scale)


class GmmAligner(Aligner):
    """GMM based speech aligner.

    This can be used to align feature matrices with reference texts.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmDiagGmm): The acoustic model.
        tree (ContextDependency): The phonetic decision tree.
        lexicon (StdFst): The lexicon FST.
        symbols (SymbolTable): The symbol table. If provided, "text" input of
            :meth:`align` should include symbols instead of integer indices.
        disambig_symbols (List[int]): Disambiguation symbols.
        graph_compiler_opts (TrainingGraphCompilerOptions): Configuration
            options for graph compiler.
        beam (float): Decoding beam used in alignment.
        transition_scale (float): The scale on non-self-loop transition
            probabilities.
        self_loop_scale (float): The scale on self-loop transition
            probabilities.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, transition_model, acoustic_model, tree, lexicon,
                 symbols=None, disambig_symbols=None, graph_compiler_opts=None,
                 beam=200.0, transition_scale=1.0, self_loop_scale=1.0,
                 acoustic_scale=0.1):
        if not isinstance(acoustic_model, _gmm_am.AmDiagGmm):
            raise TypeError("acoustic_model should be a AmDiagGmm object")
        self.acoustic_model = acoustic_model
        super(GmmAligner, self).__init__(transition_model, tree, lexicon,
                                         symbols, disambig_symbols,
                                         graph_compiler_opts, beam,
                                         transition_scale, self_loop_scale,
                                         acoustic_scale)

    @staticmethod
    def read_model(model_rxfilename):
        """Reads model from an extended filename.

        Returns:
            Tuple[TransitionModel, AmDiagGmm]: A (transition model, acoustic
            model) pair.
        """
        with _util_io.xopen(model_rxfilename) as ki:
            transition_model = _hmm.TransitionModel().read(ki.stream(),
                                                           ki.binary)
            acoustic_model = _gmm_am.AmDiagGmm().read(ki.stream(), ki.binary)
        return transition_model, acoustic_model

    @classmethod
    def from_files(cls, model_rxfilename, tree_rxfilename, lexicon_rxfilename,
                   symbols_filename=None, disambig_rxfilename=None,
                   graph_compiler_opts=None, beam=200.0, transition_scale=1.0,
                   self_loop_scale=1.0, acoustic_scale=0.1):
        """Constructs a new GMM aligner from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            tree_rxfilename (str): Extended filename for reading the phonetic
                decision tree.
            lexicon_rxfilename (str): Extended filename for reading the lexicon
                FST.
            symbols_filename (str): The symbols file. If provided, "text" input
                of :meth:`align` should include symbols instead of integer
                indices.
            disambig_rxfilename (str): Extended filename for reading the list
                of disambiguation symbols.
            graph_compiler_opts (TrainingGraphCompilerOptions): Configuration
                options for graph compiler.
            beam (float): Decoding beam used in alignment.
            transition_scale (float): The scale on non-self-loop transition
                probabilities.
            self_loop_scale (float): The scale on self-loop transition
                probabilities.
            acoustic_scale (float): Acoustic score scale.

        Returns:
            A new aligner object.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        tree = cls.read_tree(tree_rxfilename)
        lexicon = cls.read_lexicon(lexicon_rxfilename)
        symbols = cls.read_symbols(symbols_filename)
        disambig_symbols = cls.read_disambig_symbols(disambig_rxfilename)
        return cls(transition_model, acoustic_model, tree, lexicon, symbols,
                   disambig_symbols, graph_compiler_opts, beam,
                   transition_scale, self_loop_scale, acoustic_scale)

    def _make_decodable(self, features):
        """Constructs a new decodable object from input features.

        Args:
            features (object): Input features.

        Returns:
            DecodableAmDiagGmmScaled: A decodable object for computing scaled
            log-likelihoods.
        """
        if features.num_rows == 0:
            raise ValueError("Empty feature matrix.")
        return _gmm_am.DecodableAmDiagGmmScaled(self.acoustic_model,
                                                self.transition_model,
                                                features, self.acoustic_scale)


class NnetAligner(Aligner):
    """Neural network based speech aligner.

    This can be used to align feature matrices with reference texts.

    Args:
        transition_model (TransitionModel): The transition model.
        acoustic_model (AmNnetSimple): The acoustic model.
        tree (ContextDependency): The phonetic decision tree.
        lexicon (StdFst): The lexicon FST.
        symbols (SymbolTable): The symbol table. If provided, "text" input of
            :meth:`align` should include symbols instead of integer indices.
        disambig_symbols (List[int]): Disambiguation symbols.
        graph_compiler_opts (TrainingGraphCompilerOptions): Configuration
            options for graph compiler.
        beam (float): Decoding beam used in alignment.
        transition_scale (float): The scale on non-self-loop transition
            probabilities.
        self_loop_scale (float): The scale on self-loop transition
            probabilities.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            simple nnet3 am decodable objects.
        online_ivector_period (int): Onlne ivector period. Relevant only if
            online ivectors are used.
    """
    def __init__(self, transition_model, acoustic_model, tree, lexicon,
                 symbols=None, disambig_symbols=None, graph_compiler_opts=None,
                 beam=200.0, transition_scale=1.0, self_loop_scale=1.0,
                 decodable_opts=None, online_ivector_period=10):
        if not isinstance(acoustic_model, _nnet3.AmNnetSimple):
            raise TypeError("acoustic_model should be a AmNnetSimple object")
        self.acoustic_model = acoustic_model
        nnet = self.acoustic_model.get_nnet()
        _nnet3.set_batchnorm_test_mode(True, nnet)
        _nnet3.set_dropout_test_mode(True, nnet)
        _nnet3.collapse_model(_nnet3.CollapseModelConfig(), nnet)
        if decodable_opts:
            if not isinstance(decodable_opts,
                              _nnet3.NnetSimpleComputationOptions):
                raise TypeError("decodable_opts should be either None or a "
                                "NnetSimpleComputationOptions object")
            self.decodable_opts = decodable_opts
        else:
            self.decodable_opts = _nnet3.NnetSimpleComputationOptions()
        self.compiler = _nnet3.CachingOptimizingCompiler.new_with_optimize_opts(
            nnet, self.decodable_opts.optimize_config)
        self.online_ivector_period = online_ivector_period
        super(NnetAligner, self).__init__(transition_model, tree, lexicon,
                                          symbols, disambig_symbols,
                                          graph_compiler_opts, beam,
                                          transition_scale, self_loop_scale,
                                          self.decodable_opts.acoustic_scale)

    @staticmethod
    def read_model(model_rxfilename):
        """Reads model from an extended filename.

        Returns:
            Tuple[TransitionModel, AmNnetSimple]: A (transition model, acoustic
            model) pair.
        """
        with _util_io.xopen(model_rxfilename) as ki:
            transition_model = _hmm.TransitionModel().read(ki.stream(),
                                                           ki.binary)
            acoustic_model = _nnet3.AmNnetSimple().read(ki.stream(), ki.binary)
        return transition_model, acoustic_model

    @classmethod
    def from_files(cls, model_rxfilename, tree_rxfilename, lexicon_rxfilename,
                   symbols_filename=None, disambig_rxfilename=None,
                   graph_compiler_opts=None, beam=200.0, transition_scale=1.0,
                   self_loop_scale=1.0, decodable_opts=None,
                   online_ivector_period=10):
        """Constructs a new nnet3 aligner from given files.

        Args:
            model_rxfilename (str): Extended filename for reading the model.
            tree_rxfilename (str): Extended filename for reading the phonetic
                decision tree.
            lexicon_rxfilename (str): Extended filename for reading the lexicon
                FST.
            symbols_filename (str): The symbols file. If provided, "text" input
                of :meth:`align` should include symbols instead of integer
                indices.
            disambig_rxfilename (str): Extended filename for reading the list
                of disambiguation symbols.
            graph_compiler_opts (TrainingGraphCompilerOptions): Configuration
                options for graph compiler.
            beam (float): Decoding beam used in alignment.
            transition_scale (float): The scale on non-self-loop transition
                probabilities.
            self_loop_scale (float): The scale on self-loop transition
                probabilities.
            decodable_opts (NnetSimpleComputationOptions): Configuration options
                for simple nnet3 am decodable objects.
            online_ivector_period (int): Onlne ivector period. Relevant only if
                online ivectors are used.

        Returns:
            A new aligner object.
        """
        transition_model, acoustic_model = cls.read_model(model_rxfilename)
        tree = cls.read_tree(tree_rxfilename)
        lexicon = cls.read_lexicon(lexicon_rxfilename)
        disambig_symbols = cls.read_disambig_symbols(disambig_rxfilename)
        symbols = cls.read_symbols(symbols_filename)
        return cls(transition_model, acoustic_model, tree, lexicon, symbols,
                   disambig_symbols, graph_compiler_opts, beam,
                   transition_scale, self_loop_scale, decodable_opts,
                   online_ivector_period)

    def _make_decodable(self, features):
        """Constructs a new decodable object from input features.

        Input can be just a feature matrix or a tuple of a feature matrix and
        an ivector or a tuple of a feature matrix and an online ivector matrix.

        Args:
            features (Matrix or Tuple[Matrix, Vector] or Tuple[Matrix, Matrix]):
                Input features.

        Returns:
            DecodableAmNnetSimple: A decodable object for computing scaled
            log-likelihoods.
        """
        ivector, online_ivectors = None, None
        if isinstance(features, tuple):
            features, ivector_features = features
            if isinstance(ivector_features, _kaldi_matrix.MatrixBase):
                online_ivectors = ivector_features
            else:
                ivector = ivector_features
        if features.num_rows == 0:
            raise ValueError("Empty feature matrix.")
        return _nnet3.DecodableAmNnetSimple(
            self.decodable_opts, self.transition_model, self.acoustic_model,
            features, ivector, online_ivectors, self.online_ivector_period,
            self.compiler)
