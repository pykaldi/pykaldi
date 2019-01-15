from __future__ import division, print_function

import math

from . import decoder as _dec
from . import fstext as _fst
from .fstext import utils as _fst_utils
from . import matrix as _mat
from .matrix import common as _mat_comm
from . import nnet3 as _nnet3
from .util import io as _util_io


__all__ = ['Segmenter', 'NnetSAD', 'SegmentationProcessor']


class Segmenter(object):
    """Base class for speech segmenters.

    Args:
        graph (StdVectorFst): Segmentation graph.
        beam (float): Logarithmic decoding beam.
        max_active (int): Maximum number of active states in decoding.
        acoustic_scale (float): Acoustic score scale.
    """
    def __init__(self, graph, beam=8, max_active=1000, acoustic_scale=0.1):
        decoder_opts = _dec.FasterDecoderOptions()
        decoder_opts.beam = beam
        decoder_opts.max_active = max_active
        self.decoder = _dec.FasterDecoder(graph, decoder_opts)
        self.acoustic_scale = acoustic_scale

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

    def segment(self, input):
        """Segments input.

        Output is a dictionary with the following `(key, value)` pairs:

        ============== ============================== ==========================
        key            value                          value type
        ============== ============================== ==========================
        "alignment"    Frame-level segmentation       `List[int]`
        "best_path"    Best lattice path              `CompactLattice`
        "likelihood"   Log-likelihood of best path    `float`
        "weight"       Cost of best path              `LatticeWeight`
        ============== ============================== ==========================

        The "weight" output is a lattice weight consisting of (graph-score,
        acoustic-score).

        Args:
            input (object): Input to segment.

        Returns:
            A dictionary representing segmentation output.

        Raises:
            RuntimeError: If segmentation fails.
        """
        self.decoder.decode(self._make_decodable(input))

        if not self.decoder.reached_final():
            raise RuntimeError("No final state was active on the last frame.")

        try:
            best_path = self.decoder.get_best_path()
        except RuntimeError:
            raise RuntimeError("Empty segmentation output.")

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
            "weight": weight,
        }


class NnetSAD(Segmenter):
    """Neural network based speech activity detection (SAD).

    Args:
        model (Nnet): SAD model. Model output should be log-posteriors for
            [silence, speech, garbage] labels.
        transform (Matrix): Transformation applied to SAD label posteriors. It
            should be a 3x2 matrix mapping [silence, speech, garbage] posteriors
            to [silence, speech] pseudo-likelihoods.
        graph (StdVectorFst): SAD graph. Silence and speech arcs should be
            labeled respectively with 1 and 2.
        beam (float): Logarithmic decoding beam.
        max_active (int): Maximum number of active states in decoding.
        decodable_opts (NnetSimpleComputationOptions): Configuration options for
            the SAD model.
    """
    def __init__(self, model, transform, graph, beam=8, max_active=1000,
                 decodable_opts=None):
        if not isinstance(model, _nnet3.Nnet):
            raise TypeError("model argument should be a Nnet object")
        self.model = model
        self.priors = _mat.Vector()
        self.transform = transform
        _nnet3.set_batchnorm_test_mode(True, model)
        _nnet3.set_dropout_test_mode(True, model)
        _nnet3.collapse_model(_nnet3.CollapseModelConfig(), model)
        if decodable_opts:
            if not isinstance(decodable_opts,
                              _nnet3.NnetSimpleComputationOptions):
                raise TypeError("decodable_opts should be either None or a "
                                "NnetSimpleComputationOptions object")
            self.decodable_opts = decodable_opts
        else:
            self.decodable_opts = _nnet3.NnetSimpleComputationOptions()
        self.compiler = _nnet3.CachingOptimizingCompiler.new_with_optimize_opts(
            model, self.decodable_opts.optimize_config)
        super(NnetSAD, self).__init__(graph, beam, max_active,
                                      self.decodable_opts.acoustic_scale)

    def _make_decodable(self, features):
        """Constructs a new decodable object from input features.

        Args:
            features (Matrix): Input features.

        Returns:
            DecodableMatrixScaled: A decodable object for computing scaled
            log-likelihoods.
        """
        if features.num_rows == 0:
            raise ValueError("Empty feature matrix.")

        nnet_computer = _nnet3.DecodableNnetSimple(
            self.decodable_opts, self.model, self.priors,
            features, self.compiler, None, None, 0)

        post = _mat.Matrix(nnet_computer.num_frames(),
                           nnet_computer.output_dim())
        for t in range(nnet_computer.num_frames()):
            nnet_computer.get_output_for_frame(t, post[t])
        post.apply_exp_()
        # FIXME: Need to keep a reference to log_likes to keep it in scope
        self._log_likes = _mat.Matrix(post.num_rows, self.transform.num_rows)
        self._log_likes.add_mat_mat_(post, self.transform,
                                     _mat_comm.MatrixTransposeType.NO_TRANS,
                                     _mat_comm.MatrixTransposeType.TRANS,
                                     1.0, 0.0)
        self._log_likes.apply_log_()
        return _dec.DecodableMatrixScaled(self._log_likes, self.acoustic_scale)

    @staticmethod
    def read_model(model_rxfilename):
        """Reads SAD model from an extended filename."""
        with _util_io.xopen(model_rxfilename) as ki:
            return _nnet3.Nnet().read(ki.stream(), ki.binary)

    @staticmethod
    def read_average_posteriors(post_rxfilename):
        """Reads average SAD label posteriors from an extended filename."""
        with _util_io.xopen(post_rxfilename) as ki:
            return _mat.Vector().read_(ki.stream(), ki.binary)

    @staticmethod
    def make_sad_transform(priors, sil_scale=1.0,
                           sil_in_speech_weight=0.0,
                           speech_in_sil_weight=0.0,
                           garbage_in_speech_weight=0.0,
                           garbage_in_sil_weight=0.0):
        """Creates SAD posterior transformation matrix.

        The 3x2 transformation matrix is used to convert length Nx3 posterior
        probability matrices to Nx2 pseudo-likelihood matrices.

        The "priors" vector can be a proper prior probability distribution over
        SAD labels or simply average SAD label posteriors. This vector is
        normalized to derive a prior probability distribution.

        Args:
            priors (Vector): SAD label priors to remove from the neural network
                output posteriors to convert them to pseudo likelihoods.
            sil_scale (float): Scale on the silence probability. Make this more
                than one to encourage decoding silence.
            sil_in_speech_weight (float): The fraction of silence probability to
                add to speech probability.
            speech_in_sil_weight (float): The fraction of speech probability to
                add to silence probability.
            garbage_in_speech_weight (float): The fraction of garbage
                probability to add to speech probability.
            garbage_in_sil_weight (float): The fraction of garbage probability
                to add to silence probability.
        """
        priors_sum = priors.sum()
        sil_prior = priors[0] / priors_sum
        speech_prior = priors[1] / priors_sum
        garbage_prior = priors[2] / priors_sum

        return _mat.Matrix([[sil_scale / sil_prior,
                             speech_in_sil_weight / speech_prior,
                             garbage_in_sil_weight / garbage_prior],
                            [sil_in_speech_weight / sil_prior,
                             1.0 / speech_prior,
                             garbage_in_speech_weight / garbage_prior]])

    @staticmethod
    def make_sad_graph(transition_scale=1.0, self_loop_scale=0.1,
                       min_silence_duration=0.03, min_speech_duration=0.3,
                       max_speech_duration=10.0, frame_shift=0.01,
                       edge_silence_probability=0.5, transition_probability=0.1):
        """Makes a decoding graph with a simple HMM topology suitable for SAD.

        Output graph uses label 1 for 'silence' and label 2 for 'speech'.

        Args:
            transition_scale (float): Scale on transition log-probabilities
                relative to LM weights.
            self_loop_scale (float): Scale on self-loop log-probabilities
                relative to LM weights.
            min_silence_duration (float): Minimum duration for silence.
            min_speech_duration (float): Minimum duration for speech.
            max_speech_duration (float): Maximum duration for speech.
            frame_shift (float): Frame shift in seconds.
            edge_silence_probability (float): Probability of silence at the
                edges.
            transition_probability (float): Transition probability for silence
                to speech or vice-versa.

        Returns:
            StdVectorFst: A simple decoding graph suitable for SAD.
        """
        min_states_silence = int(min_silence_duration / frame_shift + 0.5)
        min_states_speech = int(min_speech_duration / frame_shift + 0.5)
        max_states_speech = int(max_speech_duration / frame_shift + 0.5)

        symbols = _fst.SymbolTable()
        symbols.add_symbol("<eps>")
        symbols.add_symbol("silence")
        symbols.add_symbol("speech")
        compiler = _fst.StdFstCompiler(symbols, symbols)

        # Initial transition to silence
        print("0 1 silence silence {cost}".format(
                    cost=-math.log(edge_silence_probability)),
              file=compiler)
        silence_start_state = 1

        # Silence min duration transitions
        # 1->2, 2->3 and so on until
        # (1 + min_states_silence - 2) -> (1 + min_states_silence - 1)  ...
        for state in range(silence_start_state,
                           silence_start_state + min_states_silence - 1):
            print ("{state} {next_state} silence silence {cost}".format(
                        state=state, next_state=state + 1, cost=0.0),
                   file=compiler)
        silence_last_state = silence_start_state + min_states_silence - 1

        # Silence self-loop
        print ("{state} {state} silence silence {cost}".format(
                    state=silence_last_state, cost=0.0),
               file=compiler)

        speech_start_state = silence_last_state + 1

        # Initial transition to speech
        print ("0 {state} speech speech {cost}".format(
                    state=speech_start_state,
                    cost=-math.log(1.0 - edge_silence_probability)),
               file=compiler)

        # Silence to speech transition
        print ("{sil_state} {speech_state} speech speech {cost}".format(
                    sil_state=silence_last_state,
                    speech_state=speech_start_state,
                    cost=-math.log(transition_probability)),
               file=compiler)

        # Speech min duration
        for state in range(speech_start_state,
                           speech_start_state + min_states_speech - 1):
            print ("{state} {next_state} speech speech {cost}".format(
                        state=state, next_state=state + 1, cost=0.0),
                   file=compiler)

        # Speech max duration
        for state in range(speech_start_state + min_states_speech - 1,
                           speech_start_state + max_states_speech - 1):
            print ("{state} {next_state} speech speech {cost}".format(
                        state=state, next_state=state + 1, cost=0.0),
                   file=compiler)

            print ("{state} {sil_state} silence silence {cost}".format(
                        state=state, sil_state=silence_start_state,
                        cost=-math.log(transition_probability)),
                   file=compiler)
        speech_last_state = speech_start_state + max_states_speech - 1

        # Transition to silence after max duration of speech
        print ("{state} {sil_state} silence silence {cost}".format(
                    state=speech_last_state, sil_state=silence_start_state,
                    cost=0.0),
               file=compiler)

        for state in range(1, speech_start_state):
            print ("{state} {cost}".format(
                        state=state, cost=-math.log(edge_silence_probability)),
                   file=compiler)

        for state in range(speech_start_state, speech_last_state + 1):
            print ("{state} {cost}".format(
                        state=state,
                        cost=-math.log(1.0 - edge_silence_probability)),
                   file=compiler)

        return compiler.compile()


class SegmentationProcessor(object):
    """Segmentation post-processor.

    This class is used for converting segmentation labels to a list of segments.
    Output includes only those segments labeled with the target labels.

    Post-processing operations include::
        * filtering out short segments
        * padding segments
        * merging consecutive segments

    Args:
        target_labels (List[int]): Target labels. Typically the speech labels.
        frame_shift (float): Frame shift in seconds.
        segment_padding (float): Additional padding on target segments.
            Padding does not go beyond the adjacent segment. This is typically
            used for padding speech segments with silence. Must be an integral
            multiple of frame shift.
        min_segment_dur (float): Minimum duration (in seconds) required for a
            segment to be included. This is before any padding. Segments shorter
            than this duration will be removed.
        max_merged_segment_dur (float): Merge consecutive segments as long as
            the merged segment is no longer than this many seconds. The segments
            are only merged if their boundaries are touching. This is after
            padding by --segment-padding seconds. 0 means do not merge. Use
            'inf' to not limit the duration.

    Attributes:
        stats (SegmentationProcessor.Stats): Global segmentation post-processing
            stats.
    """
    def __init__(self, target_labels, frame_shift=0.01, segment_padding=0.2,
                 min_segment_dur=0, max_merged_segment_dur=0):
        if not float(segment_padding / frame_shift).is_integer():
            raise ValueError("segment_padding = {} is not an integral "
                             "multiple of frame_shift = {}"
                             .format(segment_padding,frame_shift))
        self.target_labels = target_labels
        self.frame_shift = frame_shift
        self.segment_padding = int(segment_padding / frame_shift)
        self.min_segment_dur = int(math.ceil(min_segment_dur / frame_shift))
        self.max_merged_segment_dur = int(max_merged_segment_dur / frame_shift)
        self.stats = self.Stats()

    class Stats(object):
        """Stores segmentation post-processing stats."""

        def __init__(self):
            self.num_segments_initial = 0
            self.num_short_segments_filtered = 0
            self.num_merges = 0
            self.num_segments_final = 0
            self.initial_duration = 0.0
            self.padding_duration = 0.0
            self.filter_short_duration = 0.0
            self.final_duration = 0.0

        def add(self, other):
            """Adds stats from another"""
            self.num_segments_initial += other.num_segments_initial
            self.num_short_segments_filtered += other.num_short_segments_filtered
            self.num_merges += other.num_merges
            self.num_segments_final += other.num_segments_final
            self.initial_duration += other.initial_duration
            self.filter_short_duration += other.filter_short_duration
            self.padding_duration += other.padding_duration
            self.final_duration += other.final_duration

        def __str__(self):
            return ("num-segments-initial={num_segments_initial}, "
                    "num-short-segments-filtered={num_short_segments_filtered}, "
                    "num-merges={num_merges}, "
                    "num-segments-final={num_segments_final}, "
                    "initial-duration={initial_duration}, "
                    "filter-short-duration={filter_short_duration}, "
                    "padding-duration={padding_duration}, "
                    "final-duration={final_duration}".format(
                num_segments_initial=self.num_segments_initial,
                num_short_segments_filtered=self.num_short_segments_filtered,
                num_merges=self.num_merges,
                num_segments_final=self.num_segments_final,
                initial_duration=self.initial_duration,
                filter_short_duration=self.filter_short_duration,
                padding_duration=self.padding_duration,
                final_duration=self.final_duration))

    def process(self, alignment):
        """Converts frame-level segmentation labels to a list of segments.

        Args:
            alignment (List[int]): Frame-level segmentation labels.

        Returns:
            Tuple[List[Tuple[int, int, int]], SegmentationProcessor.Stats]: List
            of segments, where each entry is a (segment-beg, segment-end, label)
            tuple, along with segmentation post-processing stats.
        """
        stats = self.Stats()
        segments = self.initialize_segments(alignment, stats)
        segments = self.filter_short_segments(segments, stats)
        segments = self.pad_segments(segments, stats, len(alignment))
        segments = self.merge_consecutive_segments(segments, stats)
        self.stats.add(stats)
        return segments, stats

    def initialize_segments(self, alignment, stats):
        """Initializes segments.

        The alignment is frame-level segmentation labels. Output includes only
        those segments labeled with the target labels.
        """
        segments = []
        if not alignment:
            return segments
        num_target_frames, seg_begin, seg_label = 0, 0, alignment[0]
        for i, label in enumerate(alignment[1:], 1):
            if label != seg_label:
                if seg_label in self.target_labels:
                    segments.append((seg_begin, i, seg_label))
                    num_target_frames += i - seg_begin
                seg_begin, seg_label = i, label
        if seg_label in self.target_labels:
            segments.append((seg_begin, len(alignment), seg_label))
            num_target_frames += len(alignment) - seg_begin
        stats.num_segments_initial = len(segments)
        stats.num_segments_final = len(segments)
        stats.initial_duration = num_target_frames * self.frame_shift
        stats.final_duration = stats.initial_duration
        return segments

    def filter_short_segments(self, segments, stats):
        """Filters out short segments."""
        if self.min_segment_dur <= 0:
            return segments
        filtered_segments = []
        for segment in segments:
            dur = segment[1] - segment[0]
            if dur < self.min_segment_dur:
                stats.filter_short_duration += dur * self.frame_shift
                stats.num_short_segments_filtered += 1
            else:
                filtered_segments.append(segment)
        stats.num_segments_final = len(filtered_segments)
        stats.final_duration -= stats.filter_short_duration
        return filtered_segments

    def pad_segments(self, segments, stats, num_utt_frames=None):
        """Pads segments on both sides.

        Ensures that the segments do not go beyond the neighboring segments or
        utterance boundaries.
        """
        num_padded_frames, padded_segments = 0, []
        for i, (seg_beg, seg_end, label) in enumerate(segments):
            seg_beg -= self.segment_padding  # try padding on the left side
            num_padded_frames += self.segment_padding
            if seg_beg < 0:
                # Padded segment start is before the beginning of the utterance.
                # Reduce padding.
                num_padded_frames += seg_beg
                seg_beg = 0
            if i >= 1 and prev_seg_end > seg_beg:
                # Padded segment start is before the end of previous segment.
                # Reduce padding.
                num_padded_frames -= prev_seg_end - seg_beg
                seg_beg = prev_seg_end
            seg_end += self.segment_padding
            num_padded_frames += self.segment_padding
            if num_utt_frames is not None and seg_end > num_utt_frames:
                # Padded segment end is beyond the max duration.
                # Reduce padding.
                num_padded_frames -= seg_end - num_utt_frames
                seg_end = num_utt_frames
            if i + 1 < len(segments) and seg_end > segments[i + 1][0]:
                # Padded segment end is beyond the start of the next segment.
                # Reduce padding.
                num_padded_frames -= seg_end - segments[i + 1][0]
                seg_end = segments[i + 1][0]
            padded_segments.append((seg_beg, seg_end, label))
            prev_seg_end = seg_end
        stats.padding_duration = num_padded_frames * self.frame_shift
        stats.final_duration += stats.padding_duration
        return padded_segments

    def merge_consecutive_segments(self, segments, stats):
        """Merges consecutive segments.

        Done after padding. Consecutive segments that share a boundary are
        merged if they have the same label and the merged segment is no longer
        than 'max_merged_segment_dur'.
        """
        if self.max_merged_segment_dur <= 0 or not segments:
            return segments

        merged_segments = [segments[0]]
        for seg_beg, seg_end, label in segments[1:]:
            prev_seg_beg, prev_seg_end, prev_label = merged_segments[-1]
            if (seg_beg == prev_seg_end and label == prev_label and
                seg_end - prev_seg_beg <= self.max_merged_segment_dur):
                merged_segments[-1] = (prev_seg_beg, seg_end, label)
                stats.num_merges += 1
            else:
                merged_segments.append((seg_beg, seg_end, label))

        stats.num_segments_final = len(merged_segments)
        return merged_segments

    def write(self, key, segments, file_handle):
        """Writes segments to file"""
        for begin, end, label in segments:
            id = "{key}-{label}-{begin:07d}-{end:07d}".format(
                key=key, label=label, begin=begin, end=end)
            print("{id} {key} {begin:.2f} {end:.2f}".format(
                      id=id, key=key, begin=begin * self.frame_shift,
                      end=end * self.frame_shift),
                  file=file_handle)
