from kaldi.asr import Recognizer
from kaldi.decoder import FasterDecoder, FasterDecoderOptions
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.feat.functions import compute_deltas, DeltaFeaturesOptions
from kaldi.fstext import SymbolTable, read_fst_kaldi
from kaldi.gmm.am import AmDiagGmm, DecodableAmDiagGmmScaled
from kaldi.hmm import TransitionModel
from kaldi.util.io import xopen
from kaldi.util.table import SequentialWaveReader

# Define the feature pipeline: (wav) -> feats
def make_feat_pipeline(base, opts=DeltaFeaturesOptions()):
    def feat_pipeline(wav):
        feats = base.compute_features(wav.data()[0], wav.samp_freq, 1.0)
        return compute_deltas(opts, feats)
    return feat_pipeline

feat_pipeline = make_feat_pipeline(Mfcc(MfccOptions()))

# Read the model
with xopen("/home/dogan/tools/pykaldi/egs/models/wsj/final.mdl") as ki:
    trans_model = TransitionModel().read(ki.stream(), ki.binary)
    acoustic_model = AmDiagGmm().read(ki.stream(), ki.binary)

# Define the decodable wrapper: (features, acoustic_scale) -> decodable
def make_decodable_wrapper(trans_model, acoustic_model):
    def decodable_wrapper(features, acoustic_scale):
        return DecodableAmDiagGmmScaled(acoustic_model, trans_model,
                                        features, acoustic_scale)
    return decodable_wrapper

decodable_wrapper = make_decodable_wrapper(trans_model, acoustic_model)

# Define the decoder
decoding_graph = read_fst_kaldi("/home/dogan/tools/pykaldi/egs/models/wsj/HCLG.fst")
decoder_opts = FasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
decoder = FasterDecoder(decoding_graph, decoder_opts)

# Define the recognizer
symbols = SymbolTable.read_text("/home/dogan/tools/pykaldi/egs/models/wsj/words.txt")
asr = Recognizer(decoder, decodable_wrapper, symbols)

# Decode wave files
for key, wav in SequentialWaveReader("scp:/home/dogan/tools/pykaldi/egs/decoder/test2.scp"):
    feats = feat_pipeline(wav)
    out = asr.decode(feats)
    print(key, out["text"], flush=True)
