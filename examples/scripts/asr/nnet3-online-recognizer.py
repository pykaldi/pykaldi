#!/usr/bin/env python

from __future__ import print_function

from kaldi.asr import NnetLatticeFasterOnlineRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleLoopedComputationOptions
from kaldi.online2 import (OnlineEndpointConfig,
                           OnlineIvectorExtractorAdaptationState,
                           OnlineNnetFeaturePipelineConfig,
                           OnlineNnetFeaturePipelineInfo,
                           OnlineNnetFeaturePipeline,
                           OnlineSilenceWeighting)
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialWaveReader

chunk_size = 1440

# Define online feature pipeline
feat_opts = OnlineNnetFeaturePipelineConfig()
endpoint_opts = OnlineEndpointConfig()
po = ParseOptions("")
feat_opts.register(po)
endpoint_opts.register(po)
po.read_config_file("online.conf")
feat_info = OnlineNnetFeaturePipelineInfo.from_config(feat_opts)

# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
decodable_opts = NnetSimpleLoopedComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
decodable_opts.frames_per_chunk = 150
asr = NnetLatticeFasterOnlineRecognizer.from_files(
    "final.mdl", "HCLG.fst", "words.txt",
    decoder_opts=decoder_opts,
    decodable_opts=decodable_opts,
    endpoint_opts=endpoint_opts)

# Decode (whole utterance)
for key, wav in SequentialWaveReader("scp:wav.scp"):
    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
    asr.set_input_pipeline(feat_pipeline)
    feat_pipeline.accept_waveform(wav.samp_freq, wav.data()[0])
    feat_pipeline.input_finished()
    out = asr.decode()
    print(key, out["text"], flush=True)

# Decode (chunked + partial output)
for key, wav in SequentialWaveReader("scp:wav.scp"):
    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
    asr.set_input_pipeline(feat_pipeline)
    asr.init_decoding()
    data = wav.data()[0]
    last_chunk = False
    part = 1
    prev_num_frames_decoded = 0
    for i in range(0, len(data), chunk_size):
        if i + chunk_size >= len(data):
            last_chunk = True
        feat_pipeline.accept_waveform(wav.samp_freq, data[i:i + chunk_size])
        if last_chunk:
            feat_pipeline.input_finished()
        asr.advance_decoding()
        num_frames_decoded = asr.decoder.num_frames_decoded()
        if not last_chunk:
            if num_frames_decoded > prev_num_frames_decoded:
                prev_num_frames_decoded = num_frames_decoded
                out = asr.get_partial_output()
                print(key + "-part%d" % part, out["text"], flush=True)
                part += 1
    asr.finalize_decoding()
    out = asr.get_output()
    print(key + "-final", out["text"], flush=True)

# Decode (chunked + partial output + endpointing
#         + ivector adaptation + silence weighting)
adaptation_state = OnlineIvectorExtractorAdaptationState.from_info(
    feat_info.ivector_extractor_info)
for key, wav in SequentialWaveReader("scp:wav.scp"):
    feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
    feat_pipeline.set_adaptation_state(adaptation_state)
    asr.set_input_pipeline(feat_pipeline)
    asr.init_decoding()
    sil_weighting = OnlineSilenceWeighting(
        asr.transition_model, feat_info.silence_weighting_config,
        decodable_opts.frame_subsampling_factor)
    data = wav.data()[0]
    last_chunk = False
    utt, part = 1, 1
    prev_num_frames_decoded, offset = 0, 0
    for i in range(0, len(data), chunk_size):
        if i + chunk_size >= len(data):
            last_chunk = True
        feat_pipeline.accept_waveform(wav.samp_freq, data[i:i + chunk_size])
        if last_chunk:
            feat_pipeline.input_finished()
        if sil_weighting.active():
            sil_weighting.compute_current_traceback(asr.decoder)
            feat_pipeline.ivector_feature().update_frame_weights(
                sil_weighting.get_delta_weights(
                    feat_pipeline.num_frames_ready()))
        asr.advance_decoding()
        num_frames_decoded = asr.decoder.num_frames_decoded()
        if not last_chunk:
            if asr.endpoint_detected():
                asr.finalize_decoding()
                out = asr.get_output()
                print(key + "-utt%d-final" % utt, out["text"], flush=True)
                offset += int(num_frames_decoded
                              * decodable_opts.frame_subsampling_factor
                              * feat_pipeline.frame_shift_in_seconds()
                              * wav.samp_freq)
                feat_pipeline.get_adaptation_state(adaptation_state)
                feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
                feat_pipeline.set_adaptation_state(adaptation_state)
                asr.set_input_pipeline(feat_pipeline)
                asr.init_decoding()
                sil_weighting = OnlineSilenceWeighting(
                    asr.transition_model, feat_info.silence_weighting_config,
                    decodable_opts.frame_subsampling_factor)
                remainder = data[offset:i + chunk_size]
                feat_pipeline.accept_waveform(wav.samp_freq, remainder)
                utt += 1
                part = 1
                prev_num_frames_decoded = 0
            elif num_frames_decoded > prev_num_frames_decoded:
                prev_num_frames_decoded = num_frames_decoded
                out = asr.get_partial_output()
                print(key + "-utt%d-part%d" % (utt, part),
                      out["text"], flush=True)
                part += 1
    asr.finalize_decoding()
    out = asr.get_output()
    print(key + "-utt%d-final" % utt, out["text"], flush=True)
    feat_pipeline.get_adaptation_state(adaptation_state)
