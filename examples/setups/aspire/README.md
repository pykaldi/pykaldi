# Models

Download and extract ASpIRE chain models by running the following command
inside this directory.

    ./models.sh

This script uses `wget`, `tar` and `unzip` executables, so make sure they are
installed.

# Kaldi Setup

You need a Kaldi installation for running this example setup. If you installed
PyKaldi from source (or if you are using PyKaldi docker image), then you can
simply run the following inside this directory to add Kaldi executables to your
`PATH`.

    source path.sh

If you did not install PyKaldi from source, then you will need to install Kaldi
and edit the first line of `path.sh` before sourcing it.

# Data Preparation

There is an example data setup inside `data` directory. You can skip the rest
of this section if you simply want to run the example setup.

If you want to decode or align your own recordings, you can edit the files in
`data` directory. For decoding, you need to provide `data/wav.scp` and
`data/spk2utt` files. For alignment, you also need to provide `data/text`.

## List of Recordings

The list of utterances `data/wav.scp` has the format:

    utt1 /path/to/utt1.wav
    utt2 /path/to/utt2.wav
    utt3 /path/to/utt3.wav
    utt4 /path/to/utt4.wav
    ...

Note that each utterance should be a relatively short recording, that is seconds
long not minutes long. If you want to decode long recordings, you can segment
them into short utterances using a speech activity detection system.

Also, make sure the wav files are single channel, 16bit PCM files. If your audio
files are in a different file format, you can use `ffmpeg` and `sox` to convert
them to the required format.

## Speaker to Utterance Map

The speaker to utterance map `data/spk2utt` has the format:

    spk1 utt1 utt4 ...
    spk2 utt2 utt3 ...
    ...

If no speaker information is available, make it an identity mapping:

    utt1 utt1
    utt2 utt2
    utt3 utt3
    utt4 utt4
    ...

## Transcripts (needed for alignment)

The list of transcripts in `data/text` has the format:

    utt1 trascript of first utterance
    utt2 trascript of second utterance
    utt3 trascript of third utterance
    utt4 trascript of fourth utterance
    ...

Note that these should be tokenized transcripts and all of the tokens should be
in system vocabulary `exp/langdir/words.txt`.

# ASR

You can decode the utterances listed in `data/wav.scp` with the following
command.

    ./decode.py

Note that the models used in this setup are fairly large so it takes some time
to load models from disk. To decode with these models, you will need around 2GB
memory. The decoding script will print a bunch of logs to stderr.

Decoding outputs are written to `out/decode.out`.

# Alignment

Align the utterances listed in `data/wav.scp` with the transcripts listed in
`data/text`:

    ./align.py

Frame-level alignments are written to `out/align.out`.

Phone-level alignments are written to `out/phone_align.out`.

Word-level alignments are written to `out/word_align.out`.
