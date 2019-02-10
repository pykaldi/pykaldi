# Example Setup for Offline ASR

1. You need a Kaldi installation for running this example setup. If you
   installed PyKaldi from source, then you can simply run the following from
   within this directory and Kaldi executables will be added to your `PATH`.

        source path.sh

   If you did not install PyKaldi from source, then you will need to install
   Kaldi and edit the first line of `path.sh` before sourcing it.

2. Download and extract ASpIRE chain models.

        ./models.sh

3. (Optional) Edit the list of utterances `wav.scp`. It has the format:

        utt1 /path/to/utt1.wav
        utt2 /path/to/utt2.wav
        utt3 /path/to/utt3.wav
        utt4 /path/to/utt4.wav
        ...

   Note that each utterance should be a relatively short recording, that is
   seconds long not minutes long. If you want to decode long recordings, you can
   segment them into short utterances using a speech activity detection system.

   Also, make sure the wav files are single channel, 16bit PCM files. If your
   audio files are in a different file format, you can use `ffmpeg` and `sox`
   to convert them to the required format.

4. (Optional) Edit speaker to utterance map `spk2utt`. It has the format:

        spk1 utt1 utt4 ...
        spk2 utt2 utt3 ...
        ...

   If no speaker information is available, make it an identity mapping:

        utt1 utt1
        utt2 utt2
        utt3 utt3
        utt4 utt4
        ...

5. Decode the utterances listed in `wav.scp`:

        ./decode.py > decode.out 2> decode.log
