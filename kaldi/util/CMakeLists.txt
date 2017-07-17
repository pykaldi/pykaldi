add_pyclif_library("kaldi_io" kaldi-io.clif CLIF_DEPS kaldi_matrix)
add_pyclif_library("kaldi_holder" kaldi-holder.clif CLIF_DEPS kaldi_io)
# add_pyclif_library("kaldi_table" kaldi-table.clif CLIF_DEPS kaldi_holder)