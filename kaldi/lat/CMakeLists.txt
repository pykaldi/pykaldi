add_pyclif_library("_kaldi_lattice" kaldi-lattice.clif
	CLIF_DEPS _vector_fst
	LIBRARIES kaldi-lat
	RPATH $ORIGIN/../fstext
)