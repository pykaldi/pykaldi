
#include "util/kaldi-table.h"
#include "fstext/kaldi-fst-io.h"

namespace kaldi {

  typedef SequentialTableReader<fst::VectorFstTplHolder<fst::StdArc>> SequentialStdVectorFstReader;
  typedef SequentialTableReader<fst::VectorFstTplHolder<fst::LogArc>> SequentialLogVectorFstReader;

  typedef RandomAccessTableReader<fst::VectorFstTplHolder<fst::StdArc>> RandomAccessStdVectorFstReader;
  typedef RandomAccessTableReader<fst::VectorFstTplHolder<fst::LogArc>> RandomAccessLogVectorFstReader;

  typedef TableWriter<fst::VectorFstTplHolder<fst::StdArc>> StdVectorFstWriter;
  typedef TableWriter<fst::VectorFstTplHolder<fst::LogArc>> LogVectorFstWriter;

}
