
#include "util/kaldi-table.h"
#include "fstext/kaldi-fst-io.h"
#include "kws/kaldi-kws.h"

namespace kaldi {

  typedef SequentialTableReader<fst::VectorFstTplHolder<fst::StdArc>> SequentialStdVectorFstReader;
  typedef SequentialTableReader<fst::VectorFstTplHolder<fst::LogArc>> SequentialLogVectorFstReader;
  typedef SequentialTableReader<fst::VectorFstTplHolder<KwsLexicographicArc>> SequentialKwsIndexVectorFstReader;

  typedef RandomAccessTableReader<fst::VectorFstTplHolder<fst::StdArc>> RandomAccessStdVectorFstReader;
  typedef RandomAccessTableReader<fst::VectorFstTplHolder<fst::LogArc>> RandomAccessLogVectorFstReader;
  typedef RandomAccessTableReader<fst::VectorFstTplHolder<KwsLexicographicArc>> RandomAccessKwsIndexVectorFstReader;

  typedef TableWriter<fst::VectorFstTplHolder<fst::StdArc>> StdVectorFstWriter;
  typedef TableWriter<fst::VectorFstTplHolder<fst::LogArc>> LogVectorFstWriter;
  typedef TableWriter<fst::VectorFstTplHolder<KwsLexicographicArc>> KwsIndexVectorFstWriter;
}
