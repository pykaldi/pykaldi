#ifndef PYKALDI_KALDI_TABLE_EXT_H_
#define PYKALDI_KALDI_TABLE_EXT_H_ 1

#include "util/kaldi-table.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

  class SequentialVectorReader
      : public SequentialTableReader<KaldiObjectHolder<Vector<float>>> {
  public:
    SequentialVectorReader()
        : SequentialTableReader<KaldiObjectHolder<Vector<float>>>() {}

    explicit SequentialVectorReader(const std::string &rspecifier)
        : SequentialTableReader<KaldiObjectHolder<Vector<float>>>(rspecifier) {}

    const Vector<float> &Value() {
        return SequentialTableReader<KaldiObjectHolder<Vector<float>>>::Value();
    }

    SequentialVectorReader &operator = (SequentialVectorReader &&other) {
      this->Close();
      this->impl_ = other.impl_;
      other.impl_ = NULL;
      return *this;
    }

  };

  class SequentialMatrixReader
      : public SequentialTableReader<KaldiObjectHolder<Matrix<float>>> {

  public:
    SequentialMatrixReader()
        : SequentialTableReader<KaldiObjectHolder<Matrix<float>>>() {}

    explicit SequentialMatrixReader(const std::string &rspecifier)
        : SequentialTableReader<KaldiObjectHolder<Matrix<float>>>(rspecifier) {}

    const Matrix<float> &Value() {
      return SequentialTableReader<KaldiObjectHolder<Matrix<float>>>::Value();
    }

    SequentialMatrixReader &operator = (SequentialMatrixReader &&other) {
      this->Close();
      this->impl_ = other.impl_;
      other.impl_ = NULL;
      return *this;
    }

  };

}

#endif
