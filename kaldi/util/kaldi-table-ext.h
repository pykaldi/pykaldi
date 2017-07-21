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

  class RandomAccessVectorReader
      : public RandomAccessTableReader<KaldiObjectHolder<Vector<float>>> {

  public:
    RandomAccessVectorReader()
        : RandomAccessTableReader<KaldiObjectHolder<Vector<float>>>() {}

    explicit RandomAccessVectorReader(const std::string &rspecifier)
        : RandomAccessTableReader<KaldiObjectHolder<Vector<float>>>(rspecifier) {}

    const Vector<float> &Value(const std::string &key) {
      return RandomAccessTableReader<KaldiObjectHolder<Vector<float>>>::Value(key);
    }

    RandomAccessVectorReader &operator = (RandomAccessVectorReader &&other) {
      this->Close();
      this->impl_ = other.impl_;
      other.impl_ = NULL;
      return *this;
    }

  };

  class RandomAccessMatrixReader
      : public RandomAccessTableReader<KaldiObjectHolder<Matrix<float>>> {

  public:
    RandomAccessMatrixReader()
        : RandomAccessTableReader<KaldiObjectHolder<Matrix<float>>>() {}

    explicit RandomAccessMatrixReader(const std::string &rspecifier)
        : RandomAccessTableReader<KaldiObjectHolder<Matrix<float>>>(rspecifier) {}

    const Matrix<float> &Value(const std::string &key) {
      return RandomAccessTableReader<KaldiObjectHolder<Matrix<float>>>::Value(key);
    }

    RandomAccessMatrixReader &operator = (RandomAccessMatrixReader &&other) {
      this->Close();
      this->impl_ = other.impl_;
      other.impl_ = NULL;
      return *this;
    }

  };

  class VectorWriter : public TableWriter<KaldiObjectHolder<Vector<float>>> {
  public:
    VectorWriter() : TableWriter<KaldiObjectHolder<Vector<float>>>() {}

    explicit VectorWriter(const std::string &wspecifier) : TableWriter<KaldiObjectHolder<Vector<float>>>(wspecifier) {}

    inline void Write(const std::string &key, const T &value) const{
      TableWriter<KaldiObjectHolder<Vector<float>>>::Write(key, value);
    } 

    VectorWriter &operator = (VectorWriter &&other) {
      this->Close();
      this->impl_ = other.impl_;
      other.impl_ = NULL;
      return *this;
    }

  };

  class MatrixWriter : public TableWriter<KaldiObjectHolder<Matrix<float>>> {
  public:
    MatrixWriter() : TableWriter<KaldiObjectHolder<Matrix<float>>>() {}

    explicit MatrixWriter(const std::string &wspecifier) : TableWriter<KaldiObjectHolder<Matrix<float>>>(wspecifier) {}

    inline void Write(const std::string &key, const T &value) const {
      TableWriter<KaldiObjectHolder<Matrix<float>>>::Write(key, value);
    } 

    MatrixWriter &operator = (MatrixWriter &&other) {
      this->Close();
      this->impl_ = other.impl_;
      other.impl_ = NULL;
      return *this;
    }

  };


}

#endif
