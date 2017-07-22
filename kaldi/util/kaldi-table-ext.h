#ifndef PYKALDI_UTIL_KALDI_TABLE_EXT_H_
#define PYKALDI_UTIL_KALDI_TABLE_EXT_H_ 1

#include "util/kaldi-table.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/compressed-matrix.h"

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

  };

  class SequentialIntReader
      : public SequentialTableReader<BasicHolder<int32>> {
   public:
    SequentialIntReader()
        : SequentialTableReader<BasicHolder<int32>>() {}

    explicit SequentialIntReader(const std::string &rspecifier)
        : SequentialTableReader<BasicHolder<int32>>(rspecifier) {}

    const int32 &Value() {
        return SequentialTableReader<BasicHolder<int32>>::Value();
    }

  };

  class SequentialFloatReader
      : public SequentialTableReader<BasicHolder<float>> {
   public:
    SequentialFloatReader()
        : SequentialTableReader<BasicHolder<float>>() {}

    explicit SequentialFloatReader(const std::string &rspecifier)
        : SequentialTableReader<BasicHolder<float>>(rspecifier) {}

    const float &Value() {
        return SequentialTableReader<BasicHolder<float>>::Value();
    }

  };

  class SequentialDoubleReader
      : public SequentialTableReader<BasicHolder<double>> {
   public:
    SequentialDoubleReader()
        : SequentialTableReader<BasicHolder<double>>() {}

    explicit SequentialDoubleReader(const std::string &rspecifier)
        : SequentialTableReader<BasicHolder<double>>(rspecifier) {}

    const double &Value() {
        return SequentialTableReader<BasicHolder<double>>::Value();
    }

  };

  class SequentialBoolReader
      : public SequentialTableReader<BasicHolder<bool>> {
   public:
    SequentialBoolReader()
        : SequentialTableReader<BasicHolder<bool>>() {}

    explicit SequentialBoolReader(const std::string &rspecifier)
        : SequentialTableReader<BasicHolder<bool>>(rspecifier) {}

    const bool &Value() {
        return SequentialTableReader<BasicHolder<bool>>::Value();
    }

  };

  class SequentialIntVectorReader
      : public SequentialTableReader<BasicVectorHolder<int32>> {
   public:
    SequentialIntVectorReader()
        : SequentialTableReader<BasicVectorHolder<int32>>() {}

    explicit SequentialIntVectorReader(const std::string &rspecifier)
        : SequentialTableReader<BasicVectorHolder<int32>>(rspecifier) {}

    const std::vector<int32> &Value() {
        return SequentialTableReader<BasicVectorHolder<int32>>::Value();
    }

  };

  class SequentialIntVectorVectorReader
      : public SequentialTableReader<BasicVectorVectorHolder<int32>> {
   public:
    SequentialIntVectorVectorReader()
        : SequentialTableReader<BasicVectorVectorHolder<int32>>() {}

    explicit SequentialIntVectorVectorReader(const std::string &rspecifier)
        : SequentialTableReader<BasicVectorVectorHolder<int32>>(rspecifier) {}

    const std::vector<std::vector<int32>> &Value() {
        return SequentialTableReader<BasicVectorVectorHolder<int32>>::Value();
    }

  };

  class SequentialIntPairVectorReader
      : public SequentialTableReader<BasicPairVectorHolder<int32>> {
   public:
    SequentialIntPairVectorReader()
        : SequentialTableReader<BasicPairVectorHolder<int32>>() {}

    explicit SequentialIntPairVectorReader(const std::string &rspecifier)
        : SequentialTableReader<BasicPairVectorHolder<int32>>(rspecifier) {}

    const std::vector<std::pair<int32, int32>> &Value() {
        return SequentialTableReader<BasicPairVectorHolder<int32>>::Value();
    }

  };

  class SequentialFloatPairVectorReader
      : public SequentialTableReader<BasicPairVectorHolder<float>> {
   public:
    SequentialFloatPairVectorReader()
        : SequentialTableReader<BasicPairVectorHolder<float>>() {}

    explicit SequentialFloatPairVectorReader(const std::string &rspecifier)
        : SequentialTableReader<BasicPairVectorHolder<float>>(rspecifier) {}

    const std::vector<std::pair<float, float>> &Value() {
        return SequentialTableReader<BasicPairVectorHolder<float>>::Value();
    }

  };

  // Random Access Readers

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

  };

  class RandomAccessIntReader
      : public RandomAccessTableReader<BasicHolder<int32>> {
   public:
    RandomAccessIntReader()
        : RandomAccessTableReader<BasicHolder<int32>>() {}

    explicit RandomAccessIntReader(const std::string &rspecifier)
        : RandomAccessTableReader<BasicHolder<int32>>(rspecifier) {}

    const int32 &Value(const std::string &key) {
        return RandomAccessTableReader<BasicHolder<int32>>::Value(key);
    }

  };

  class RandomAccessFloatReader
      : public RandomAccessTableReader<BasicHolder<float>> {
   public:
    RandomAccessFloatReader()
        : RandomAccessTableReader<BasicHolder<float>>() {}

    explicit RandomAccessFloatReader(const std::string &rspecifier)
        : RandomAccessTableReader<BasicHolder<float>>(rspecifier) {}

    const float &Value(const std::string &key) {
        return RandomAccessTableReader<BasicHolder<float>>::Value(key);
    }

  };

  class RandomAccessDoubleReader
      : public RandomAccessTableReader<BasicHolder<double>> {
   public:
    RandomAccessDoubleReader()
        : RandomAccessTableReader<BasicHolder<double>>() {}

    explicit RandomAccessDoubleReader(const std::string &rspecifier)
        : RandomAccessTableReader<BasicHolder<double>>(rspecifier) {}

    const double &Value(const std::string &key) {
        return RandomAccessTableReader<BasicHolder<double>>::Value(key);
    }

  };

  class RandomAccessBoolReader
      : public RandomAccessTableReader<BasicHolder<bool>> {
   public:
    RandomAccessBoolReader()
        : RandomAccessTableReader<BasicHolder<bool>>() {}

    explicit RandomAccessBoolReader(const std::string &rspecifier)
        : RandomAccessTableReader<BasicHolder<bool>>(rspecifier) {}

    const bool &Value(const std::string &key) {
        return RandomAccessTableReader<BasicHolder<bool>>::Value(key);
    }

  };

  class RandomAccessIntVectorReader
      : public RandomAccessTableReader<BasicVectorHolder<int32>> {
   public:
    RandomAccessIntVectorReader()
        : RandomAccessTableReader<BasicVectorHolder<int32>>() {}

    explicit RandomAccessIntVectorReader(const std::string &rspecifier)
        : RandomAccessTableReader<BasicVectorHolder<int32>>(rspecifier) {}

    const std::vector<int32> &Value(const std::string &key) {
        return RandomAccessTableReader<BasicVectorHolder<int32>>::Value(key);
    }

  };

  class RandomAccessIntVectorVectorReader
      : public RandomAccessTableReader<BasicVectorVectorHolder<int32>> {
   public:
    RandomAccessIntVectorVectorReader()
        : RandomAccessTableReader<BasicVectorVectorHolder<int32>>() {}

    explicit RandomAccessIntVectorVectorReader(const std::string &rspecifier)
        : RandomAccessTableReader<BasicVectorVectorHolder<int32>>(rspecifier) {}

    const std::vector<std::vector<int32>> &Value(const std::string &key) {
        return RandomAccessTableReader<BasicVectorVectorHolder<int32>>::Value(key);
    }

  };

  class RandomAccessIntPairVectorReader
      : public RandomAccessTableReader<BasicPairVectorHolder<int32>> {
   public:
    RandomAccessIntPairVectorReader()
        : RandomAccessTableReader<BasicPairVectorHolder<int32>>() {}

    explicit RandomAccessIntPairVectorReader(const std::string &rspecifier)
        : RandomAccessTableReader<BasicPairVectorHolder<int32>>(rspecifier) {}

    const std::vector<std::pair<int32, int32>> &Value(const std::string &key) {
        return RandomAccessTableReader<BasicPairVectorHolder<int32>>::Value(key);
    }

  };

  class RandomAccessFloatPairVectorReader
      : public RandomAccessTableReader<BasicPairVectorHolder<float>> {
   public:
    RandomAccessFloatPairVectorReader()
        : RandomAccessTableReader<BasicPairVectorHolder<float>>() {}

    explicit RandomAccessFloatPairVectorReader(const std::string &rspecifier)
        : RandomAccessTableReader<BasicPairVectorHolder<float>>(rspecifier) {}

    const std::vector<std::pair<float, float>> &Value(const std::string &key) {
        return RandomAccessTableReader<BasicPairVectorHolder<float>>::Value(key);
    }

  };

  // Writers

  class VectorWriter : public TableWriter<KaldiObjectHolder<Vector<float>>> {
  public:
    VectorWriter() : TableWriter<KaldiObjectHolder<Vector<float>>>() {}

    explicit VectorWriter(const std::string &wspecifier)
        : TableWriter<KaldiObjectHolder<Vector<float>>>(wspecifier) {}

    inline void Write(const std::string &key, const T &value) const{
      TableWriter<KaldiObjectHolder<Vector<float>>>::Write(key, value);
    }

  };

  class MatrixWriter : public TableWriter<KaldiObjectHolder<Matrix<float>>> {
  public:
    MatrixWriter() : TableWriter<KaldiObjectHolder<Matrix<float>>>() {}

    explicit MatrixWriter(const std::string &wspecifier)
        : TableWriter<KaldiObjectHolder<Matrix<float>>>(wspecifier) {}

    inline void Write(const std::string &key, const T &value) const {
      TableWriter<KaldiObjectHolder<Matrix<float>>>::Write(key, value);
    }

  };

  class CompressedMatrixWriter
      : public TableWriter<KaldiObjectHolder<CompressedMatrix>> {
  public:
    CompressedMatrixWriter()
        : TableWriter<KaldiObjectHolder<CompressedMatrix>>() {}

    explicit CompressedMatrixWriter(const std::string &wspecifier)
        : TableWriter<KaldiObjectHolder<CompressedMatrix>>(wspecifier) {}

    inline void Write(const std::string &key, const T &value) const {
      TableWriter<KaldiObjectHolder<CompressedMatrix>>::Write(key, value);
    }

  };

  class IntVectorWriter : public TableWriter<BasicVectorHolder<int32>> {
   public:
    IntVectorWriter() : TableWriter<BasicVectorHolder<int32>>() {}

    explicit IntVectorWriter(const std::string &wspecifier)
        : TableWriter<BasicVectorHolder<int32>>(wspecifier) {}

    inline void Write(const std::string &key,
                      const std::vector<int32> &value) const {
      TableWriter<BasicVectorHolder<int32>>::Write(key, value);
    }

  };

  class IntVectorVectorWriter
      : public TableWriter<BasicVectorVectorHolder<int32>> {
   public:
    IntVectorVectorWriter() : TableWriter<BasicVectorVectorHolder<int32>>() {}

    explicit IntVectorVectorWriter(const std::string &wspecifier)
        : TableWriter<BasicVectorVectorHolder<int32>>(wspecifier) {}

    inline void Write(const std::string &key,
                      const std::vector<std::vector<int32>> &value) const {
      TableWriter<BasicVectorVectorHolder<int32>>::Write(key, value);
    }

  };

  class IntPairVectorWriter
      : public TableWriter<BasicPairVectorHolder<int32>> {
   public:
    IntPairVectorWriter() : TableWriter<BasicPairVectorHolder<int32>>() {}

    explicit IntPairVectorWriter(const std::string &wspecifier)
        : TableWriter<BasicPairVectorHolder<int32>>(wspecifier) {}

    inline void Write(const std::string &key,
                      const std::vector<std::pair<int32, int32>> &value) const {
      TableWriter<BasicPairVectorHolder<int32>>::Write(key, value);
    }

  };

  class FloatPairVectorWriter
      : public TableWriter<BasicPairVectorHolder<float>> {
   public:
    FloatPairVectorWriter() : TableWriter<BasicPairVectorHolder<float>>() {}

    explicit FloatPairVectorWriter(const std::string &wspecifier)
        : TableWriter<BasicPairVectorHolder<float>>(wspecifier) {}

    inline void Write(const std::string &key,
                      const std::vector<std::pair<float, float>> &value) const {
      TableWriter<BasicPairVectorHolder<float>>::Write(key, value);
    }

  };

}  // namespace kaldi

#endif  // PYKALDI_UTIL_KALDI_TABLE_EXT_H_
