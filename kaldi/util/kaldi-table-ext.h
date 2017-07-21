#include "util/kaldi-table.h"
#include "matrix/kaldi-vector.h"

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

}
