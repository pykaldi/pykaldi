#include "util/kaldi-table.h"
#include "matrix/kaldi-vector.h"

namespace kaldi {

  template<typename Real>
  class SequentialVectorReader
      : public SequentialTableReader<KaldiObjectHolder<Vector<Real>>> {
  public:
    SequentialVectorReader()
        : SequentialTableReader<KaldiObjectHolder<Vector<Real>>>() {}

    explicit SequentialVectorReader(const std::string &rspecifier)
        : SequentialTableReader<KaldiObjectHolder<Vector<Real>>>(rspecifier) {}

    const Vector<Real> &Value() {
        return SequentialTableReader<KaldiObjectHolder<Vector<Real>>>::Value();
    }

    SequentialVectorReader<Real> &operator = (
        const SequentialVectorReader<Real> &&other) {
      this->Close();
      this->impl_ = other.impl_;
      other.impl_ = NULL;
      return *this;
    }

  };

}
