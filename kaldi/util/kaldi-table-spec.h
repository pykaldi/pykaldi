#include "util/kaldi-table.h"
#include "matrix/kaldi-vector.h"

namespace kaldi {

    template<typename Real>
    class SequentialKaldiVectorTableReader : public SequentialTableReader<KaldiObjectHolder<Vector<Real>>>{
    public:
        SequentialKaldiVectorTableReader() : SequentialTableReader<KaldiObjectHolder<Vector<Real>>>() { }
        SequentialKaldiVectorTableReader(const std::string &rspecifier) : SequentialTableReader<KaldiObjectHolder<Vector<Real>>>(&rspecifier) { }
        const Vector<Real> &Value() {
            return SequentialTableReader<KaldiObjectHolder<Vector<Real>>>::Value();
        }
    };

}