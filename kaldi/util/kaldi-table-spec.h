#include "util/kaldi-table.h"
#include "matrix/kaldi-vector.h"

namespace kaldi {

    template<typename Real>
    class SequentialKaldiVectorTableReader : public SequentialTableReaderImplBase<KaldiObjectHolder<Vector<Real>>>{
    public:
        SequentialKaldiVectorTableReader() : SequentialTableReaderImplBase<KaldiObjectHolder<Vector<Real>>>() { }
        SequentialKaldiVectorTableReader(const std::string &rspecifier) : SequentialTableReaderImplBase<KaldiObjectHolder<Vector<Real>>>(&rspecifier) { }
        const Vector<Real> &Value(){
            SequentialTableReaderImplBase<KaldiObjectHolder<Vector<Real>>>::CheckImpl();
            return SequentialTableReaderImplBase<KaldiObjectHolder<Vector<Real>>>::impl_->Value();  // This may throw (if EnsureObjectLoaded() returned false you
                                    // are safe.).
        }
    };

}