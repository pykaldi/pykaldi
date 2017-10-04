
#include <memory>
#include "clif/python/optional.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "clif/python/postconv.h"

namespace kaldi {
using namespace ::clif;

// CLIF use `::kaldi::SubVector<float>` as SubVector
bool Clif_PyObjAs(PyObject* input, ::kaldi::SubVector<float>** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::kaldi::SubVector<float>>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::kaldi::SubVector<float>>* output);
bool Clif_PyObjAs(PyObject* input, ::kaldi::SubVector<float>* output);
bool Clif_PyObjAs(PyObject* input, ::gtl::optional<::kaldi::SubVector<float>>* output);
PyObject* Clif_PyObjFrom(::kaldi::SubVector<float>*, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::kaldi::SubVector<float>>, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::kaldi::SubVector<float>>, py::PostConv);
PyObject* Clif_PyObjFrom(const ::kaldi::SubVector<float>&, py::PostConv);
// CLIF use `::kaldi::SubMatrix<float>` as SubMatrix
bool Clif_PyObjAs(PyObject* input, ::kaldi::SubMatrix<float>** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::kaldi::SubMatrix<float>>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::kaldi::SubMatrix<float>>* output);
bool Clif_PyObjAs(PyObject* input, ::kaldi::SubMatrix<float>* output);
bool Clif_PyObjAs(PyObject* input, ::gtl::optional<::kaldi::SubMatrix<float>>* output);
PyObject* Clif_PyObjFrom(::kaldi::SubMatrix<float>*, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::kaldi::SubMatrix<float>>, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::kaldi::SubMatrix<float>>, py::PostConv);
PyObject* Clif_PyObjFrom(const ::kaldi::SubMatrix<float>&, py::PostConv);

}  // namespace kaldi

// CLIF init_module if (PyObject* m = PyImport_ImportModule("_matrix_ext")) Py_DECREF(m);
// CLIF init_module else goto err;
