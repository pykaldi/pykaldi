/*
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CLIF_PYTHON_POSTCONV_H_
#define CLIF_PYTHON_POSTCONV_H_

// Python postconversion functions for all (esp. container) types.
//
// The need for this arises from the fact that a single C++ type (eg.
// std::string) might need an extra conversion whether it's converting to 'str'
// or 'bytes' (in Python 3). When such conversion occurs inside a nested
// container (e.g. dict<str, tuple<str, bytes>>), passing that info
// along is complicated.

// Implementation:
// CLIF generates calls to PyObjFrom(T, {}) when no conversion is needed or
// PyObjFrom(T, {...{...{..., F ...}}}) with initalizer list corresponding to
// Python types that need postconversion function F.

// Each PyObjFrom implementation call Apply() on plain types or Get(N)
// on container types with one or more Apply() calls inside like that:
//   f(T x, pc) { return pc.Apply(x); }  // pc can be {} or Function
//   f(list<T> x, {F}) { pc.Get(0).Apply(each x) }

#include <vector>

struct _object; typedef _object PyObject;  // We only need PyObject* here.

namespace clif {

namespace py {
namespace postconv {
inline PyObject* PASS(PyObject* x) { return x; }
}  // namespace postconv

class PostConv {
  typedef PyObject* (*Func)(PyObject*);  // postcoversion function

 public:
  typedef std::vector<PostConv> Array;
  PyObject* Apply(PyObject* x) const {
    if (noop_) return x;
        return f_(x);
  }
  const PostConv& Get(Array::size_type i) const {
    if (noop_) return getNoop();
        return c_.at(i);
  }
  PostConv() : noop_(true), f_(nullptr) {}
  PostConv(Func f) : noop_(false), f_(f ? f : postconv::PASS) {}
  PostConv(std::initializer_list<PostConv> lst)
      : noop_(!lst.size()), f_(nullptr), c_(lst) {}

 private:
  friend class PostConvTest;
  static const PostConv& getNoop() {
    static PostConv& noconversions = *new PostConv();
    return noconversions;
  }
  bool noop_;
  Func f_;
  Array c_;
};
}  // namespace py
}  // namespace clif
#endif  // CLIF_PYTHON_POSTCONV_H_
