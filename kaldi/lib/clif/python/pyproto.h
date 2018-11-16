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
#ifndef CLIF_PYTHON_PYPROTO_H_
#define CLIF_PYTHON_PYPROTO_H_

#include <memory>

#include "google/protobuf/message.h"

namespace proto2 = google::protobuf;
struct _object; typedef _object PyObject;

namespace clif {

// CLIF use `::proto2::Message` as proto2_Message
//
// Since proto2::Message and subclasses are not wrapped by CLIF, it does not
// know about the base class. Let's teach CLIF to do the conversion.
bool Clif_PyObjAs(PyObject*, std::unique_ptr<::proto2::Message>*);

namespace proto {
// Check the given pyproto to be class_name type.
bool TypeCheck(PyObject* pyproto, PyObject* imported_pyproto_class,
               const char* element_name, const char* class_name);

// Return bytes serialization of the given pyproto.
PyObject* Serialize(PyObject* pyproto);

// Use underlying C++ protocol message pointer if available and safe.
inline const proto2::Message* GetCProto(PyObject* py) {
  return nullptr;
}

// Return pyproto converted from cproto.
PyObject* PyProtoFrom(const ::proto2::Message* cproto,
                      PyObject* imported_pyproto_class,
                      const char* elemnt_name);
}  // namespace proto
}  // namespace clif

#endif  // CLIF_PYTHON_PYPROTO_H_
