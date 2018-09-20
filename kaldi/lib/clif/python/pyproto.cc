// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
From .../python-2.7.3-docs-html/c-api/intro.html#include-files:
Since Python may define some pre-processor definitions which affect the
standard headers on some systems, you must include Python.h before any standard
headers are included.
*/
#include <Python.h>
#include <string>

#include "clif/python/pyproto.h"
#include "clif/python/runtime.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"

#if PY_MAJOR_VERSION < 3
#define DESCRIPTOR_TypeCheck PyBytes_Check
#else
#define DESCRIPTOR_TypeCheck PyUnicode_Check
#define PyString_AS_STRING PyUnicode_AsUTF8
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#endif  // Python version check.

namespace {

class ModNameComponents {
 public:
  explicit ModNameComponents(const std::string& str) : str_(str) {
    Split();
  }

  std::vector<std::string>::const_iterator begin() const noexcept {
    return components_.begin();
  }

  std::vector<std::string>::const_iterator end() const noexcept {
    return components_.end();
  }

 private:
  void Split() {
    std::size_t cur_start = 0;
    while (true) {
      std::size_t i = str_.find('.', cur_start);
      if (i == std::string::npos) {
        components_.emplace_back(str_.substr(cur_start));
        break;
      } else {
        components_.emplace_back(str_.substr(cur_start, i - cur_start));
        cur_start = i + 1;
      }
    }
  }

  std::string str_;
  std::vector<std::string> components_;
};

}  // namespace

namespace clif {

bool Clif_PyObjAs(PyObject* py, std::unique_ptr<::proto2::Message>* c) {
  assert(c != nullptr);
  PyObject* pyd = PyObject_GetAttrString(py, C("DESCRIPTOR"));
  if (pyd == nullptr) return false;
  PyObject* fn = PyObject_GetAttrString(pyd, C("full_name"));
  Py_DECREF(pyd);
  if (fn == nullptr) return false;
  if (!DESCRIPTOR_TypeCheck(fn)) {
    PyErr_SetString(PyExc_TypeError, "DESCRIPTOR.full_name must return str");
    Py_DECREF(fn);
    return false;
  }
  const proto2::DescriptorPool* dp = proto2::DescriptorPool::generated_pool();
  if (dp == nullptr) {
    PyErr_SetNone(PyExc_MemoryError);
    Py_DECREF(fn);
    return false;
  }
  const proto2::Descriptor* d = dp->FindMessageTypeByName(
      PyString_AS_STRING(fn));
  if (d == nullptr) {
    PyErr_Format(PyExc_TypeError, "DESCRIPTOR.full_name %s not found",
                 PyString_AS_STRING(fn));
    Py_DECREF(fn);
    return false;
  }
  Py_DECREF(fn);
  proto2::Message* m = proto2::MessageFactory::generated_factory()->
      GetPrototype(d)->New();
  if (m == nullptr) {
    PyErr_SetNone(PyExc_MemoryError);
    return false;
  }
  {
    if (!proto::TypeCheck(
        py, ImportFQName("google.protobuf.message.Message"),
        "", "proto2_Message_subclass")) return false;
    PyObject* ser = proto::Serialize(py);
    if (ser == nullptr) return false;
    std::unique_ptr<proto2::io::CodedInputStream> coded_input_stream(new
        proto2::io::CodedInputStream(
            reinterpret_cast<uint8_t*>(PyBytes_AS_STRING(ser)),
            PyBytes_GET_SIZE(ser)));
    if (!m->MergePartialFromCodedStream(coded_input_stream.get())) {
      PyErr_SetString(PyExc_ValueError, "Parse from serialization failed");
      Py_DECREF(ser);
      return false;
    }
    Py_DECREF(ser);
  }
  *c = ::std::unique_ptr<::proto2::Message>(m);
  return true;
}

namespace proto {

bool SetNestedName(PyObject** module_name, const char* nested_name) {
  assert(module_name != nullptr);
  assert(*module_name != nullptr);
  assert(nested_name != nullptr);
  if (*nested_name) {
    for (const auto& n : ModNameComponents(nested_name)) {
      PyObject* atr = PyObject_GetAttr(
          *module_name, PyString_FromStringAndSize(n.data(), n.size()));
      Py_DECREF(*module_name);
      if (atr == nullptr) return false;
      *module_name = atr;
    }
  }
  return true;
}

// Check the given pyproto to be class_name instance.
bool TypeCheck(PyObject* pyproto,
              PyObject* imported_pyproto_class,  // takes ownership
              const char* nested_name,
              const char* class_name) {
  if (imported_pyproto_class == nullptr) return false;  // Import failed.
  if (!SetNestedName(&imported_pyproto_class, nested_name)) return false;
  int proto_instance = PyObject_IsInstance(pyproto, imported_pyproto_class);
  Py_DECREF(imported_pyproto_class);
  if (proto_instance < 0 ) return false;  // Exception already set.
  if (!proto_instance)
    PyErr_Format(PyExc_TypeError, "expecting %s proto, got %s %s",
                 class_name, ClassName(pyproto), ClassType(pyproto));
  return proto_instance;
}

// Return bytes serialization of the given pyproto.
PyObject* Serialize(PyObject* pyproto) {
  PyObject* raw = PyObject_CallMethod(pyproto, C("SerializePartialToString"),
                                      nullptr);
  if (raw == nullptr) return nullptr;
  if (!PyBytes_Check(raw)) {
    PyErr_Format(PyExc_TypeError, "%s.SerializePartialToString() must return"
                 " bytes, got %s %s", ClassName(pyproto),
                 ClassName(raw), ClassType(raw));
    Py_DECREF(raw);
    return nullptr;
  }
  return raw;
}


PyObject* PyProtoFrom(const ::proto2::Message* cproto,
                      PyObject* imported_pyproto_class,
                      const char* nested_name) {
  assert(cproto != nullptr);
  if (imported_pyproto_class == nullptr) return nullptr;  // Import failed.
  if (!SetNestedName(&imported_pyproto_class, nested_name)) return nullptr;
  PyObject* pb = PyObject_CallObject(imported_pyproto_class, nullptr);
  Py_DECREF(imported_pyproto_class);
  if (pb == nullptr) return nullptr;
    string bytes = cproto->SerializePartialAsString();
#if PY_MAJOR_VERSION < 3
  // Python will automatically intern strings that are small, or look like
  // identifiers, so there is no actual need to call InternFromString and it's
  // gone in Py3.
  PyObject* merge = PyString_InternFromString("MergeFromString");
  PyObject* cpb = PyBuffer_FromMemory((void*)bytes.data(), bytes.size());  // NOLINT[readability/casting]
#else
  PyObject* merge = PyUnicode_FromString("MergeFromString");
  PyObject* cpb = PyMemoryView_FromMemory(const_cast<char*>(bytes.data()),
                                          bytes.size(), PyBUF_READ);
#endif
  if (!merge || !cpb) {
    Py_DECREF(pb);
    Py_XDECREF(merge);
    Py_XDECREF(cpb);
    return nullptr;
  }
  PyObject* ret = PyObject_CallMethodObjArgs(pb, merge, cpb, nullptr);
  Py_DECREF(merge);
  Py_DECREF(cpb);
  if (ret == nullptr) {
    Py_DECREF(pb);
    return nullptr;
  }
  Py_DECREF(ret);
  return pb;
}
}  // namespace proto
}  // namespace clif
