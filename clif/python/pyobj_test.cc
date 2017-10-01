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

#include "clif/python/pyobj.h"
#include "testing/base/public/googletest.h"
#include "testing/base/public/gunit.h"

namespace clif {
namespace {

class PyObjectTest : public ::testing::Test {
 protected:
  PyObjectTest() { Py_Initialize(); }
};

TEST_F(PyObjectTest, EmptyCtor) {
  py::Object a;
  EXPECT_EQ(a, nullptr);
}

TEST_F(PyObjectTest, NullCtor) {
  py::Object a = nullptr;
  EXPECT_EQ(a, nullptr);
}

TEST_F(PyObjectTest, Equality) {
  PyObject* obj = PyTuple_New(0);
  py::Object a(obj);
  EXPECT_EQ(a, obj);
  EXPECT_NE(a, nullptr);
  EXPECT_NE(a, Py_None);
  Py_DECREF(obj);

  py::Object b(obj);
  EXPECT_EQ(a, b);
}

TEST_F(PyObjectTest, CtorAndDtorManageRefcount) {
  PyObject* obj = PyTuple_New(0);
  int initial_ref_count = Py_REFCNT(obj);

  {
    py::Object a(obj);
    EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
  }
  EXPECT_EQ(initial_ref_count, Py_REFCNT(obj));
  Py_DECREF(obj);
}

TEST_F(PyObjectTest, CopyCtorIncrements) {
  PyObject* obj = PyTuple_New(0);
  int initial_ref_count = Py_REFCNT(obj);

  {
    py::Object a(obj);
    EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
    {
      py::Object b = a;
      EXPECT_EQ(a, obj);
      EXPECT_EQ(b, obj);
      EXPECT_EQ(initial_ref_count + 2, Py_REFCNT(obj));
    }
    EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
  }
  EXPECT_EQ(initial_ref_count, Py_REFCNT(obj));
  Py_DECREF(obj);
}

TEST_F(PyObjectTest, CopyIncrements) {
  PyObject* obj = PyTuple_New(0);
  int initial_ref_count = Py_REFCNT(obj);

  {
    py::Object a(obj);
    EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
    {
      py::Object b;
      b = a;
      EXPECT_EQ(a, obj);
      EXPECT_EQ(b, obj);
      EXPECT_EQ(initial_ref_count + 2, Py_REFCNT(obj));
    }
    EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
  }
  EXPECT_EQ(initial_ref_count, Py_REFCNT(obj));
  Py_DECREF(obj);
}

TEST_F(PyObjectTest, MoveCtorDoesNotIncrement) {
  PyObject* obj = PyTuple_New(0);
  int initial_ref_count = Py_REFCNT(obj);

  {
    py::Object a(obj);
    EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));

    py::Object b = std::move(a);
    EXPECT_EQ(a, nullptr);
    EXPECT_EQ(b, obj);
    EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
  }
  EXPECT_EQ(initial_ref_count, Py_REFCNT(obj));
  Py_DECREF(obj);
}

TEST_F(PyObjectTest, MoveDoesNotIncrement) {
  PyObject* obj = PyTuple_New(0);
  int initial_ref_count = Py_REFCNT(obj);

  {
    py::Object a(obj);
    EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));

    py::Object b;
    b = std::move(a);
    EXPECT_EQ(a, nullptr);
    EXPECT_EQ(b, obj);
    EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
  }
  EXPECT_EQ(initial_ref_count, Py_REFCNT(obj));
  Py_DECREF(obj);
}

TEST_F(PyObjectTest, SwapDoesNotIncrement) {
  PyObject* obj1 = PyTuple_New(0);
  PyObject* obj2 = PyList_New(0);
  int initial_ref_count1 = Py_REFCNT(obj1);
  int initial_ref_count2 = Py_REFCNT(obj2);
  ASSERT_NE(initial_ref_count1, initial_ref_count2);

  py::Object scoped1(obj1);
  py::Object scoped2(obj2);
  EXPECT_EQ(initial_ref_count1 + 1, Py_REFCNT(obj1));
  EXPECT_EQ(initial_ref_count2 + 1, Py_REFCNT(obj2));

  using std::swap;  // go/using-std-swap
  swap(scoped1, scoped2);
  EXPECT_EQ(scoped2, obj1);
  EXPECT_EQ(scoped1, obj2);
  EXPECT_EQ(initial_ref_count1 + 1, Py_REFCNT(obj1));
  EXPECT_EQ(initial_ref_count2 + 1, Py_REFCNT(obj2));

  Py_DECREF(obj1);
  Py_DECREF(obj2);
  EXPECT_GT(Py_REFCNT(obj1), 0);
  EXPECT_GT(Py_REFCNT(obj2), 0);
}

TEST_F(PyObjectTest, ReleaseDoesNotDescrement) {
  PyObject* obj = PyTuple_New(0);
  int initial_ref_count = Py_REFCNT(obj);

  py::Object scoped(obj);
  EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
  scoped.release();
  EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
  EXPECT_EQ(scoped, nullptr);

  Py_DECREF(obj);  // extra deref, because we released
  Py_DECREF(obj);
}

TEST_F(PyObjectTest, CopyDoesNotLeakOldValue) {
  PyObject* obj = PyTuple_New(0);
  int initial_ref_count = Py_REFCNT(obj);

  py::Object scoped(obj);
  EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
  scoped = py::Object();
  EXPECT_EQ(initial_ref_count, Py_REFCNT(obj));
}

TEST_F(PyObjectTest, AssignFromPyDoesNotLeakOldValue) {
  PyObject* obj = PyTuple_New(0);
  int initial_ref_count = Py_REFCNT(obj);

  py::Object scoped(obj);
  EXPECT_EQ(initial_ref_count + 1, Py_REFCNT(obj));
  scoped = PyList_New(0);
  EXPECT_EQ(initial_ref_count, Py_REFCNT(obj));
}

TEST_F(PyObjectTest, CompareAndBoolOperators) {
  py::Object full(Py_None);
  EXPECT_TRUE(full);
  EXPECT_FALSE(!full);
  EXPECT_NE(full, nullptr);

  py::Object empty(nullptr);
  EXPECT_FALSE(empty);
  EXPECT_TRUE(!empty);
  EXPECT_EQ(empty, nullptr);
}

}  // namespace
}  // namespace clif
