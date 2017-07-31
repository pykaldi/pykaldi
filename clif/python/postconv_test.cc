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

#include "clif/python/postconv.h"
#include "testing/base/public/gunit.h"

namespace {
PyObject* pc1(PyObject* i) { return i; }
PyObject* pc2(PyObject* i) { return nullptr; }
}  // namespace

namespace clif {
namespace py {

#define _0 postconv::PASS
#define _1 pc1

struct PostConvTest : testing::Test {
  size_t size(PostConv pc) const { return pc.c_.size(); }
  PostConv::Func getf(PostConv pc) const { return pc.f_; }
};

TEST_F(PostConvTest, EmptyInit) {
  PostConv pc = {};
  EXPECT_EQ(nullptr, pc.Apply(nullptr));
  // Implementation-dependent tests:
  EXPECT_EQ(nullptr, getf(pc));
  EXPECT_EQ(0, size(pc));
}

TEST_F(PostConvTest, EmptyPass) {
  PostConv pc = PostConv({});
  EXPECT_EQ(nullptr, pc.Apply(nullptr));
  // Test it works as a container as well.
  pc.Get(0);
}

TEST_F(PostConvTest, Int0Ctor) {
  PostConv pc(_0);
  EXPECT_TRUE(_0 == getf(pc));
}

TEST_F(PostConvTest, Int1Ctor) {
  PostConv pc(_1);
  EXPECT_TRUE(_1 == getf(pc));
}

TEST_F(PostConvTest, Init1) {
  PostConv pc{_1};
  EXPECT_EQ(1, size(pc));
  EXPECT_TRUE(_1 == getf(pc.Get(0)));
}

TEST_F(PostConvTest, Init01) {
  PostConv pc{_0, _1};
  EXPECT_EQ(2, size(pc));
  pc.Get(0);
  pc.Get(1);
}

TEST_F(PostConvTest, Nested) {
  PostConv pc{_0, {_0}, _1};
  EXPECT_EQ(3, size(pc));
  PostConv nested = pc.Get(1);
  EXPECT_EQ(1, size(nested));
  EXPECT_EQ(nullptr, nested.Get(0).Apply(nullptr));
  pc.Get(2);
}

TEST_F(PostConvTest, Nested2) {
  PostConv pc{_0, {{_0, _1}, _0}, _1};
  EXPECT_EQ(3, size(pc));
  {
    PostConv nested1 = pc.Get(1);
    EXPECT_EQ(2, size(nested1));
    {
      PostConv nested2 = nested1.Get(0);
      EXPECT_EQ(2, size(nested2));
    }
  }
}
}  // namespace py
}  // namespace clif
