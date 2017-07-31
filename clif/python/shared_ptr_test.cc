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

#include "clif/python/shared_ptr.h"
#include "testing/base/public/gunit.h"

namespace clif {

class MyData {
 public:
  int a_, b_, c_;
};

class SharedPtrTest : public ::testing::Test {
};

TEST_F(SharedPtrTest, TestCreationFromRawPointerOwn) {
  SharedPtr<MyData> csp1(new MyData, OwnedResource());

  std::unique_ptr<MyData> up1 = MakeStdUnique(&csp1);
  EXPECT_TRUE(up1);
  EXPECT_FALSE(csp1);
  EXPECT_TRUE(csp1 == nullptr);

  SharedPtr<MyData> csp2(up1.release(), OwnedResource());
  std::shared_ptr<MyData> sp = MakeStdShared(csp2);
  std::unique_ptr<MyData> up2 = MakeStdUnique(&csp2);
  EXPECT_FALSE(up2);
  EXPECT_TRUE(csp2);
  EXPECT_TRUE(sp);
  EXPECT_TRUE(csp2 != nullptr);
}

TEST_F(SharedPtrTest, TestCreationFromRawPointerNotOwn) {
  std::unique_ptr<MyData> up(new MyData);
  SharedPtr<MyData> csp1(up.get(),  UnOwnedResource());

  std::unique_ptr<MyData> up1 = MakeStdUnique(&csp1);
  EXPECT_FALSE(up1);
  EXPECT_TRUE(csp1);

  std::shared_ptr<MyData> sp = MakeStdShared(csp1);
  std::unique_ptr<MyData> up2 = MakeStdUnique(&csp1);
  EXPECT_FALSE(up2);
  EXPECT_TRUE(csp1);
  EXPECT_TRUE(sp);
}

TEST_F(SharedPtrTest, TestCreationFromUniquePointer) {
  std::unique_ptr<MyData> up(new MyData);
  SharedPtr<MyData> csp1(std::move(up));

  EXPECT_FALSE(up);

  std::unique_ptr<MyData> up1 = MakeStdUnique(&csp1);
  EXPECT_TRUE(up1);
  EXPECT_FALSE(csp1);

  SharedPtr<MyData> csp2(move(up1));
  std::shared_ptr<MyData> sp = MakeStdShared(csp2);
  std::unique_ptr<MyData> up2 = MakeStdUnique(&csp2);
  EXPECT_FALSE(up2);
  EXPECT_TRUE(csp2);
  EXPECT_TRUE(sp);
}

TEST_F(SharedPtrTest, TestCreationFromSharedPointer) {
  std::shared_ptr<MyData> sp1(new MyData);
  SharedPtr<MyData> csp1(sp1);

  EXPECT_TRUE(sp1);
  EXPECT_TRUE(csp1);

  std::unique_ptr<MyData> up1 = MakeStdUnique(&csp1);
  EXPECT_FALSE(up1);
  EXPECT_TRUE(sp1);
  EXPECT_TRUE(csp1);

  std::shared_ptr<MyData> sp2 = MakeStdShared(csp1);
  std::unique_ptr<MyData> up2 = MakeStdUnique(&csp1);
  EXPECT_FALSE(up2);
  EXPECT_TRUE(csp1);
  EXPECT_TRUE(sp2);
}

}  // namespace clif
