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

#include "clif/python/instance.h"
#include "testing/base/public/gunit.h"

namespace clif {

class PrivateDestructor {
 public:
  PrivateDestructor() = default;
  PrivateDestructor(const PrivateDestructor& other) = delete;
  PrivateDestructor& operator=(const PrivateDestructor& other) = delete;

  void Delete() { delete this; }

 private:
  ~PrivateDestructor() = default;
};

class MyData {
 public:
  int a_, b_, c_;
};

TEST(InstanceTest, TestCreationFromRawPointerOwn) {
  Instance<MyData> csp1(new MyData, OwnedResource());

  std::unique_ptr<MyData> up1 = MakeStdUnique(&csp1);
  EXPECT_TRUE(up1);
  EXPECT_FALSE(csp1);
  EXPECT_TRUE(csp1 == nullptr);

  Instance<MyData> csp2(up1.release(), OwnedResource());
  std::shared_ptr<MyData> sp = MakeStdShared(csp2);
  std::unique_ptr<MyData> up2 = MakeStdUnique(&csp2);
  EXPECT_FALSE(up2);
  EXPECT_TRUE(csp2);
  EXPECT_TRUE(sp);
  EXPECT_TRUE(csp2 != nullptr);
}

TEST(InstanceTest, TestCreationFromRawPointerNotOwn) {
  std::unique_ptr<MyData> up(new MyData);
  Instance<MyData> csp1(up.get(),  UnOwnedResource());

  std::unique_ptr<MyData> up1 = MakeStdUnique(&csp1);
  EXPECT_FALSE(up1);
  EXPECT_TRUE(csp1);

  std::shared_ptr<MyData> sp = MakeStdShared(csp1);
  std::unique_ptr<MyData> up2 = MakeStdUnique(&csp1);
  EXPECT_FALSE(up2);
  EXPECT_TRUE(csp1);
  EXPECT_TRUE(sp);
}

TEST(InstanceTest, TestCreateUnownedPrivateDestructpr) {
  PrivateDestructor* obj = new PrivateDestructor();
  Instance<PrivateDestructor> shared(obj, UnOwnedResource());
  EXPECT_FALSE(shared == nullptr);
  shared.Destruct();
  obj->Delete();
}

TEST(InstanceTest, TestCreationFromUniquePointer) {
  std::unique_ptr<MyData> up(new MyData);
  Instance<MyData> csp1(std::move(up));

  EXPECT_FALSE(up);

  std::unique_ptr<MyData> up1 = MakeStdUnique(&csp1);
  EXPECT_TRUE(up1);
  EXPECT_FALSE(csp1);

  Instance<MyData> csp2(move(up1));
  std::shared_ptr<MyData> sp = MakeStdShared(csp2);
  std::unique_ptr<MyData> up2 = MakeStdUnique(&csp2);
  EXPECT_FALSE(up2);
  EXPECT_TRUE(csp2);
  EXPECT_TRUE(sp);
}

TEST(InstanceTest, TestCreationFromUniquePointerWithDefaultDeleter) {
  std::unique_ptr<MyData, std::default_delete<MyData>> up(new MyData);
  EXPECT_TRUE(up);
  Instance<MyData> csp3(move(up));
  EXPECT_FALSE(up);
  EXPECT_TRUE(csp3);
}

TEST(InstanceTest, TestCreationFromSharedPointer) {
  std::shared_ptr<MyData> sp1(new MyData);
  Instance<MyData> csp1(sp1);

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
