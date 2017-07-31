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
#ifndef CLIF_PYTHON_SHARED_PTR_H_
#define CLIF_PYTHON_SHARED_PTR_H_

#include <cstddef>
#include <functional>
#include <memory>

namespace clif {

template <typename T>
class SharedPtr;

template <typename T>
std::shared_ptr<T> MakeStdShared(const SharedPtr<T>& sp);

template <typename T>
std::unique_ptr<T> MakeStdUnique(SharedPtr<T>* sp);

// This class and the class UnOwnedResource are required so that two different
// overloads of the SharedPtr constructor, reflecting the respective object
// ownership, can be defined. Since SharedPtr is a template class, if one does
// not ever use the OwnedResource version for a particular object type (the
// template type of the SharedPtr), it does not get emitted by the compiler.
// Hence, if the object type does not have a public destructor, this avoids
// instantiation of a "real" deleter for the SharedPtr.
class OwnedResource {
};

// See comment above.
class UnOwnedResource {
};

// A shared pointer class which is equivalent to std::shared_ptr, but with an
// added notion of ownership of the pointee. This helps in renouncing the
// ownership when possible (for example, creating a std::unique_ptr having a
// default deleter from this shared pointer).
//
// In clif, we store wrapped objects in shared pointers. These shared pointers
// are created in different ways, each having a different sense of ownership
// of the pointee object. A shared pointer does not own the object if:
//
// a) It was created from a raw pointer passed to Python
// b) It was created from a shared pointer (std::shared_ptr) passed to Python.
//
// In all other cases, the shared pointer owns the pointee.
template <typename T>
class SharedPtr {
 public:
  SharedPtr() : deleter_(UnOwnedResource()) { }

  // Creates a shared pointer owns the data pointed to by |data|.
  SharedPtr(T* data, OwnedResource res) : deleter_(res), sp_(data, deleter_) {
  }

  // Creates a shared pointer which does not own the data pointed to by |data|.
  SharedPtr(T* data, UnOwnedResource res): deleter_(res), sp_(data, deleter_) {
  }

  // Captures ownership of the pointee.
  explicit SharedPtr(std::unique_ptr<T> up)
      : deleter_(OwnedResource()), sp_(up.release(), deleter_) {
  }

  // Creates a shared pointer which is essentially a copy of |sp|.
  // The new shared pointer does not own the pointee.
  explicit SharedPtr(std::shared_ptr<T> sp)
      : deleter_(UnOwnedResource()), sp_(std::move(sp)) {
    // When do not own the pointee and hence our deleter is not used.
  }

  // Does not take ownership of the pointee when a shared ptr is created from a
  // unique pointer having non-default deleter.
  template <typename D>
  explicit SharedPtr(std::unique_ptr<T, D> up)
      : deleter_(UnOwnedResource()), sp_(up) {
  }

  T *get() const {
    return sp_.get();
  }

  T &operator*() const {
    return *sp_;
  }

  T *operator->() const {
    return sp_.operator->();
  }

  explicit operator bool() const {
    return sp_.operator bool();
  }

  bool operator==(std::nullptr_t n) const {
    return sp_ == n;
  }

  bool operator!=(std::nullptr_t n) const {
    return sp_ != n;
  }

  // Returns true if the ownership of the contained pointer could be renounced
  // successfully.
  bool Detach() {
    return Renounce() != nullptr;
  }

 private:
  // Will give up ownership of the pointee if the pointee is a) owned by this
  // shared ptr, and b) if this shared pointer is the only pointer to the
  // pointee.
  //
  // The raw pointer to the pointee is returned. Returns nullptr if the pointee
  // cannot be disowned safely.
  T *Renounce() {
    // In the clif use case, if the shared ptr was created by Python,
    // this->unique() will return false if the shared ptr was shared with C++
    // (for example, passed as an argument to a function).
    if (deleter_ && sp_.unique()) {
      deleter_.Ignore();
      T *obj = sp_.get();
      sp_.reset();

      return obj;
    }

    return nullptr;
  }

  template <typename X>
  friend std::shared_ptr<X> MakeStdShared(const SharedPtr<X>& sp);

  template <typename X>
  friend std::unique_ptr<X> MakeStdUnique(SharedPtr<X>* sp);

  static void RealDelete(T* d) {
    delete d;
  }

  static void NoopDelete(T* d) { }

  class Deleter {
   public:
    explicit Deleter(OwnedResource res)
        : del_func_(RealDelete), del_sp_(std::make_shared<bool>(true)) {
    }

    explicit Deleter(UnOwnedResource res)
        : del_func_(NoopDelete), del_sp_(std::make_shared<bool>(false)) {
    }

    operator bool() const {
      return *del_sp_.get();
    }

    void operator()(T* d) {
      if (*del_sp_.get()) {
        del_func_(d);
      }
    }

    void Ignore() {
      *del_sp_.get() = false;
    }

   private:
    std::function<void(T*)> del_func_;

    // The boolean state has to be stored in a shared ptr as we need to save a
    // copy of the deleter in SharedPtr. This way, the boolean state is shared
    // between the copies in the base and derived classes.
    std::shared_ptr<bool> del_sp_;
  };

  // The notion of pointee ownership is encapsulated in this deleter.
  // We cannot use std::get_deleter as it depends on RTTI. That is, it does not
  // work with -fno-rtti. Hence, we have to store a copy of the deleter
  // explicitly.
  Deleter deleter_;

  // The actual shared ptr
  std::shared_ptr<T> sp_;
};

// Returns the std::shared_ptr encapsulated in |sp|.
template <typename T>
std::shared_ptr<T> MakeStdShared(const SharedPtr<T>& sp) {
  return sp.sp_;
}

// Returns an aliasing std::shared_ptr sharing resources with |sp|.
template <typename Y, typename T>
std::shared_ptr<T> MakeStdShared(const SharedPtr<Y>& sp, T* alias) {
  return std::shared_ptr<T>(MakeStdShared(sp), alias);
}

// Creates a unique pointer pointing to the data pointed to by |sp|.
// If |sp| cannot renounce its pointee safely, then nullptr is returned (via
// the unique pointer.) If |sp| renounces the pointee successfully, then it is
// set to nullptr.
template <typename T>
std::unique_ptr<T> MakeStdUnique(SharedPtr<T>* sp) {
  return std::unique_ptr<T>(sp->Renounce());
}

// Creates a shared pointer which owns the new object.
template <typename T, typename... Args>
SharedPtr<T> MakeShared(Args&&... args) {
  return SharedPtr<T>(new T(std::forward<Args>(args)...), OwnedResource());
}

}  // namespace clif

#endif  // CLIF_PYTHON_SHARED_PTR_H_
