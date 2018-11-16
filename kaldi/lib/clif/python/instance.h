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
#ifndef CLIF_PYTHON_INSTANCE_H_
#define CLIF_PYTHON_INSTANCE_H_

#include <cstddef>
#include <functional>
#include <memory>

namespace clif {

// This class and the class UnOwnedResource are required so that two different
// overloads of the Instance constructor, reflecting the respective object
// ownership, can be defined. Since Instance is a template class, if one does
// not ever use the OwnedResource version for a particular object type (the
// template type of the Instance), it does not get emitted by the compiler.
// Hence, if the object type does not have a public destructor, this avoids
// instantiation of a "real" deleter for the Instance.
class OwnedResource {
};

// See comment above.
class UnOwnedResource {
};

// An Instance class manage the user object instance. It contained in the
// Python object wrapper created by CLIF. The instance is stored in a
// std::shared_ptr, but with an added notion of ownership of the pointee.
// This helps in renouncing the ownership when possible (for example, creating
// a std::unique_ptr having a default deleter from this Instance).
//
// The Instance does not own the object if:
//
// a) It was created from a raw pointer passed to Python
// b) It was created from a shared pointer (std::shared_ptr) passed to Python.
//
// In all other cases, the Instance owns the pointee.
template <typename T>
class Instance {
 public:
  Instance() : maybe_deleter_(nullptr) {}

  // Creates an Instance that owns the data pointed to by |data|.
  Instance(T* data, OwnedResource res) : Instance(std::unique_ptr<T>(data)) {}

  // Creates an Instance that does not own the data pointed to by |data|.
  Instance(T* data, UnOwnedResource res)
      : Instance(std::shared_ptr<T>(data, NopDeleter())) {}

  // Captures ownership of the pointee.
  explicit Instance(std::unique_ptr<T> unique) {
    // Create a shared MaybeDeleter that will be used by the
    // custom deleter in `shared_`.  Save a pointer to it in
    // `maybe_deleter_` in case Detach() is called.
    auto maybe_deleter = std::make_shared<MaybeDeleter>();
    maybe_deleter_ = maybe_deleter.get();
    ptr_ = std::shared_ptr<T>(unique.release(),
                              SharedMaybeDeleter(std::move(maybe_deleter)));
  }

  // Creates an Instance that is essentially a copy of |shared|.
  // The new Instance does not own the pointee.
  explicit Instance(std::shared_ptr<T> shared)
      : ptr_(std::move(shared)), maybe_deleter_(nullptr) {
    // When do not own the pointee and hence `maybe_deleter_` is null.
  }

  // Does not take ownership of the pointee when an Instance is created from an
  // unique pointer having non-default deleter.
  template <typename D>
  explicit Instance(std::unique_ptr<T, D> unique)
      : Instance(std::shared_ptr<T>(unique)) {}

  // An Instance is not copyable, but movable.
  Instance(const Instance& other) = default;
  Instance& operator=(const Instance& other) = default;
  Instance(Instance&& other) = default;
  Instance& operator=(Instance&& other) = default;

  T* get() const { return ptr_.get(); }

  T& operator*() const { return *ptr_; }

  T* operator->() const { return ptr_.operator->(); }

  explicit operator bool() const { return ptr_.operator bool(); }

  bool operator==(std::nullptr_t n) const { return ptr_ == n; }

  bool operator!=(std::nullptr_t n) const { return ptr_ != n; }

  // Returns true if the ownership of the contained pointer could be renounced
  // successfully.
  bool Detach() {
    return Renounce() != nullptr;
  }

  // Nice way to call object dtor on tp_dealloc (which needs to invalidate the
  // holded object without calling the dtor (it may be called afterwards).
  // At this point we don't care about deleter_ and shared_ might be
  // already NULL.
  void Destruct() { ptr_.reset(); }

 private:
  // Will give up ownership of the pointee if the pointee is a) owned by this
  // Instance, and b) if this Instance is the only pointer to the pointee.
  //
  // The raw pointer to the pointee is returned. Returns nullptr if the pointee
  // cannot be disowned safely.
  T *Renounce() {
    // Implementation note:
    //
    // Condition (a) is satisfied by a non-null maybe_deleter_.  If
    // the deleter is null, then this Instance doesn't can't disown
    // the pointee.
    //
    // Condition (b) is satisfied by ptr_.use_count() == 1.  This is
    // believed correct, but is quite subtle.
    //
    // A use count of zero implies a null ptr_, and a use count of >1
    // implies other copies of this Instance<T>.  In either case,
    // disowning the pointer is not possible.
    //
    // The tricky part comes from the std::weak_ptr<T>::lock() API.
    // If some other thread has a weak_ptr copy of Instance::ptr_,
    // then it could race with this code.  This class does support
    // construction directly from "foreign" std::shared_ptr<T>, so
    // weak_ptr copies of ptr_ are possible.  But, in this case
    // maybe_deleter_ is always null (see the constructor that accepts
    // a shared_ptr<T>), so we can claim that a non-null
    // maybe_deleter_ implies that ptr_ has no weak_ptr copies.
    if (ptr_.use_count() == 1 && maybe_deleter_ != nullptr) {
      maybe_deleter_->Disable();
      maybe_deleter_ = nullptr;
      T* obj = ptr_.get();
      ptr_.reset();
      return obj;
    }
    return nullptr;
  }

  // Provides a MaybeDelete(T*) function that may be disabled
  // (into a no-op) at some future point.
  class MaybeDeleter {
   public:
    // Constructs the deleter in the enabled state.
    MaybeDeleter() = default;

    // Not copyable or assignable.
    MaybeDeleter(const MaybeDeleter& other) = delete;
    MaybeDeleter& operator=(const MaybeDeleter& other) = delete;

    // If enabled, deletes obj.
    void MaybeDelete(T* obj) {
      if (enabled_) {
        delete obj;
      }
    }

    // Disables this deleter.  May be called only once.
    void Disable() { enabled_ = false; }

   private:
    bool enabled_ = true;
  };

  // Provides an operator()(T*) that forwards to
  // delegate->MaybeDelete(T*).  All copies will share the same
  // delegate.  This is used as a custom std::shared_ptr deleter.
  class SharedMaybeDeleter {
   public:
    explicit SharedMaybeDeleter(std::shared_ptr<MaybeDeleter> delegate)
        : delegate_(std::move(delegate)) {}

    // Copyable and assignable.
    SharedMaybeDeleter(const SharedMaybeDeleter& other) = default;
    SharedMaybeDeleter(SharedMaybeDeleter&& other) = default;
    SharedMaybeDeleter& operator=(const SharedMaybeDeleter& other) = default;
    SharedMaybeDeleter& operator=(SharedMaybeDeleter&& other) = default;

    // Calls delegate->MaybeDelete(obj).
    void operator()(T* obj) { delegate_->MaybeDelete(obj); }

   private:
    std::shared_ptr<MaybeDeleter> delegate_;
  };

  // Provides an operator()(T*) that does nothing.  Intended for use
  // as a std::shared_ptr deleter when the shared_ptr doesn't actually
  // own the object.
  struct NopDeleter {
    // An operator()(T*) that does nothing.
    void operator()(T* obj) {}
  };

  template <typename X>
  friend std::shared_ptr<X> MakeStdShared(const Instance<X>& cpp);

  template <typename X>
  friend std::unique_ptr<X> MakeStdUnique(Instance<X>* cpp);

  // The actual user object.
  std::shared_ptr<T> ptr_;

  // Valid only if shared_ != nullptr.  If not null, points to a
  // MaybeDeleter mutually owned by all copies of `shared_`.  See
  // SharedMaybeDeleter and the Instance constructors that use it.
  MaybeDeleter* maybe_deleter_;
};

// Returns the std::shared_ptr encapsulated in |cpp|.
template <typename T>
std::shared_ptr<T> MakeStdShared(const Instance<T>& cpp) {
  return cpp.ptr_;
}

// Returns an aliasing std::shared_ptr sharing resources with |cpp|.
template <typename Y, typename T>
std::shared_ptr<T> MakeStdShared(const Instance<Y>& cpp, T* alias) {
  return std::shared_ptr<T>(MakeStdShared(cpp), alias);
}

// Creates a unique pointer pointing to the data pointed to by |cpp|.
// If |cpp| cannot renounce its pointee safely, then nullptr is returned (via
// the unique pointer.) If |cpp| renounces the pointee
// successfully, then it is set to nullptr.
template <typename T>
std::unique_ptr<T> MakeStdUnique(Instance<T>* cpp) {
  return std::unique_ptr<T>(cpp->Renounce());
}

// Creates a manager which owns the new object.
template <typename T, typename... Args>
Instance<T> MakeShared(Args&&... args) {
  return Instance<T>(new T(std::forward<Args>(args)...), OwnedResource());
}

}  // namespace clif

#endif  // CLIF_PYTHON_INSTANCE_H_
