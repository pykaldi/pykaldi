// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Float weight set and associated semiring operation definitions.

#ifndef FST_LIB_FLOAT_WEIGHT_H_
#define FST_LIB_FLOAT_WEIGHT_H_

#include <climits>
#include <cmath>
#include <cstdlib>

#include <limits>
#include <sstream>
#include <string>

#include <fst/util.h>
#include <fst/weight.h>


namespace fst {

// Numeric limits class.
template <class T>
class FloatLimits {
 public:
  static constexpr T PosInfinity() {
    return std::numeric_limits<T>::infinity();
  }

  static constexpr T NegInfinity() { return -PosInfinity(); }

  static constexpr T NumberBad() { return std::numeric_limits<T>::quiet_NaN(); }
};

// Weight class to be templated on floating-points types.
template <class T = float>
class FloatWeightTpl {
 public:
  using ValueType = T;

  FloatWeightTpl() {}

  FloatWeightTpl(T f) : value_(f) {}

  FloatWeightTpl(const FloatWeightTpl<T> &weight) : value_(weight.value_) {}

  FloatWeightTpl<T> &operator=(const FloatWeightTpl<T> &weight) {
    value_ = weight.value_;
    return *this;
  }

  std::istream &Read(std::istream &strm) { return ReadType(strm, &value_); }

  std::ostream &Write(std::ostream &strm) const {
    return WriteType(strm, value_);
  }

  size_t Hash() const {
    union {
      T f;
      size_t s;
    } u;
    u.s = 0;
    u.f = value_;
    return u.s;
  }

  const T &Value() const { return value_; }

 protected:
  void SetValue(const T &f) { value_ = f; }

  static constexpr const char *GetPrecisionString() {
    return sizeof(T) == 4
               ? ""
               : sizeof(T) == 1
                     ? "8"
                     : sizeof(T) == 2 ? "16"
                                      : sizeof(T) == 8 ? "64" : "unknown";
  }

 private:
  T value_;
};

// Single-precision float weight.
using FloatWeight = FloatWeightTpl<float>;

template <class T>
inline bool operator==(const FloatWeightTpl<T> &w1,
                       const FloatWeightTpl<T> &w2) {
  // Volatile qualifier thwarts over-aggressive compiler optimizations that
  // lead to problems esp. with NaturalLess().
  volatile T v1 = w1.Value();
  volatile T v2 = w2.Value();
  return v1 == v2;
}

// inline bool operator==(const FloatWeightTpl<double> &w1,
//                        const FloatWeightTpl<double> &w2) {
//   return operator==<double>(w1, w2);
// }
//
// inline bool operator==(const FloatWeightTpl<float> &w1,
//                        const FloatWeightTpl<float> &w2) {
//   return operator==<float>(w1, w2);
// }

template <class T>
inline bool operator!=(const FloatWeightTpl<T> &w1,
                       const FloatWeightTpl<T> &w2) {
  return !(w1 == w2);
}

// inline bool operator!=(const FloatWeightTpl<double> &w1,
//                        const FloatWeightTpl<double> &w2) {
//   return operator!=<double>(w1, w2);
// }
//
// inline bool operator!=(const FloatWeightTpl<float> &w1,
//                        const FloatWeightTpl<float> &w2) {
//   return operator!=<float>(w1, w2);
// }

template <class T>
inline bool ApproxEqual(const FloatWeightTpl<T> &w1,
                        const FloatWeightTpl<T> &w2, float delta = kDelta) {
  return w1.Value() <= w2.Value() + delta && w2.Value() <= w1.Value() + delta;
}

template <class T>
inline std::ostream &operator<<(std::ostream &strm,
                                const FloatWeightTpl<T> &w) {
  if (w.Value() == FloatLimits<T>::PosInfinity()) {
    return strm << "Infinity";
  } else if (w.Value() == FloatLimits<T>::NegInfinity()) {
    return strm << "-Infinity";
  } else if (w.Value() != w.Value()) {  // Fails for NaN.
    return strm << "BadNumber";
  } else {
    return strm << w.Value();
  }
}

template <class T>
inline std::istream &operator>>(std::istream &strm, FloatWeightTpl<T> &w) {
  string s;
  strm >> s;
  if (s == "Infinity") {
    w = FloatWeightTpl<T>(FloatLimits<T>::PosInfinity());
  } else if (s == "-Infinity") {
    w = FloatWeightTpl<T>(FloatLimits<T>::NegInfinity());
  } else {
    char *p;
    T f = strtod(s.c_str(), &p);
    if (p < s.c_str() + s.size()) {
      strm.clear(std::ios::badbit);
    } else {
      w = FloatWeightTpl<T>(f);
    }
  }
  return strm;
}

// Tropical semiring: (min, +, inf, 0).
template <class T>
class TropicalWeightTpl : public FloatWeightTpl<T> {
 public:
  using typename FloatWeightTpl<T>::ValueType;
  using FloatWeightTpl<T>::Value;
  using ReverseWeight = TropicalWeightTpl<T>;

  constexpr TropicalWeightTpl() : FloatWeightTpl<T>() {}

  constexpr TropicalWeightTpl(T f) : FloatWeightTpl<T>(f) {}

  constexpr TropicalWeightTpl(const TropicalWeightTpl<T> &weight)
      : FloatWeightTpl<T>(weight) {}

  static const TropicalWeightTpl<T> &Zero() {
    static const TropicalWeightTpl zero(FloatLimits<T>::PosInfinity());
    return zero;
  }

  static const TropicalWeightTpl<T> &One() {
    static const TropicalWeightTpl one(0.0F);
    return one;
  }

  static const TropicalWeightTpl<T> &NoWeight() {
    static const TropicalWeightTpl no_weight(FloatLimits<T>::NumberBad());
    return no_weight;
  }

  static const string &Type() {
    static const string type =
        string("tropical") + FloatWeightTpl<T>::GetPrecisionString();
    return type;
  }

  bool Member() const {
    // First part fails for IEEE NaN.
    return Value() == Value() && Value() != FloatLimits<T>::NegInfinity();
  }

  TropicalWeightTpl<T> Quantize(float delta = kDelta) const {
    if (Value() == FloatLimits<T>::NegInfinity() ||
        Value() == FloatLimits<T>::PosInfinity() || Value() != Value()) {
      return *this;
    } else {
      return TropicalWeightTpl<T>(floor(Value() / delta + 0.5F) * delta);
    }
  }

  TropicalWeightTpl<T> Reverse() const { return *this; }

  static constexpr uint64 Properties() {
    return kLeftSemiring | kRightSemiring | kCommutative | kPath | kIdempotent;
  }
};

// Single precision tropical weight.
using TropicalWeight = TropicalWeightTpl<float>;

template <class T>
inline TropicalWeightTpl<T> Plus(const TropicalWeightTpl<T> &w1,
                                 const TropicalWeightTpl<T> &w2) {
  if (!w1.Member() || !w2.Member()) return TropicalWeightTpl<T>::NoWeight();
  return w1.Value() < w2.Value() ? w1 : w2;
}

// inline TropicalWeightTpl<float> Plus(const TropicalWeightTpl<float> &w1,
//                                      const TropicalWeightTpl<float> &w2) {
//   return Plus<float>(w1, w2);
// }
//
// inline TropicalWeightTpl<double> Plus(const TropicalWeightTpl<double> &w1,
//                                       const TropicalWeightTpl<double> &w2) {
//   return Plus<double>(w1, w2);
// }

template <class T>
inline TropicalWeightTpl<T> Times(const TropicalWeightTpl<T> &w1,
                                  const TropicalWeightTpl<T> &w2) {
  if (!w1.Member() || !w2.Member()) return TropicalWeightTpl<T>::NoWeight();
  T f1 = w1.Value(), f2 = w2.Value();
  if (f1 == FloatLimits<T>::PosInfinity()) {
    return w1;
  } else if (f2 == FloatLimits<T>::PosInfinity()) {
    return w2;
  } else {
    return TropicalWeightTpl<T>(f1 + f2);
  }
}

// inline TropicalWeightTpl<float> Times(const TropicalWeightTpl<float> &w1,
//                                       const TropicalWeightTpl<float> &w2) {
//   return Times<float>(w1, w2);
// }
//
// inline TropicalWeightTpl<double> Times(const TropicalWeightTpl<double> &w1,
//                                        const TropicalWeightTpl<double> &w2) {
//   return Times<double>(w1, w2);
// }

template <class T>
inline TropicalWeightTpl<T> Divide(const TropicalWeightTpl<T> &w1,
                                   const TropicalWeightTpl<T> &w2,
                                   DivideType typ = DIVIDE_ANY) {
  if (!w1.Member() || !w2.Member()) return TropicalWeightTpl<T>::NoWeight();
  T f1 = w1.Value(), f2 = w2.Value();
  if (f2 == FloatLimits<T>::PosInfinity()) {
    return FloatLimits<T>::NumberBad();
  } else if (f1 == FloatLimits<T>::PosInfinity()) {
    return FloatLimits<T>::PosInfinity();
  } else {
    return TropicalWeightTpl<T>(f1 - f2);
  }
}

// inline TropicalWeightTpl<float> Divide(const TropicalWeightTpl<float> &w1,
//                                        const TropicalWeightTpl<float> &w2,
//                                        DivideType typ = DIVIDE_ANY) {
//   return Divide<float>(w1, w2, typ);
// }
//
// inline TropicalWeightTpl<double> Divide(const TropicalWeightTpl<double> &w1,
//                                         const TropicalWeightTpl<double> &w2,
//                                         DivideType typ = DIVIDE_ANY) {
//   return Divide<double>(w1, w2, typ);
// }

template <class T>
inline TropicalWeightTpl<T> Power(const TropicalWeightTpl<T> &weight,
                                  T scalar) {
  return TropicalWeightTpl<T>(weight.Value() * scalar);
}

// Log semiring: (log(e^-x + e^-y), +, inf, 0).
template <class T>
class LogWeightTpl : public FloatWeightTpl<T> {
 public:
  using typename FloatWeightTpl<T>::ValueType;
  using FloatWeightTpl<T>::Value;
  using ReverseWeight = LogWeightTpl;

  constexpr LogWeightTpl() : FloatWeightTpl<T>() {}

  constexpr LogWeightTpl(T f) : FloatWeightTpl<T>(f) {}

  constexpr LogWeightTpl(const LogWeightTpl<T> &weight)
      : FloatWeightTpl<T>(weight) {}

  static const LogWeightTpl &Zero() {
    static const LogWeightTpl zero(FloatLimits<T>::PosInfinity());
    return zero;
  }

  static const LogWeightTpl &One() {
    static const LogWeightTpl one(0.0F);
    return one;
  }

  static const LogWeightTpl &NoWeight() {
    static const LogWeightTpl no_weight(FloatLimits<T>::NumberBad());
    return no_weight;
  }

  static const string &Type() {
    static const string type =
        string("log") + FloatWeightTpl<T>::GetPrecisionString();
    return type;
  }

  bool Member() const {
    // First part fails for IEEE NaN.
    return Value() == Value() && Value() != FloatLimits<T>::NegInfinity();
  }

  LogWeightTpl<T> Quantize(float delta = kDelta) const {
    if (Value() == FloatLimits<T>::NegInfinity() ||
        Value() == FloatLimits<T>::PosInfinity() || Value() != Value()) {
      return *this;
    } else {
      return LogWeightTpl<T>(floor(Value() / delta + 0.5F) * delta);
    }
  }

  LogWeightTpl<T> Reverse() const { return *this; }

  static constexpr uint64 Properties() {
    return kLeftSemiring | kRightSemiring | kCommutative;
  }
};

// Single-precision log weight.
using LogWeight = LogWeightTpl<float>;

// Double-precision log weight.
using Log64Weight = LogWeightTpl<double>;

namespace internal {

// -log(e^-x + e^-y) = x - LogPosExp(y - x)
// Assumes x >= 0.0.
inline double LogPosExp(double x) {
  return std::log(1.0 + std::exp(-x));
}

// -log(e^-x - e^-y) = x - LogNegExp(y - x)
// Assumes x >= 0.0.
inline double LogNegExp(double x) {
  return std::log(1.0 - std::exp(-x));
}

// Alternative LogPosExp that is more accurate for large x.
// Assumes x >= 0.0.
inline double AltLogPosExp(double x) {
  double y = std::exp(-x);
  if (y > kDelta) {
    return std::log(1.0 + y);
  } else {
    // Mercator series
    double y2 = y * y;
    double y3 = y2 * y;
    double y4 = y2 * y2;
    return y - y2/2.0 + y3/3.0 - y4/4.0;
  }
}

// Alternative LogNegExp that is more accurate for large x.
// Assumes x > 0.0.
inline double AltLogNegExp(double x) {
  double y = std::exp(-x);
  if (y > kDelta) {
    return std::log(1.0 - y);
  } else {
    // Mercator series
    double y2 = y * y;
    double y3 = y2 * y;
    double y4 = y2 * y2;
    return -y - y2/2.0 - y3/3.0 - y4/4.0;
  }
}

// a +_log b = -log(e^-a + e^-b) = KahanLogSum(a, b, ...).
// Kahan compensated summation provides an error bound that is
// independent of the number of addends. Assumes b >= a;
// c is the compensation.
inline double KahanLogSum(double a, double b, double *c) {
  double y = -AltLogPosExp(b - a) - *c;
  double t = a + y;
  *c = (t - a) - y;
  return t;
}

// a -_log b = -log(e^-a - e^-b) = KahanLogDiff(a, b, ...).
// Kahan compensated summation provides an error bound that is
// independent of the number of addends. Assumes b > a;
// c is the compensation.
inline double KahanLogDiff(double a, double b, double *c) {
  double y = -AltLogNegExp(b - a) - *c;
  double t = a + y;
  *c = (t - a) - y;
  return t;
}

}  // namespace internal

template <class T>
inline LogWeightTpl<T> Plus(const LogWeightTpl<T> &w1,
                            const LogWeightTpl<T> &w2) {
  T f1 = w1.Value(), f2 = w2.Value();
  if (f1 == FloatLimits<T>::PosInfinity()) {
    return w2;
  } else if (f2 == FloatLimits<T>::PosInfinity()) {
    return w1;
  } else if (f1 > f2) {
    return LogWeightTpl<T>(f2 - internal::LogPosExp(f1 - f2));
  } else {
    return LogWeightTpl<T>(f1 - internal::LogPosExp(f2 - f1));
  }
}

// inline LogWeightTpl<float> Plus(const LogWeightTpl<float> &w1,
//                                 const LogWeightTpl<float> &w2) {
//   return Plus<float>(w1, w2);
// }
//
// inline LogWeightTpl<double> Plus(const LogWeightTpl<double> &w1,
//                                  const LogWeightTpl<double> &w2) {
//   return Plus<double>(w1, w2);
// }

template <class T>
inline LogWeightTpl<T> Times(const LogWeightTpl<T> &w1,
                             const LogWeightTpl<T> &w2) {
  if (!w1.Member() || !w2.Member()) return LogWeightTpl<T>::NoWeight();
  T f1 = w1.Value(), f2 = w2.Value();
  if (f1 == FloatLimits<T>::PosInfinity()) {
    return w1;
  } else if (f2 == FloatLimits<T>::PosInfinity()) {
    return w2;
  } else {
    return LogWeightTpl<T>(f1 + f2);
  }
}

// inline LogWeightTpl<float> Times(const LogWeightTpl<float> &w1,
//                                  const LogWeightTpl<float> &w2) {
//   return Times<float>(w1, w2);
// }
//
// inline LogWeightTpl<double> Times(const LogWeightTpl<double> &w1,
//                                   const LogWeightTpl<double> &w2) {
//   return Times<double>(w1, w2);
// }

template <class T>
inline LogWeightTpl<T> Divide(const LogWeightTpl<T> &w1,
                              const LogWeightTpl<T> &w2,
                              DivideType typ = DIVIDE_ANY) {
  if (!w1.Member() || !w2.Member()) return LogWeightTpl<T>::NoWeight();
  T f1 = w1.Value(), f2 = w2.Value();
  if (f2 == FloatLimits<T>::PosInfinity()) {
    return FloatLimits<T>::NumberBad();
  } else if (f1 == FloatLimits<T>::PosInfinity()) {
    return FloatLimits<T>::PosInfinity();
  } else {
    return LogWeightTpl<T>(f1 - f2);
  }
}

// inline LogWeightTpl<float> Divide(const LogWeightTpl<float> &w1,
//                                   const LogWeightTpl<float> &w2,
//                                   DivideType typ = DIVIDE_ANY) {
//   return Divide<float>(w1, w2, typ);
// }
//
// inline LogWeightTpl<double> Divide(const LogWeightTpl<double> &w1,
//                                    const LogWeightTpl<double> &w2,
//                                    DivideType typ = DIVIDE_ANY) {
//   return Divide<double>(w1, w2, typ);
// }

template <class T>
inline LogWeightTpl<T> Power(const LogWeightTpl<T> &weight, T scalar) {
  return LogWeightTpl<T>(weight.Value() * scalar);
}

// Specialization using the Kahan compensated summation
template <class T>
class Adder<LogWeightTpl<T>> {
 public:
  using Weight = LogWeightTpl<T>;

  explicit Adder(Weight w = Weight::Zero())
      : sum_(w.Value()),
        c_(0.0) { }

  Weight Add(const Weight &w) {
    T f = w.Value();
    if (f == FloatLimits<T>::PosInfinity()) {
      return Sum();
    } else if (sum_ == FloatLimits<T>::PosInfinity()) {
      sum_ = f;
      c_ = 0.0;
    } else if (f > sum_) {
      sum_ = internal::KahanLogSum(sum_, f, &c_);
    } else {
      sum_ = internal::KahanLogSum(f, sum_, &c_);
    }
    return Sum();
  }

  Weight Sum() { return Weight(sum_); }

  void Reset(Weight w = Weight::Zero()) {
    sum_ = w.Value();
    c_ = 0.0;
  }

 private:
  double sum_;
  double c_;   // Kahan compensation
};

// MinMax semiring: (min, max, inf, -inf).
template <class T>
class MinMaxWeightTpl : public FloatWeightTpl<T> {
 public:
  using typename FloatWeightTpl<T>::ValueType;
  using FloatWeightTpl<T>::Value;

  using ReverseWeight = MinMaxWeightTpl<T>;

  MinMaxWeightTpl() : FloatWeightTpl<T>() {}

  MinMaxWeightTpl(T f) : FloatWeightTpl<T>(f) {}

  MinMaxWeightTpl(const MinMaxWeightTpl<T> &weight)
      : FloatWeightTpl<T>(weight) {}

  static const MinMaxWeightTpl &Zero() {
    static const MinMaxWeightTpl zero(FloatLimits<T>::PosInfinity());
    return zero;
  }

  static const MinMaxWeightTpl &One() {
    static const MinMaxWeightTpl one(FloatLimits<T>::NegInfinity());
    return one;
  }

  static const MinMaxWeightTpl &NoWeight() {
    static const MinMaxWeightTpl no_weight(FloatLimits<T>::NumberBad());
    return no_weight;
  }

  static const string &Type() {
    static const string type =
        string("minmax") + FloatWeightTpl<T>::GetPrecisionString();
    return type;
  }

  bool Member() const {
    // Fails for IEEE NaN
    return Value() == Value();
  }

  MinMaxWeightTpl<T> Quantize(float delta = kDelta) const {
    // If one of infinities, or a NaN
    if (Value() == FloatLimits<T>::NegInfinity() ||
        Value() == FloatLimits<T>::PosInfinity() || Value() != Value()) {
      return *this;
    } else {
      return MinMaxWeightTpl<T>(floor(Value() / delta + 0.5F) * delta);
    }
  }

  MinMaxWeightTpl<T> Reverse() const { return *this; }

  static constexpr uint64 Properties() {
    return kLeftSemiring | kRightSemiring | kCommutative | kIdempotent | kPath;
  }
};

// Single-precision min-max weight.
using MinMaxWeight = MinMaxWeightTpl<float>;

// Min.
template <class T>
inline MinMaxWeightTpl<T> Plus(const MinMaxWeightTpl<T> &w1,
                               const MinMaxWeightTpl<T> &w2) {
  if (!w1.Member() || !w2.Member()) return MinMaxWeightTpl<T>::NoWeight();
  return w1.Value() < w2.Value() ? w1 : w2;
}

inline MinMaxWeightTpl<float> Plus(const MinMaxWeightTpl<float> &w1,
                                   const MinMaxWeightTpl<float> &w2) {
  return Plus<float>(w1, w2);
}

inline MinMaxWeightTpl<double> Plus(const MinMaxWeightTpl<double> &w1,
                                    const MinMaxWeightTpl<double> &w2) {
  return Plus<double>(w1, w2);
}

// Max.
template <class T>
inline MinMaxWeightTpl<T> Times(const MinMaxWeightTpl<T> &w1,
                                const MinMaxWeightTpl<T> &w2) {
  if (!w1.Member() || !w2.Member()) return MinMaxWeightTpl<T>::NoWeight();
  return w1.Value() >= w2.Value() ? w1 : w2;
}

inline MinMaxWeightTpl<float> Times(const MinMaxWeightTpl<float> &w1,
                                    const MinMaxWeightTpl<float> &w2) {
  return Times<float>(w1, w2);
}

inline MinMaxWeightTpl<double> Times(const MinMaxWeightTpl<double> &w1,
                                     const MinMaxWeightTpl<double> &w2) {
  return Times<double>(w1, w2);
}

// Defined only for special cases.
template <class T>
inline MinMaxWeightTpl<T> Divide(const MinMaxWeightTpl<T> &w1,
                                 const MinMaxWeightTpl<T> &w2,
                                 DivideType typ = DIVIDE_ANY) {
  if (!w1.Member() || !w2.Member()) return MinMaxWeightTpl<T>::NoWeight();
  // min(w1, x) = w2, w1 >= w2 => min(w1, x) = w2, x = w2.
  return w1.Value() >= w2.Value() ? w1 : FloatLimits<T>::NumberBad();
}

inline MinMaxWeightTpl<float> Divide(const MinMaxWeightTpl<float> &w1,
                                     const MinMaxWeightTpl<float> &w2,
                                     DivideType typ = DIVIDE_ANY) {
  return Divide<float>(w1, w2, typ);
}

inline MinMaxWeightTpl<double> Divide(const MinMaxWeightTpl<double> &w1,
                                      const MinMaxWeightTpl<double> &w2,
                                      DivideType typ = DIVIDE_ANY) {
  return Divide<double>(w1, w2, typ);
}

// Converts to tropical.
template <>
struct WeightConvert<LogWeight, TropicalWeight> {
  TropicalWeight operator()(const LogWeight &w) const { return w.Value(); }
};

inline std::function<TropicalWeight(const LogWeight&)> GetLogToTropicalConverter() {
  return WeightConvert<LogWeight, TropicalWeight>();
}

template <>
struct WeightConvert<Log64Weight, TropicalWeight> {
  TropicalWeight operator()(const Log64Weight &w) const { return w.Value(); }
};

// Converts to log.
template <>
struct WeightConvert<TropicalWeight, LogWeight> {
  LogWeight operator()(const TropicalWeight &w) const { return w.Value(); }
};

inline std::function<LogWeight(const TropicalWeight&)> GetTropicalToLogConverter() {
  return WeightConvert<TropicalWeight, LogWeight>();
}


template <>
struct WeightConvert<Log64Weight, LogWeight> {
  LogWeight operator()(const Log64Weight &w) const { return w.Value(); }
};

// Converts to log64.
template <>
struct WeightConvert<TropicalWeight, Log64Weight> {
  Log64Weight operator()(const TropicalWeight &w) const { return w.Value(); }
};

template <>
struct WeightConvert<LogWeight, Log64Weight> {
  Log64Weight operator()(const LogWeight &w) const { return w.Value(); }
};

// This function object returns random integers chosen from [0,
// num_random_weights). The boolean 'allow_zero' determines whether Zero() and
// zero divisors should be returned in the random weight generation. This is
// intended primary for testing.
template <class Weight>
class FloatWeightGenerate {
 public:
  explicit FloatWeightGenerate(
      bool allow_zero = true,
      const size_t num_random_weights = kNumRandomWeights)
      : allow_zero_(allow_zero), num_random_weights_(num_random_weights) {}

  Weight operator()() const {
    int n = rand() % (num_random_weights_ + allow_zero_);  // NOLINT
    if (allow_zero_ && n == num_random_weights_) return Weight::Zero();
    return Weight(n);
  }

 private:
  // Permits Zero() and zero divisors.
  bool allow_zero_;
  // Number of alternative random weights.
  const size_t num_random_weights_;
};

template <class T>
class WeightGenerate<TropicalWeightTpl<T>>
    : public FloatWeightGenerate<TropicalWeightTpl<T>> {
 public:
  using Weight = TropicalWeightTpl<T>;
  using Generate = FloatWeightGenerate<Weight>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t num_random_weights = kNumRandomWeights)
      : Generate(allow_zero, num_random_weights) {}

  Weight operator()() const { return Weight(Generate::operator()()); }
};

template <class T>
class WeightGenerate<LogWeightTpl<T>>
    : public FloatWeightGenerate<LogWeightTpl<T>> {
 public:
  using Weight = LogWeightTpl<T>;
  using Generate = FloatWeightGenerate<Weight>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t num_random_weights = kNumRandomWeights)
      : Generate(allow_zero, num_random_weights) {}

  Weight operator()() const { return Weight(Generate::operator()()); }
};

// This function object returns random integers chosen from [0,
// num_random_weights). The boolean 'allow_zero' determines whether Zero() and
// zero divisors should be returned in the random weight generation. This is
// intended primary for testing.
template <class T>
class WeightGenerate<MinMaxWeightTpl<T>> {
 public:
  using Weight = MinMaxWeightTpl<T>;

  explicit WeightGenerate(bool allow_zero = true,
                          size_t num_random_weights = kNumRandomWeights)
      : allow_zero_(allow_zero), num_random_weights_(num_random_weights) {}

  Weight operator()() const {
    int n = (rand() % (2 * num_random_weights_ + allow_zero_)) -  // NOLINT
            num_random_weights_;
    if (allow_zero_ && n == num_random_weights_) {
      return Weight::Zero();
    } else if (n == -num_random_weights_) {
      return Weight::One();
    } else {
      return Weight(n);
    }
  }

 private:
  // Permits Zero() and zero divisors.
  bool allow_zero_;
  // Number of alternative random weights.
  const size_t num_random_weights_;
};

}  // namespace fst

#endif  // FST_LIB_FLOAT_WEIGHT_H_
