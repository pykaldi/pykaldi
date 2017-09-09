#ifndef PYKALDI_BASE_KALDI_MATH_EXT_H_
#define PYKALDI_BASE_KALDI_MATH_EXT_H_ 1

#include "base/kaldi-math.h"
#include <cmath>

inline bool FLOAT_KALDI_ISNAN(float arg){
    return std::isnan(arg);
}

inline bool DOUBLE_KALDI_ISNAN(double arg){
    return std::isnan(arg);
}

inline bool FLOAT_KALDI_ISINF(float arg){
    return std::isinf(arg);
}

inline bool DOUBLE_KALDI_ISINF(double arg){
    return std::isinf(arg);
}

inline bool FLOAT_KALDI_ISFINITE(float arg){
    return std::isfinite(arg);
}

inline bool DOUBLE_KALDI_ISFINITE(double arg){
    return std::isfinite(arg);
}

inline float FLOAT_KALDI_SQR(float x){
    return ((x) * (x));
}

inline double DOUBLE_KALDI_SQR(double x){
    return ((x) * (x));
}

namespace kaldi {

    inline float GetkLogZeroFloat(){
        return kLogZeroFloat;
    }

    inline double GetkLogZeroDouble(){
        return kLogZeroDouble;
    }

    inline double DoubleExp(double x){
        return exp(x);
    }

    #ifndef KALDI_NO_EXPF
        inline float FloatExp(float x){
            return expf(x);
        }
    #else
        inline float FloatExp(float x){
            return exp(x);
        }
    #endif

    inline double DoubleLog(double x){
        return log(x);
    }

    inline float FloatLog(float x){
        return log(x);
    }

    inline float FloatLog1p(float x){
        const float cutoff = 1.0e-07;
        if (x < cutoff)
            return x - 2 * x * x;
        else
            return Log(1.0 + x);
    }

    inline double DoubleLog1p(double x){
        const double cutoff = 1.0e-08;
        if (x < cutoff)
            return x - 2 * x * x;
        else
            return Log(1.0 + x);
    }

    // Modified argument order to comply with clif
    inline void FloatRandGauss2(RandomState *state = NULL, float *a = nullptr, float *b = nullptr){
        RandGauss2(a, b, state);
    }

    // Modified argument order to comply with clif
    inline void DoubleRandGauss2(RandomState *state = NULL, double *a = nullptr, double *b = nullptr){
        return RandGauss2(a, b, state);
    }

    inline float FloatLogAdd(float x, float y){
        return LogAdd(x, y);
    }

    inline double DoubleLogAdd(double x, double y){
        return LogAdd(x, y);
    }

    inline float FloatLogSub(float x, float y){
        return LogSub(x, y);
    }

    inline double DoubleLogSub(double x, double y){
        return LogSub(x, y);
    }

    inline double DoubleHypot(double x, double y){
        return Hypot(x, y);
    }

    inline float FloatHypot(float x, float y){
        return Hypot(x, y);
    }

}

#endif