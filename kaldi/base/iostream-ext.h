#ifndef PYKALDI_UTIL_IOSTREAM_EXT_H_
#define PYKALDI_UTIL_IOSTREAM_EXT_H_ 1

#include <iostream>

namespace std {

istream *GetStdinPtr() {
  return &cin;
}

ostream *GetStdoutPtr() {
  return &cout;
}

ostream *GetStderrPtr() {
  return &cerr;
}

ostream *GetStdlogPtr() {
  return &clog;
}

}  // namespace std

#endif  // PYKALDI_UTIL_IOSTREAM_EXT_H_
