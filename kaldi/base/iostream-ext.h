#ifndef PYKALDI_UTIL_IOSTREAM_EXT_H_
#define PYKALDI_UTIL_IOSTREAM_EXT_H_ 1

#include <iostream>
#include <sstream>
#include <string>


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

void Flush(ostream &os) {
  os.flush();
}

string Read(istream &is) {
  stringstream buffer;   // Since input stream may or may not support seeking,
  buffer << is.rdbuf();  // we read the whole stream into a stringstream and
  return buffer.str();   // return the string that the stringstream uses.
}

string ReadLine(istream &is) {
  string s;
  getline(is, s);
  if (is)                // If EOF is not reached, append '\n' to the output
    s.push_back('\n');   // string since getline discards newline characters.
  return s;
}

void Write(ostream &os, const string &s) {
  os << s;
}

}  // namespace std

#endif  // PYKALDI_UTIL_IOSTREAM_EXT_H_
