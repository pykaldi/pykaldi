#ifndef PYKALDI_UTIL_IOSTREAM_EXT_H_
#define PYKALDI_UTIL_IOSTREAM_EXT_H_ 1

#include <iostream>
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

// Note (VM):
// Allows the user to read lines from istream.
// While I know this is not optimal (we don't want the user to use this)
// I see no other way around to use istream after you get it from 
// some method.
string ReadLine(istream &input) {
	string output;
	getline(input, output);
	return output;
}

}  // namespace std

#endif  // PYKALDI_UTIL_IOSTREAM_EXT_H_
