#ifndef PYKALDI_UTIL_OPTIONS_EXT_H_
#define PYKALDI_UTIL_OPTIONS_EXT_H_ 1

#include "util/simple-options.h"
#include "util/parse-options.h"

namespace kaldi {

class SimpleOptionsExt : public SimpleOptions {
 public:
   bool SetOptionInt(const std::string &key, const int32 &value) {
     return SimpleOptions::SetOption(key, value);
   }

   bool SetOptionUInt(const std::string &key, const uint32 &value) {
     return SimpleOptions::SetOption(key, value);
   }

   bool SetOptionFloat(const std::string &key, const float &value) {
     return SimpleOptions::SetOption(key, value);
   }

   bool SetOptionDouble(const std::string &key, const double &value) {
     return SimpleOptions::SetOption(key, value);
   }
};

class ParseOptionsExt : public ParseOptions {
 public:
  explicit ParseOptionsExt(const std::string &usage)
      : ParseOptions(usage.c_str()) { }

  ParseOptionsExt(const std::string &prefix, OptionsItf *other)
      : ParseOptions(prefix, other) { }

  int Read(const std::vector<std::string> &argv) {
    std::vector<const char*> cargv{};
    for(const std::string& arg : argv)
      cargv.push_back(arg.c_str());
    return ParseOptions::Read(cargv.size(), cargv.data());
  }

  void PrintConfig() {
    ParseOptions::PrintConfig(std::cout);
  }
};

}  // namespace kaldi

#endif  // PYKALDI_UTIL_OPTIONS_EXT_H_
