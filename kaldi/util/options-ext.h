#ifndef PYKALDI_UTIL_OPTIONS_EXT_H_
#define PYKALDI_UTIL_OPTIONS_EXT_H_ 1

#include "util/simple-options.h"
#include "util/parse-options.h"

namespace kaldi {

// NOTE(Dogan):
// We probably do not need to wrap this class. Commented out for the moment.
//
// class SimpleOptionsExt : public SimpleOptions {
//  public:
//    bool SetOptionInt(const std::string &key, const int32 &value) {
//      return SimpleOptions::SetOption(key, value);
//    }
//
//    bool SetOptionUInt(const std::string &key, const uint32 &value) {
//      return SimpleOptions::SetOption(key, value);
//    }
//
//    bool SetOptionFloat(const std::string &key, const float &value) {
//      return SimpleOptions::SetOption(key, value);
//    }
//
//    bool SetOptionDouble(const std::string &key, const double &value) {
//      return SimpleOptions::SetOption(key, value);
//    }
// };

class ParseOptionsExt : public ParseOptions {
 public:
  // This constructor accepts a usage string and moves its contents to the
  // internal member usage_. This ensures that the char pointer passed to the
  // base class stays alive as long as this ParseOptionsExt object is in scope.
  explicit ParseOptionsExt(std::string usage)
      : ParseOptions(usage.c_str()), usage_(std::move(usage)) {}

  ParseOptionsExt(const std::string &prefix, OptionsItf *other)
      : ParseOptions(prefix, other) { }

  int Read(const std::vector<std::string> &argv) {
    std::vector<const char*> cargv{};
    for(const std::string& arg : argv)
      cargv.push_back(arg.c_str());
    return ParseOptions::Read(cargv.size(), cargv.data());
  }

  struct Options {
    std::map<std::string, bool> bool_map;
    std::map<std::string, int32> int_map;
    std::map<std::string, uint32> uint_map;
    std::map<std::string, float> float_map;
    std::map<std::string, double> double_map;
    std::map<std::string, string> str_map;
  };

  Options &GetOptions() {
    return options_;
  }

  void PrintConfig() {
    ParseOptions::PrintConfig(std::cout);
  }

  void NormalizeOptionName(std::string *str) {
    std::string out;
    std::string::iterator it;

    for (it = str->begin(); it != str->end(); ++it) {
      if (*it == '-')
        out += '_';  // convert - to _
      else
        out += std::tolower(*it);
    }
    *str = out;

    KALDI_ASSERT(str->length() > 0);
  }

  void RegisterBool(const std::string &name, const bool &value,
                    const std::string &doc) {
    std::string idx = name;
    NormalizeOptionName(&idx);
    auto ret = options_.bool_map.emplace(std::make_pair(std::move(idx),
                                                        std::move(value)));
    Register(name, &(ret.first->second), doc);
  }

  void RegisterInt(const std::string &name, const int32 &value,
                   const std::string &doc) {
    std::string idx = name;
    NormalizeOptionName(&idx);
    auto ret = options_.int_map.emplace(std::make_pair(std::move(idx),
                                                        std::move(value)));
    Register(name, &(ret.first->second), doc);
  }

  void RegisterUInt(const std::string &name, const uint32 &value,
                    const std::string &doc) {
    std::string idx = name;
    NormalizeOptionName(&idx);
    auto ret = options_.uint_map.emplace(std::make_pair(std::move(idx),
                                                        std::move(value)));
    Register(name, &(ret.first->second), doc);
  }

  void RegisterFloat(const std::string &name, const float &value,
                     const std::string &doc) {
    std::string idx = name;
    NormalizeOptionName(&idx);
    auto ret = options_.float_map.emplace(std::make_pair(std::move(idx),
                                                        std::move(value)));
    Register(name, &(ret.first->second), doc);
  }

  void RegisterDouble(const std::string &name, const double &value,
                      const std::string &doc) {
    std::string idx = name;
    NormalizeOptionName(&idx);
    auto ret = options_.double_map.emplace(std::make_pair(std::move(idx),
                                                        std::move(value)));
    Register(name, &(ret.first->second), doc);
  }

  void RegisterString(const std::string &name, const std::string &value,
                      const std::string &doc) {
    std::string idx = name;
    NormalizeOptionName(&idx);
    auto ret = options_.str_map.emplace(std::make_pair(std::move(idx),
                                                        std::move(value)));
    Register(name, &(ret.first->second), doc);
  }

private:
  Options options_;
  std::string usage_;
};

}  // namespace kaldi

#endif  // PYKALDI_UTIL_OPTIONS_EXT_H_
