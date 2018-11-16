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
#ifndef CLIF_PYTHON_PROTO_UTIL_H_
#define CLIF_PYTHON_PROTO_UTIL_H_

#include <string>
#include <vector>

#include "google/protobuf/descriptor.h"

namespace proto2 = google::protobuf;
using std::string;

namespace clif_proto {

// This same info type is used for both enums as well as messages.
struct ProtoTypeInfo {
  // The name of the message after the last '.' in its fully qualified name.
  string name;

  // The fully qualified name of the message delimited by '.'.
  string fqname;

  // The package of the proto file in which this message is declared.
  string package;

  // The proto file in which this proto message is defined.
  string srcfile;
};

struct MethodInfo {
  // The method name.
  string name;

  // The descriptor of the request message type.
  ProtoTypeInfo request;

  // The descriptor of the reply message type.
  ProtoTypeInfo reply;

  };

struct ServiceInfo {
  // The name of the service.
  string name;

  // The fulle qualified name of the service delimited by '.'.
  string fqname;

  // The proto file in which this service is defined.
  string srcfile;

  // The list of methods in this service.
  std::vector<MethodInfo> methods;

  };

class ProtoFileInfo {
 public:
  // Parses the proto file with path |proto_file_path| and indexes it.
  // If the proto file is succesfully parsed, then the "IsValid" method
  // returns true, else false. The errors encountered while parsing are
  // returned by the "ErrorMsg" method.
    ProtoFileInfo(const string& proto_file_path,
                const string& additional_import_path);

  bool IsValid() const {
    return valid_;
  }

  // Returns a string of errors encountered when parsing the proto file.
  // The returned string is a true error message only when "IsValid" returns
  // false.
  const string& ErrorMsg() const {
    return error_msg_;
  }

  // The getters below return meaningful values only if the "IsValid" method
  // returns true.
  const string &SrcFile() const {
    return proto_file_path_;
  }

  const string &PackageName() const {
    return package_;
  }

  const std::vector<ProtoTypeInfo> &Messages() const {
    return messages_;
  }

  const std::vector<ProtoTypeInfo> &Enums() const {
    return enums_;
  }

  const std::vector<ServiceInfo> &Services() const {
    return services_;
  }

 private:
  void Index();
  void IndexMessage(const proto2::Descriptor& d);

  bool valid_;
  string proto_file_path_;
  string additional_import_path_;

  string package_;
  string error_msg_;
  std::vector<ProtoTypeInfo> messages_;
  std::vector<ProtoTypeInfo> enums_;
  std::vector<ServiceInfo> services_;

  ProtoFileInfo(const ProtoFileInfo&) = delete;
  ProtoFileInfo& operator=(const ProtoFileInfo&) = delete;
};

}  // namespace clif_proto

#endif  // CLIF_PYTHON_PROTO_UTIL_H_
