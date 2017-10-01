// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fcntl.h>
#include <sys/stat.h>

#include <memory>
#include <sstream>

#include "google/protobuf/compiler/importer.h"
#include "proto_util.h"
using proto2::Descriptor;
using proto2::EnumDescriptor;
using proto2::FileDescriptor;
using proto2::MethodDescriptor;
using proto2::ServiceDescriptor;
using proto2::compiler::DiskSourceTree;
using proto2::compiler::Importer;
using proto2::compiler::MultiFileErrorCollector;

namespace {

class ProtoFileErrorCollector : public MultiFileErrorCollector {
 public:
  explicit ProtoFileErrorCollector(string* error_str) : error_str_(error_str) {}

  void AddError(
      const string& file, int line, int col, const string& msg) override {
    std::ostringstream stream;
    stream << "Error parsing " << file;
    if (line > 0) {
      stream << ":" << line << ":" << col;
    } else {
      stream << ": " << msg;
    }

    *error_str_ += stream.str();
  }

 private:
  string* error_str_;
};

template <typename T>
clif_proto::ProtoTypeInfo MakeProtoTypeInfo(const T& d) {
  const FileDescriptor* fd = d.file();
  clif_proto::ProtoTypeInfo info{
    d.name(), d.full_name(), fd->package(), fd->name()};
  return info;
}

}  // namespace

namespace clif_proto {

ProtoFileInfo::ProtoFileInfo(const string& proto_file_path,
                             const string& additional_import_path)
    : valid_(false),  // The 'Index' method will set it to the correct validity.
      proto_file_path_(proto_file_path),
      additional_import_path_(additional_import_path) {
  Index();
}

void ProtoFileInfo::Index() {
  std::unique_ptr<DiskSourceTree> src_tree(new DiskSourceTree);
  // Map the root and source tree directories.
  src_tree->MapPath("/", "/");
  src_tree->MapPath("", ".");
  src_tree->MapPath("", additional_import_path_);

  std::unique_ptr<ProtoFileErrorCollector> errors(
      new ProtoFileErrorCollector(&error_msg_));
  std::unique_ptr<Importer> importer(
      new Importer(src_tree.get(), errors.get()));

  const FileDescriptor* fd = importer->Import(proto_file_path_);
  if (fd == nullptr) {
    valid_ = false;
    return;
  }

  package_ = fd->package();
  for (int i = 0; i < fd->message_type_count(); ++i) {
    const Descriptor* d = fd->message_type(i);
    IndexMessage(*d);
  }

  for (int i = 0; i < fd->enum_type_count(); ++i) {
    const EnumDescriptor* ed = fd->enum_type(i);
    enums_.push_back(MakeProtoTypeInfo(*ed));
  }

  for (int i = 0; i < fd->service_count(); ++i) {
    const ServiceDescriptor* sd = fd->service(i);
    ServiceInfo csd;
    csd.name = sd->name();
    csd.fqname = sd->full_name();
    csd.srcfile = sd->file()->name();
    
    for (int j = 0; j < sd->method_count(); ++j) {
      const MethodDescriptor* m = sd->method(j);
      MethodInfo md;
      md.name = m->name();
      
      const Descriptor* d = m->input_type();
      md.request = MakeProtoTypeInfo(*d);

      d = m->output_type();
      md.reply = MakeProtoTypeInfo(*d);

      csd.methods.push_back(md);
    }

    services_.push_back(csd);
  }

  valid_ = true;
}

void ProtoFileInfo::IndexMessage(const Descriptor& d) {
  // Map entries show up as nested message types, but we do not want them
  // as such.
  if (d.options().has_map_entry() && d.options().map_entry()) {
    return;
  }

  messages_.push_back(MakeProtoTypeInfo(d));

  for (int i = 0; i < d.nested_type_count(); ++i) {
    const Descriptor* nd = d.nested_type(i);
    IndexMessage(*nd);
  }

  for (int i = 0; i < d.enum_type_count(); ++i) {
    const EnumDescriptor* ed = d.enum_type(i);
    enums_.push_back(MakeProtoTypeInfo(*ed));
  }
}

}  // namespace clif_proto
