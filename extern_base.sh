h=$1
shift
for name in "$@"; do
namespace=${name%::*}
base=${name#*::}
cat >> $h <<EOF

namespace $namespace {

namespace $base {
  extern PyTypeObject wrapper_Type;
}

}  // namespace $namespace
EOF
done
