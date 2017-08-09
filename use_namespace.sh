cc=$1
shift
for namespace in "$@"; do
  sed -i "s/using namespace clif;$/using namespace clif;\nusing namespace ${namespace};/g" $cc
done
