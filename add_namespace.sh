cc=$1
shift
for name in "$@"; do
  base=${name#*::}
  sed -i "s/${base}/${name}/g" $cc
done
