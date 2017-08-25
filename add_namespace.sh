cc=$1
shift
for name in "$@"; do
  base=${name#*::}
  sed -i "/${name}/! s/${base}/${name}/g" $cc
done
