input_dir=$1
output_dir=$2
for f in "$input_dir"/*jp2; do
  filename=$output_dir"/"$(basename $f)
  convert $f "${filename/.jp2/.png}"
done
