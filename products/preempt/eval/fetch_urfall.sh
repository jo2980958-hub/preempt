#!/usr/bin/env bash
# Fetch a subset of the UR Fall Detection Dataset into eval/data.
#
#   Kwolek B., Kepski M., "Human fall detection on embedded platform using
#   depth maps and wireless accelerometer", Computer Methods and Programs in
#   Biomedicine, 2014. http://fenix.ur.edu.pl/~mkepski/ds/uf.html
#
# Licence, quoted from the dataset page: "This work is licensed under a Creative
# Commons Attribution-NonCommercial-ShareAlike 4.0 International License and is
# intended for non-commercial academic use."
#
# That is why nothing it downloads is committed to this repository, and why no
# frame or image derived from it appears in any document here. It is fetched at
# evaluation time and it stays in eval/data, which is gitignored.
#
#   eval/fetch_urfall.sh [count]      default 12 falls and 12 activity sequences
set -euo pipefail
cd "$(dirname "$0")"
COUNT="${1:-12}"
BASE="https://fenix.ur.edu.pl/~mkepski/ds/data"
mkdir -p data
cd data

curl -fsSL -o urfall-cam0-falls.csv "$BASE/urfall-cam0-falls.csv"
curl -fsSL -o urfall-cam0-adls.csv  "$BASE/urfall-cam0-adls.csv"

for kind in fall adl; do
  for i in $(seq -w 1 "$COUNT"); do
    name="$kind-$i"
    [[ -d "$name" ]] && continue
    echo "fetching $name"
    curl -fsSL -o "$name.zip" "$BASE/$name-cam0-rgb.zip"
    mkdir -p "$name"
    unzip -q -o -j "$name.zip" -d "$name"
    rm -f "$name.zip"
  done
done

du -sh .
