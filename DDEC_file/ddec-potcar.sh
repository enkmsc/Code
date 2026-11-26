#!/bin/bash

# Split each atomic POTCAR section into temporary files
csplit -s -f potcar_chunk_ POTCAR '/^   TITEL/' '{*}'

# Process each chunk
for file in potcar_chunk_*; do
  # Extract element symbol from VRHFIN line
  element=$(grep "VRHFIN" $file | sed -E 's/.*VRHFIN *= *([^:]+):.*/\1/')

  # Remove problematic lines
  sed -i '/SHA256/d;/COPYR/d;/LEXCH/d' "$file"

  # Replace TITEL line with simplified version
  sed -i "s/^.*TITEL.*/   TITEL  = PAW ${element}/" "$file"
done

# Concatenate all processed chunks into new POTCAR
cat potcar_chunk_* > POTCAR
rm potcar_chunk_*

echo "All elements have been processed."


