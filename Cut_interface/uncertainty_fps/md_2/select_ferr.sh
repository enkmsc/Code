#!/usr/bin/env bash
set -euo pipefail

# Run inside uncertainty_fps/md_2 (or any md_N directory).
# Optional argument: directory holding the LCO_LLZO_*.data files.
THRESHOLD=0.1
INPUT=uncertainty.dat
OUTPUT_DIR=selected_data

if [[ ! -f "$INPUT" ]]; then
    echo "Error: cannot find $INPUT. Run this script in the md_N directory." >&2
    exit 1
fi

md_name=$(basename "$PWD")
if [[ $# -gt 0 ]]; then
    SEARCH_DIRS=("$1")
else
    SEARCH_DIRS=(
        "."
        "./Dump"
        "./inputs"
        "../../md_sample/$md_name/Dump"
        "../../md_sample/$md_name"
        "../../../md_sample/$md_name/Dump"
        "../../../md_sample/$md_name"
    )
fi

new_dat=$(mktemp "./.uncertainty.dat.XXXXXX")
selected_list=$(mktemp)
manifest=$(mktemp)
trap 'rm -f "$new_dat" "$selected_list" "$manifest"' EXIT

awk -v threshold="$THRESHOLD" -v list="$selected_list" '
    /^[[:space:]]*#/ || NF == 0 { print; next }
    NF >= 6 && $1 ~ /^[0-9]+$/ {
        chosen = (($4 + 0) > (threshold + 0)) ? 1 : 0
        printf "%6d %8d %-26s %12.6f %12.6f %d\n", $1, $2, $3, $4, $5, chosen
        if (chosen) print $3 >> list
        next
    }
    { print }
' "$INPUT" > "$new_dat"

# Resolve every selected file before changing uncertainty.dat.
missing=0
while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    found=""
    for dir in "${SEARCH_DIRS[@]}"; do
        if [[ -f "$dir/$name" ]]; then
            found="$dir/$name"
            break
        fi
    done
    if [[ -n "$found" ]]; then
        printf '%s\t%s\n' "$name" "$found" >> "$manifest"
    elif [[ -f "$OUTPUT_DIR/$name" ]]; then
        # Previously copied; already available in selected_data.
        printf '%s\t%s\n' "$name" "$OUTPUT_DIR/$name" >> "$manifest"
    else
        echo "Missing: $name" >&2
        missing=$((missing + 1))
    fi
done < "$selected_list"

if (( missing > 0 )); then
    echo "Error: $missing source .data files not found; uncertainty.dat was not changed." >&2
    echo "Try: bash select_ferr.sh /absolute/path/to/your/Dump" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
while IFS=$'\t' read -r name src; do
    [[ -n "$name" ]] || continue
    if [[ ! -e "$OUTPUT_DIR/$name" || ! "$src" -ef "$OUTPUT_DIR/$name" ]]; then
        cp -p -- "$src" "$OUTPUT_DIR/$name"
    fi
done < "$manifest"

backup="${INPUT}.bak.$(date +%Y%m%d_%H%M%S)"
cp -p -- "$INPUT" "$backup"
mv -f -- "$new_dat" "$INPUT"

printf 'Threshold: ferr_max > %s\n' "$THRESHOLD"
printf 'Selected: %s files\n' "$(wc -l < "$selected_list")"
printf 'Updated:  %s\n' "$INPUT"
printf 'Saved to: %s/\n' "$OUTPUT_DIR"
printf 'Backup:   %s\n' "$backup"

