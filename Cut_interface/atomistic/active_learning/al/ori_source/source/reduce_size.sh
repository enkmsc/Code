#!/bin/bash

# Check if sufficient arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_file> <output_file> <max_size_in_MB>"
    exit 1
fi

# Parse command-line arguments
input_file="$1"
output_file="$2"
max_size_mb="$3"

# Convert MB to bytes
max_size=$((max_size_mb * 1024 * 1024))

# Step 1: Truncate the file to the size limit (less than or equal to max_size)
truncate_file="${output_file}.tmp"
dd if="$input_file" of="$truncate_file" bs=1M count="$max_size_mb" status=none

# Step 2: Scan forward, keeping blocks and printing all complete BEGIN ... END blocks
awk '
    /^BEGIN/ { 
        block = "";                          # Start a new block
        block_started = 1                     # Indicate that we are in a block
    }
    { 
        block = block $0 ORS                  # Accumulate lines in the current block
    }
    /^END/ { 
        if (block_started == 1) {
            print block                         # Print the completed block
            block_started = 0                  # Reset block status
        }
    }
' "$truncate_file" > "$output_file"

# Clean up the temporary file
rm "$truncate_file"
