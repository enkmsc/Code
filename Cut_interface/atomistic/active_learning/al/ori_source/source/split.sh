#!/bin/bash

input_file="xxxINPUTxxx"
output_prefix="data"
output_file=""
feature_number=""
buffer=""

while IFS= read -r line; do
    # Check if the line starts with "BEGIN"
    if [[ "$line" == BEGIN* ]]; then
        # Reset the buffer when we encounter "BEGIN"
        buffer="$line"$'\n'
        output_file=""
    elif [[ "$line" == *"Feature   idx"* ]]; then
        # Extract the number after "Feature ind"
        feature_number=$(echo "$line" | awk '{print $NF}')

        # Set the output file name based on the feature number
        output_file="${output_prefix}.${feature_number}"

        # Write the buffered content to the output file
        echo -n "$buffer" > "$output_file"

        # Add the current "Feature ind" line to the file
        echo "$line" >> "$output_file"

        # Reset the buffer
        buffer=""
    else
        # Add non-"Feature ind" lines to the buffer
        buffer+="$line"$'\n'
    fi

    # After "Feature ind", continue writing lines directly to the file
    if [ -n "$output_file" ] && [[ "$line" != *"Feature   idx"* ]]; then
        echo "$line" >> "$output_file"
    fi
done < "$input_file"
