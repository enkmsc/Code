#!/bin/bash

numbers=$(ls ./data.* 2>/dev/null | sed -n 's/.*data\.//p')
number_list=($numbers)

input_file="../sampledcfgs"
output_prefix="oridata"

numbers_regex=$(IFS=\|; echo "${number_list[*]}")

# Use awk to process the input file
awk -v pattern="$numbers_regex" -v output_prefix="$output_prefix" '
    BEGIN {
        # Convert pattern into a regex array
        split(pattern, nums, "|")
        for (i in nums) {
            num_set[nums[i]] = 1
        }
    }
    /BEGIN/ {
        in_block = 1
        block = ""
        feature_number = ""
    }
    in_block {
        block = block $0 "\n"
    }
    /Feature\s+idx/ {
        split($0, arr, " ")
        feature_number = arr[length(arr)]
	if (feature_number in num_set) {
            match_block = 1
        } else {
            match_block = 0
        }
    }
    /END/ {
        if (match_block && feature_number != "") {
            file_name = output_prefix "." feature_number
            print block > file_name
            print "Created file: " file_name > "/dev/stderr"  # Print status message to stderr
        }
        in_block = 0
        match_block = 0
        block = ""
    }
' "$input_file"

for f in data.*; do
        ori_data=./ori$f
        filter_data=./$f

        in_atom=$(sed -n 3p $ori_data)
        out_atom=$(sed -n 3p $filter_data)

        short_end=$((5+$out_atom-1))
        short_start=$((5+$out_atom))
        short_endend=$(wc -l < $filter_data)
        long_end=$((5+$in_atom-1))

        clean=$(sed -n 5,${short_end}p $ori_data | awk '{print $1,$2,$3,$4,$5,$6}')
        add=$(sed -n 5,${short_end}p $filter_data | awk '{print $NF}')

        rm -f allin$f
        echo "$(sed -n 1,4p $ori_data)" >> allin$f
        paste <(echo "$clean") <(echo "$add") -d '      '  >> allin$f
        echo "$(sed -n ${short_start},${long_end}p $ori_data)" >> allin$f
        echo "$(sed -n ${short_start},${short_endend}p $filter_data)" >> allin$f
        echo "" >> allin$f
done

cat allin* > ../sampledcfgs_new
