#!/bin/bash

# Script to clean up text files
# Lee Killough
# lee.killough@amd.com

set -ex

export PATH=/usr/bin:/bin

# Go through the entire repository, excluding files normally excluded by Git
git ls-files -z --exclude-standard | while read -rd '' file; do

    # Operate only on regular files of MIME type text/*
    if [[ -f "$file" && "$(file -b --mime-type "$file")" == text/* ]]; then

	# Remove editor backup files ending in ~
	if [[ "$file" = *~ ]]; then
	    git rm "$file"
	    continue
        fi
	
	# Remove trailing whitespace at end of lines (also converts CR-LF to LF)
	sed -i -e 's/[[:space:]]*$//' "$file"

        # Add missing newline to end of file
	sed -i -e '$a\' "$file"

	# Convert UTF8 non-ASCII to ASCII
	temp=$(mktemp)
	iconv -s -f utf-8 -t ascii//TRANSLIT "$file" > "$temp"
	chmod --reference="$file" "$temp"
	mv -f "$temp" "$file"

	# Add the file to the index if it has changed
	git add -u "$file"
    fi
done

cat<<EOF

All of the text files in the repository have been cleaned up.

Review the changes, and commit them if they are acceptable.
EOF
