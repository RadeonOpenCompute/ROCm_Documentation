#!/usr/bin/env bash
for file in $(find . -type f -iname '*.rst'); do
    contents="$(cat $file)"
    cat <<EOF > $file
# ROCm Documentation has moved to docs.amd.com

.. meta::
   :http-equiv=Refresh: 0; url='https://docs.amd.com'
EOF
    echo "$contents" >>$file
done
