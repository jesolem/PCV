#!/bin/sh

URL="http://programmingcomputervision.com/downloads/pcv_data.zip"

if /usr/bin/env wget $URL
    then

    mkdir data-unpack.tmp
    cd data-unpack.tmp

    unzip -q ../pcv_data.zip
    rm -rf __MACOSX

    cd ..

    if [ -d data/ ]
        then
        echo Directory 'pcv-data' already exists.
        echo Look in the data-unpack.tmp directory for the unzipped data.
    else
        mv data-unpack.tmp/data .
        rmdir data-unpack.tmp
    fi

    rm pcv_data.zip
else
    echo Could not get data archive. Check URL and make sure you have wget.
fi
