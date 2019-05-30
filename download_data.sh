#!/bin/sh

if [ -d "./data/umbrella" ]
then
    echo "Data already exists" && exit 0
fi

# you can download more data from https://www.dropbox.com/sh/qgy2n9bmioofqnj/AABUnT7pi2ECpxSi80EmXOXna?dl=0
wget -O umbrella_data.zip "https://uc352ee2248ce04ee564574b1079.dl.dropboxusercontent.com/cd/0/get/AhZS-4EEDb7n8IdQXjsZBwEiBcTR913pcjf5UXDRxIklKyWX3ZTDNl8uiTPi3k3-sD7mIkx2l7BKouR0AzJRX6VLZTBfm4xbStrGcUhfiwRzgg/file?_download_id=6294509971162504286597723557996592959721079245428546948759767549&_notify_domain=www.dropbox.com&dl=1"
mkdir -p data/umbrella/depth
mkdir -p data/umbrella/color

mv umbrella_data.zip data/umbrella
cd data/umbrella
unzip umbrella_data.zip
rm *.txt
mv *color*.png color/
mv *depth*.png depth/
rm umbrella_data.zip

