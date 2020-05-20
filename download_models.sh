#!/usr/bin/env bash

mkdir -p ./data/models/snapshots
cd data/models/snapshots
wget -O tmp.zip https://www.dropbox.com/sh/n9zija0hk68g499/AACJ0pciymwdi4WtY2sX2i9Wa
unzip tmp.zip -d lms_wflw
rm tmp.zip

