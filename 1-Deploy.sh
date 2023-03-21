#!/bin/sh

git submodule update --init --recursive

chmod +x "External/Packman/packman"
chmod +x "External/NRIFramework/External/Packman/packman"
chmod +x "External/NRIFramework/External/NRI/External/Packman/packman"
chmod +x "2-Build.sh"
chmod +x "3-Prepare NRD SDK.sh"
chmod +x "4-Clean.sh"

mkdir -p "_Compiler"

cd "_Compiler"
cmake ..
cd ..
