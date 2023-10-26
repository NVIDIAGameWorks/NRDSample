#!/bin/bash

NRD_DIR=External/NRD
NRI_DIR=External/NRIFramework/External/NRI

rm -rf "_NRD_SDK"

mkdir -p "_NRD_SDK/Include"
mkdir -p "_NRD_SDK/Lib/Debug"
mkdir -p "_NRD_SDK/Lib/Release"
mkdir -p "_NRD_SDK/Shaders"
mkdir -p "_NRD_SDK/Shaders/Include"

cd "_NRD_SDK"

cp -r ../$NRD_DIR/Include/ "Include"
cp -H ../_Bin/Debug/libNRD.so "Lib/Debug"
cp -H ../_Bin/Release/libNRD.so "Lib/Release"
cp ../$NRD_DIR/Shaders/Include/NRD.hlsli "Shaders/Include"
cp ../$NRD_DIR/Shaders/Include/NRDEncoding.hlsli "Shaders/Include"
cp ../$NRD_DIR/LICENSE.txt "."
cp ../$NRD_DIR/README.md "."

read -p "Do you need the shader source code for a white-box integration? [y/n]" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    mkdir -p "Shaders"

    cp -r ../$NRD_DIR/Shaders/ "Shaders"
    cp ../$NRD_DIR/External/MathLib/*.hlsli "Shaders\Source"
fi

cd ..

read -p "Do you need NRD integration layer? [y/n]" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    mkdir -p "_NRD_SDK/Integration"
    cp -r $NRD_DIR/Integration/ "_NRD_SDK/Integration"

    rm -rf "_NRI_SDK"

    mkdir -p "_NRI_SDK/Include/Extensions"
    mkdir -p "_NRI_SDK/Lib/Debug"
    mkdir -p "_NRI_SDK/Lib/Release"

    cd "_NRI_SDK"

    cp -r ../$NRI_DIR/Include/ "Include"
    cp -r ../$NRI_DIR/Include/Extensions/ "Include/Extensions"
    cp -H ../_Bin/Debug/libNRI.so "Lib/Debug"
    cp -H ../_Bin/Release/libNRI.so "Lib/Release"
    cp ../$NRI_DIR/LICENSE.txt "."

    cd ..
fi
