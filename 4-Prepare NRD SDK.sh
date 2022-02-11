#!/bin/bash

NRD_DIR=External/NRD
NRI_DIR=External/NRIFramework/External/NRI

rm -rf "_NRD_SDK"

mkdir -p "_NRD_SDK/Include"
mkdir -p "_NRD_SDK/Integration"
mkdir -p "_NRD_SDK/Lib/Debug"
mkdir -p "_NRD_SDK/Lib/Release"

cd "_NRD_SDK"

cp -r ../$NRD_DIR/Integration/ "Integration"
cp -r ../$NRD_DIR/Include/ "Include"
cp -H ../_Build/Debug/libNRD.so "Lib/Debug"
cp -H ../_Build/Release/libNRD.so "Lib/Release"
cp ../$NRD_DIR/LICENSE.txt "."
cp ../$NRD_DIR/README.md "."

read -p "Do you need the shader source code for a white-box integration? [y/n]" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    mkdir -p "Shaders"

    cp ../$NRD_DIR/Source/Shaders/*.* "Shaders"
    cp ../$NRD_DIR/Source/Shaders/Include/* "Shaders"
    cp ../$NRD_DIR/External/MathLib/*.hlsli "Shaders"
    cp ../$NRD_DIR/Include/*.hlsli "Shaders"
fi

cd ..

read -p "Do you need NRI required for NRDIntegration? [y/n]" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -rf "_NRI_SDK"

    mkdir -p "_NRI_SDK/Include"
    mkdir -p "_NRI_SDK/Lib/Debug"
    mkdir -p "_NRI_SDK/Lib/Release"

    cd "_NRI_SDK"

    cp -r ../$NRI_DIR/Include/ "Include"
    cp -H ../_Build/Debug/libNRI.so "Lib/Debug"
    cp -H ../_Build/Release/libNRI.so "Lib/Release"
    cp ../$NRI_DIR/LICENSE.txt "."

    cd ..
fi
