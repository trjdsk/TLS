#!/usr/bin/env bash
set -e

BUILD_DIR=build
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake ..
cmake --build . --config Release -j$(nproc)

echo "✅ Build finished. Library is in $BUILD_DIR/lib/"
