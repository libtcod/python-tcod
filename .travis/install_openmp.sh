#!/bin/sh
brew update
brew install llvm
export PATH=/usr/local/opt/llvm/bin:$PATH
export LDFLAGS=-L/usr/local/opt/llvm/lib $LDFLAGS
export CPPFLAGS=-I/usr/local/opt/llvm/include $CPPFLAGS
# Installs clang with OpenMP via brew
