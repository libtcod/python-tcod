#!/bin/bash
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    export DISPLAY=:99.0
    sh -e /etc/init.d/xvfb start

    # Update SDL2 to a recent version.
    wget -O - https://www.libsdl.org/release/SDL2-2.0.8.tar.gz | tar xz
    (cd SDL2-* && ./configure --prefix=$HOME/.local && make -j 3 install)
    PATH=~/.local/bin:$PATH
fi
