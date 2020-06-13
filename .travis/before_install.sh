#!/bin/bash
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    SDL_MAJOR=2
    SDL_MINOR=5
    UPDATE_SDL=true

    if [[ $(sdl2-config --version) =~ ([0-9]*).([0-9]*).([0-9]*) ]]; then
        if (( ${BASH_REMATCH[1]} >= $SDL_MAJOR && ${BASH_REMATCH[3]} >= $SDL_MINOR )); then
            UPDATE_SDL=false
        fi
    fi

    if [[ $UPDATE_SDL == "true" ]]; then
        # Update SDL2 to a recent version.
        wget -O - https://www.libsdl.org/release/SDL2-2.0.8.tar.gz | tar xz
        (cd SDL2-* && ./configure --prefix=$HOME/.local && make -j 3 install)
        PATH=~/.local/bin:$PATH
    fi
fi
