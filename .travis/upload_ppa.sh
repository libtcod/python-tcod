#!/bin/bash
function deb_install {
    if [[ "$TRAVIS_OS_NAME" == "linux" && "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
        pip install stdeb
        pip3 install --upgrade pip --user
        pip3 install --requirement requirements.txt --user
        ./setup.py --command-packages=stdeb.command sdist_dsc --with-python2=True --with-python3=True
        export DEB_READY='yes'
    fi
}
function deb_upload {
    if [[ -n "$TRAVIS_TAG" && -n "$DEB_READY" ]]; then
        openssl aes-256-cbc -K $encrypted_765c87af1f2f_key -iv $encrypted_765c87af1f2f_iv -in .travis/launchpad.key.enc | gpg --import
        debsign -k5B69F065 deb_dist/*.changes
        dput ppa:4b796c65/ppa deb_dist/*.changes
    fi
}
