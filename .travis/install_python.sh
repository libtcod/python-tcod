#!/bin/bash
# *********************
# Copyright and License
# *********************

# The multibuild package, including all examples, code snippets and attached
# documentation is covered by the 2-clause BSD license.

    # Copyright (c) 2013-2016, Matt Terry and Matthew Brett; all rights
    # reserved.

    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are
    # met:

    # 1. Redistributions of source code must retain the above copyright notice,
    # this list of conditions and the following disclaimer.

    # 2. Redistributions in binary form must reproduce the above copyright
    # notice, this list of conditions and the following disclaimer in the
    # documentation and/or other materials provided with the distribution.

    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    # IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    # THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    # PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
set -e

# Work round bug in travis xcode image described at
# https://github.com/direnv/direnv/issues/210
shell_session_update() { :; }

MACPYTHON_URL=https://www.python.org/ftp/python
PYPY_URL=https://bitbucket.org/pypy/pypy/downloads
MACPYTHON_PY_PREFIX=/Library/Frameworks/Python.framework/Versions
WORKING_SDIR=working
DOWNLOADS_SDIR=/tmp

function untar {
    local in_fname=$1
    if [ -z "$in_fname" ];then echo "in_fname not defined"; exit 1; fi
    local extension=${in_fname##*.}
    case $extension in
        tar) tar -xf $in_fname ;;
        gz|tgz) tar -zxf $in_fname ;;
        bz2) tar -jxf $in_fname ;;
        zip) unzip $in_fname ;;
        xz) unxz -c $in_fname | tar -xf ;;
        *) echo Did not recognize extension $extension; exit 1 ;;
    esac
}

function lex_ver {
    # Echoes dot-separated version string padded with zeros
    # Thus:
    # 3.2.1 -> 003002001
    # 3     -> 003000000
    echo $1 | awk -F "." '{printf "%03d%03d%03d", $1, $2, $3}'
}

function unlex_ver {
    # Reverses lex_ver to produce major.minor.micro
    # Thus:
    # 003002001 -> 3.2.1
    # 003000000 -> 3.0.0
    echo "$((10#${1:0:3}+0)).$((10#${1:3:3}+0)).$((10#${1:6:3}+0))"
}

function strip_ver_suffix {
    echo $(unlex_ver $(lex_ver $1))
}

function check_var {
    if [ -z "$1" ]; then
        echo "required variable not defined"
        exit 1
    fi
}

function pyinst_ext_for_version {
    # echo "pkg" or "dmg" depending on the passed Python version
    # Parameters
    #   $py_version (python version in major.minor.extra format)
    #
    # Earlier Python installers are .dmg, later are .pkg.
    local py_version=$1
    check_var $py_version
    local py_0=${py_version:0:1}
    if [ $py_0 -eq 2 ]; then
        if [ "$(lex_ver $py_version)" -ge "$(lex_ver 2.7.9)" ]; then
            echo "pkg"
        else
            echo "dmg"
        fi
    elif [ $py_0 -ge 3 ]; then
        if [ "$(lex_ver $py_version)" -ge "$(lex_ver 3.4.2)" ]; then
            echo "pkg"
        else
            echo "dmg"
        fi
    fi
}

function get_pypy_build_prefix {
    # gets the file prefix of the pypy.org PyPy2
    #
    # Parameters:
    #   $version : pypy2 version number
    local version=$1
    if [[ $version =~ ([0-9]+)\.([0-9]+) ]]; then
        local major=${BASH_REMATCH[1]}
        local minor=${BASH_REMATCH[2]}
        if (( $major > 5 || ($major == 5 && $minor >= 3) )); then
            echo "pypy2-v"
        else
            echo "pypy-"
        fi
    else
        echo "error: expected version number, got $1" 1>&2
        exit 1
    fi
}

function get_pypy3_build_prefix {
    # gets the file prefix of the pypy.org PyPy3
    #
    # Parameters:
    #   $version : pypy3 version number
    local version=$1
    if [[ $version =~ ([0-9]+)\.([0-9]+) ]]; then
        local major=${BASH_REMATCH[1]}
        local minor=${BASH_REMATCH[2]}
        if (( $major == 5 && $minor <= 5 )); then
            echo "pypy3.3-v"
        elif (( $major < 5 )); then
            echo "pypy3-"
        else
            echo "pypy3-v"
        fi
    else
        echo "error: expected version number, got $1" 1>&2
        exit 1
    fi
}

function install_python {
    # Picks an implementation of Python determined by the current enviroment
    # variables then installs it
    # Sub-function will set $PYTHON_EXE variable to the python executable
    if [ -n "$MB_PYTHON_VERSION" ]; then
        install_macpython $MB_PYTHON_VERSION
    elif [ -n "$PYPY_VERSION" ]; then
        install_mac_pypy $PYPY_VERSION
    elif [ -n "$PYPY3_VERSION" ]; then
        install_mac_pypy3 $PYPY3_VERSION
    fi
}

function install_macpython {
    # Installs Python.org Python
    # Parameter $version
    # Version given in major or major.minor or major.minor.micro e.g
    # "3" or "3.4" or "3.4.1".
    # sets $PYTHON_EXE variable to python executable
    local py_version=$1
    local py_stripped=$(strip_ver_suffix $py_version)
    local inst_ext=$(pyinst_ext_for_version $py_version)
    local py_inst=python-$py_version-macosx10.6.$inst_ext
    local inst_path=$DOWNLOADS_SDIR/$py_inst
    mkdir -p $DOWNLOADS_SDIR
    wget -nv $MACPYTHON_URL/$py_stripped/${py_inst} -P $DOWNLOADS_SDIR
    if [ "$inst_ext" == "dmg" ]; then
        hdiutil attach $inst_path -mountpoint /Volumes/Python
        inst_path=/Volumes/Python/Python.mpkg
    fi
    sudo installer -pkg $inst_path -target /
    local py_mm=${py_version:0:3}
    PYTHON_EXE=$MACPYTHON_PY_PREFIX/$py_mm/bin/python$py_mm
    # Install certificates for Python 3.6
    local inst_cmd="/Applications/Python ${py_mm}/Install Certificates.command"
    if [ -e "$inst_cmd" ]; then
        sh "$inst_cmd"
    fi
}

function install_mac_pypy {
    # Installs pypy.org PyPy
    # Parameter $version
    # Version given in full major.minor.micro e.g "3.4.1".
    # sets $PYTHON_EXE variable to python executable
    local py_version=$1
    local py_build=$(get_pypy_build_prefix $py_version)$py_version-osx64
    local py_zip=$py_build.tar.bz2
    local zip_path=$DOWNLOADS_SDIR/$py_zip
    mkdir -p $DOWNLOADS_SDIR
    wget -nv $PYPY_URL/${py_zip} -P $DOWNLOADS_SDIR
    untar $zip_path
    PYTHON_EXE=$(realpath $py_build/bin/pypy)
}

function install_mac_pypy3 {
    # Installs pypy.org PyPy3
    # Parameter $version
    # Version given in full major.minor.micro e.g "3.4.1".
    # sets $PYTHON_EXE variable to python executable
    local py_version=$1
    local py_build=$(get_pypy3_build_prefix $py_version)$py_version-osx64
    local py_zip=$py_build.tar.bz2
    local zip_path=$DOWNLOADS_SDIR/$py_zip
    mkdir -p $DOWNLOADS_SDIR
    wget -nv $PYPY_URL/${py_zip} -P $DOWNLOADS_SDIR
    mkdir -p $py_build
    tar -xjf $zip_path -C $py_build --strip-components=1
    PYTHON_EXE=$(realpath $py_build/bin/pypy3)
}

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    install_python
    if [[ -n "$PYTHON_EXE" ]]; then
        virtualenv venv -p $PYTHON_EXE
    else
        virtualenv venv
    fi
    source venv/bin/activate
fi
