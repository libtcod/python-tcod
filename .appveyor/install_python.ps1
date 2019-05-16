if ($env:PYPY -or $env:PYPY3) {
    if($env:PYPY3){
        $env:PYPY_EXE='pypy3.exe'
        $env:PYPY=$env:PYPY3
    } else {
        $env:PYPY_EXE='pypy.exe'
    }
    $env:PYPY = $env:PYPY + '-win32'
    $env:PYTHON = 'C:\' + $env:PYPY + '\' + $env:PYPY_EXE
    $env:PATH += ';' + 'C:\' + $env:PYPY + '\'
    $PYPY_DOWNLOAD = 'https://bitbucket.org/pypy/pypy/downloads/' + $env:PYPY + '.zip'
    Invoke-WebRequest $PYPY_DOWNLOAD -OutFile C:\pypy.zip
    & '7z' x C:\pypy.zip -oC:\
    & $env:PYTHON -m ensurepip
}
if ($env:WEB_PYTHON) {
    $PYTHON_INSTALLER = 'C:\python-webinstall.exe'
    Start-FileDownload $env:WEB_PYTHON -FileName $PYTHON_INSTALLER
    Start-Process $PYTHON_INSTALLER -Wait -ArgumentList "/quiet InstallAllUsers=1 TargetDir=C:\UserPython Include_doc=0 Include_launcher=0 Include_test=0 Shortcuts=0"
    $env:PYTHON = 'C:\UserPython\python.exe'
}
& $env:PYTHON -m pip install --disable-pip-version-check --upgrade pip
& $env:PYTHON -m pip install --no-warn-script-location "virtualenv>=16"
& $env:PYTHON -m virtualenv venv

$env:ACTIVATE_VENV='venv\Scripts\activate.bat'

if($LastExitCode -ne 0) { $host.SetShouldExit($LastExitCode )  }
