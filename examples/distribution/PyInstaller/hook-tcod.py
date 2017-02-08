from PyInstaller.utils.hooks import collect_data_files
hiddenimports = ['cffi']
datas = collect_data_files('tcod')
# You may get: 'WARNING: lib not found: SDL.dll', you can ignore this.
