from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_data_files
hiddenimports = ['cffi']
datas = collect_data_files('tcod')
