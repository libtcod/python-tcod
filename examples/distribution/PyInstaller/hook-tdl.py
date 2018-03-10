from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
hiddenimports = ['cffi']
datas = collect_data_files('tdl') # packages the 'default' font
# install all binaries to the working directory
binaries = [(lib, '') for lib, _ in collect_dynamic_libs('tcod')]
