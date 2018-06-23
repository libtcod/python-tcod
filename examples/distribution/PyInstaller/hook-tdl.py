"""
    Hook for https://github.com/HexDecimal/python-tdl

    You should skip this hook if you're using a custom font.
"""
from PyInstaller.utils.hooks import collect_data_files

# Package tdl's 'default' font file.
datas = collect_data_files('tdl')
