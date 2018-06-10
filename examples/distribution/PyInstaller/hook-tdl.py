
from PyInstaller.utils.hooks import collect_data_files

# Package the 'default' font.
# You can skip this hook if you're using a custom font.
datas = collect_data_files('tdl')
