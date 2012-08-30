cd ..
python setup.py install
cd documentation
del *.html
python -m pydoc -w tdl
python -m pydoc -w tdl.local
python -m pydoc -w tdl.event
python -m pydoc -w tdl.tcod