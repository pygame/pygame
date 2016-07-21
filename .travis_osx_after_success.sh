CXXFLAGS="-arch i386 -arch x86_64" CFLAGS="-arch i386 -arch x86_64" LDFLAGS="-arch i386 -arch x86_64" $PYTHON_EXE setup.py bdist_wheel

# This copies in the .dylib files that are linked to the .so files.
# It also rewrites the .so file linking options. https://pypi.python.org/pypi/delocate
delocate-wheel -v dist/*.whl


$PYTHON_EXE .travis_osx_upload_whl.py --no-config
