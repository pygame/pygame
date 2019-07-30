if [[ ${BUILD_UNIVERSAL} == "1" ]]; then
  export CXXFLAGS="-arch i386 -arch x86_64" CFLAGS="-arch i386 -arch x86_64" LDFLAGS="-arch i386 -arch x86_64"
  echo "Set flags for universal build"
  echo $CFLAGS
fi

echo "Building wheel..."
$PYTHON_EXE setup.py bdist_wheel

# This copies in the .dylib files that are linked to the .so files.
# It also rewrites the .so file linking options. https://pypi.python.org/pypi/delocate
delocate-wheel -v dist/*.whl

$PYTHON_EXE buildconfig/ci/travis/.travis_osx_rename_whl.py


# if a tag is pushed we upload wheels. TWINE_USERNAME TWINE_PASSWORD need to be set.
if [ "${TRAVIS_PULL_REQUEST}" = "false" ] && [ -n "${TRAVIS_TAG}" ]; then
  $PYTHON_EXE -m twine upload dist/*.whl
fi
