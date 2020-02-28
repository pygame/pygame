# build the wheels.
cd buildconfig/manylinux-build
make pull
make wheels
mkdir -p ../../dist/
mv wheelhouse/*.whl ../../dist/
cd ../..
