A config script which uses the cross platform "conan" package manager for managing dependencies.

## links

- [conan](https://conan.io/) the C/C++ package manager.
- [virtualrunenv generator docs](https://docs.conan.io/en/latest/mastering/virtualenv.html#virtualrunenv-generator) for adding library paths (libsdl.so etc) to your environment.
- [json generator docs](https://docs.conan.io/en/latest/reference/generators/json.html)
- [conanfile.txt docs](https://docs.conan.io/en/latest/reference/conanfile_txt.html) the dependency file is called `conanfile.txt`
## Files

- `buildconfig/conanconf`
- `buildconfig/conanconf/README.md` this file
- `buildconfig/conanconf/conanfile.txt` dependencies
- `buildconfig/config_conan.py` integrating conan with pygame build system.
- `build/conan/` (temporary generated files)


## How to compile using conan.

```bash
# install conan
python3 -m pip install conan

# add the bincrafters conan repository.
conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan

# install dependencies with conan, and write a `Setup` file for pygame to build with.
python3 buildconfig/config.py -conan

# add conan library paths to your environment using virtualrunenv
source build/conan/activate_run.sh # Windows: activate_run.bat without the source

# now install pygame either with setup install, or make a wheel.
python3 setup.py install

# OR create the wheel
# python3 setup.py bdist_wheel

# OSX only: bundle the libraries with the wheels using `delocate`
# python3 -m pip install delocate --user
# delocate-wheel -v dist/*.whl

# or install install the pygame wheel file
# python3 -m pip install dist/*.whl --user

# test the build
python3 -m pygame.example.aliens
python3 -m pygame.tests
```
