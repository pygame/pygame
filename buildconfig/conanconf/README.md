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

# add the pygame conan repository.
conan remote add pygame-repo https://api.bintray.com/conan/pygame/pygame

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


## pygame conan repo on 'bintray'.

Here's the pygame organization on "[bintray.com/pygame](https://bintray.com/pygame)". It's where some conan packages can be uploaded.

Custom packages are useful to work around major bugs in released dependencies that may not have new releases with the fixes in them for several months.

Another use case for a conan repo is uploading binaries.

In order to make a custom package, I started by [setting up bintray](https://docs.conan.io/en/latest/uploading_packages/using_bintray.html)

Then this one for [uploading packages to remotes](https://docs.conan.io/en/latest/uploading_packages/uploading_to_remotes.html)



## upload packages to bintray

```
conan upload portmidi/217 --all -r=pygame-repo
```

