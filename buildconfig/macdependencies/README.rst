# Mac dependencies using manylinux build scripts

This uses manylinux build scripts to build dependencies on MacOS.

Designed to be run on a Virtual Machine that can be destroyed.
It deletes some homebrew files, and messes with /usr/local/.

Warning: *do not run on your own machine*.


It tries to work as far back as Mac OSX 10.9, for the x64 architecture.

If there needs to be separate configure options between linux and mac
then something like the following can be used.

```bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
      # linux
      export SDL_IMAGE_CONFIGURE=
elif [[ "$OSTYPE" == "darwin"* ]]; then
      # Mac OSX
      # --disable-imageio is so it doesn't use the built in mac image loading.
      #     Since it is not as compatible with some jpg/png files.
      export SDL_IMAGE_CONFIGURE=--disable-imageio
fi
```

## Future work

### Other architectures

Future work would be to support other architectures, which should be
fairly possible since a lot of the dependencies are built from source.
This includes the older i386 and ppc, the newer arm64 as well as
universal (i386 and ppc) and universal2 (x86_64 and arm64).

### MacOS doesn't come with gnu compatible readlink

It currently relies on GNU `readlink` to build, which is provided
by the coreutils homebrew package. However, this could be fixed to be
cross platform, since mac `readlink` does not support `-f`.

### Build is slow

It does take 22+ minutes or so of build time... oooof.

Caching could help? https://github.com/actions/cache

Below is untested, but is the general idea.

```yml

    - name: Cache dependencies
      id: cache-deps
      if: matrix.os == 'macos-10.15'
      uses: actions/cache@v2
      with:
        path: .pygame-deps
        key: ${{ runner.os }}-pygame-deps

    - name: Build deps
      if: steps.cache-deps.outputs.cache-hit != 'true' && matrix.os == 'macos-10.15'
      run: |
        mkdir -p ~/.pygame-deps/
        cp -a buildconfig/manylinux-build/docker_base ~/.pygame-deps/
        cp -a buildconfig/macdependencies ~/.pygame-deps/
        brew install coreutils
        cd ~/.pygame-deps/macdependencies
        ./install_mac_deps.sh
        cd ../..
```

We could also enable homebrew deps for testing until we are close to a release?


### Future packages to add?

- libopus
- brotli
- bzip2
- libjpegturbo
