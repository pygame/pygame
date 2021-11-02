# Mac dependencies using manylinux build scripts

This uses manylinux build scripts to build dependencies on MacOS.

Designed to be run on a Virtual Machine that can be destroyed.
It deletes some homebrew files, and messes with /usr/local/.

Warning: *do not run on your own machine*.

It tries to work as far back as Mac OSX 10.9, for x64 and arm64 (cross compiled) 
architectures.

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

### MacOS doesn't come with gnu compatible readlink

It currently relies on GNU `readlink` to build, which is provided
by the coreutils homebrew package. However, this could be fixed to be
cross platform, since mac `readlink` does not support `-f`.

