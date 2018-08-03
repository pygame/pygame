
# ls -la /usr/local/lib/
# ls -la /usr/local/include/
# cat Setup
# otool -L `find . -name imageext.so`
# otool -L `find . -name image.so`
# otool -L /usr/local/opt/sdl_image/lib/libSDL_image-1.2.0.dylib

$PYTHON_EXE setup.py -config -auto

# Removes any X11R6 stuff so that png headers are not picked up by accident.
sed 's/-I\/usr\/X11R6\/include //g' Setup > Setup.new
mv Setup.new Setup


# Makes for i386 and x86_64. With older 'core2' cpus.
HOMEBREW_BUILD_BOTTLE=1 HOMEBREW_BOTTLE_ARCH=core2 CXXFLAGS="-arch i386 -arch x86_64 -march=core2" CFLAGS="-arch i386 -arch x86_64 -march=core2" LDFLAGS="-arch i386 -arch x86_64" $PYTHON_EXE setup.py install -noheaders
