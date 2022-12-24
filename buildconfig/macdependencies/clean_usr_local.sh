# Cleans /usr/local for the install of mac deps, deleting things that are not
# required, or things that will be replaced with something else

# First clean up some homebrew stuff we don't want linked in
# ----------------------------------------------------------

rm -rf /usr/local/bin/curl
rm -rf /usr/local/opt/curl
rm -rf /usr/local/bin/git
rm -rf /usr/local/opt/git
# Use the apple provided curl, and git.
#     The homebrew ones depend on libs we don't want to include.
# ln -s /usr/bin/curl /usr/local/bin/curl
ln -s /usr/bin/git /usr/local/bin/git

rm -rf /usr/local/lib/libtiff*
rm -rf /usr/local/lib/libzstd*
rm -rf /usr/local/lib/libwebp*
rm -rf /usr/local/lib/libsndfile*
rm -rf /usr/local/lib/glib*
rm -rf /usr/local/lib/libglib*
rm -rf /usr/local/lib/libgthread*
rm -rf /usr/local/lib/libintl*
rm -rf /usr/local/lib/libbrotlidec*
rm -rf /usr/local/lib/libopus*
rm -rf /usr/local/opt/freetype

rm -rf /usr/local/Cellar/libtiff
rm -rf /usr/local/Cellar/libsndfile
rm -rf /usr/local/Cellar/glib
rm -rf /usr/local/Cellar/brotli
rm -rf /usr/local/Cellar/pcre
rm -rf /usr/local/Cellar/opusfile
rm -rf /usr/local/Cellar/opus
rm -rf /usr/local/Cellar/freetype

rm -rf /usr/local/share/doc/tiff-*
rm -rf /usr/local/share/doc/libsndfile
rm -rf /usr/local/share/doc/opusfile
rm -rf /usr/local/share/glib-2.0
rm -rf /usr/local/share/gdb/auto-load

# The installer fails when it tries to create this directory and it already
# exists, so clean it before that
rm -rf /usr/local/share/bash-completion

rm -rf /usr/local/include/glib-2.0
rm -rf /usr/local/include/gio-unix-2.0
rm -rf /usr/local/include/brotli
