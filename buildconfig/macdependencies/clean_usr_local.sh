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
rm -rf /usr/local/lib/libomp*
rm -rf /usr/local/lib/libmp3lame*
rm -rf /usr/local/lib/libpcre*
rm -rf /usr/local/opt/freetype

rm -rf /usr/local/Cellar/libtiff
rm -rf /usr/local/Cellar/libsndfile
rm -rf /usr/local/Cellar/glib*
rm -rf /usr/local/Cellar/brotli
rm -rf /usr/local/Cellar/pcre
rm -rf /usr/local/Cellar/pcre2
rm -rf /usr/local/Cellar/opusfile
rm -rf /usr/local/Cellar/opus
rm -rf /usr/local/Cellar/freetype
rm -rf /usr/local/Cellar/libomp
rm -rf /usr/local/Cellar/lame

rm -rf /usr/local/share/doc/tiff-*
rm -rf /usr/local/share/doc/libsndfile
rm -rf /usr/local/share/doc/opusfile
rm -rf /usr/local/share/gdb/auto-load

# glib gunk
rm -rf /usr/local/share/glib-2.0
rm -rf /usr/local/bin/pcre2*
rm -rf /usr/local/lib/libgobject*
rm -rf /usr/local/lib/libgmodule*
rm -rf /usr/local/lib/libgio*
rm -rf /usr/local/bin/gtester
rm -rf /usr/local/bin/gobject-query
rm -rf /usr/local/bin/gio*
rm -rf /usr/local/bin/gresource
rm -rf /usr/local/bin/glib*
rm -rf /usr/local/bin/gsettings
rm -rf /usr/local/bin/gdbus-codegen
rm -rf /usr/local/bin/gi-decompile-typelib
rm -rf /usr/local/bin/gi-inspect-typelib
rm -rf /usr/local/bin/gtester-report
rm -rf /usr/local/include/glib*
rm -rf /usr/local/bin/gdbus
rm -rf /usr/local/lib/libgirepository*
rm -rf /usr/local/bin/gi-compile-repository
rm -rf /usr/local/include/pcre*
rm -rf /usr/local/lib/pkgconfig/libpcre2*
rm -rf /usr/local/lib/pkgconfig/glib*
rm -rf /usr/local/lib/pkgconfig/gobject*

find /usr/local -type l | while read -r file; do
    link=$(readlink "${file}")
    if [[ "${link}" == *"Cellar"* && "${link}" == *"glib"* ]]; then
        echo "2. Removing symlink ${file}"
        rm -f "${file}"
    fi
done

# The installer fails when it tries to create this directory and it already
# exists, so clean it before that
rm -rf /usr/local/share/bash-completion

rm -rf /usr/local/include/glib-2.0
rm -rf /usr/local/include/gio-unix-2.0
rm -rf /usr/local/include/brotli
rm -rf /usr/local/include/lame

# Remove all dangling symlinks
find -L /usr/local/bin -type l -exec rm -i {} \;
find -L /usr/local/lib -type l -exec rm -i {} \;
find -L /usr/local/include -type l -exec rm -i {} \;
