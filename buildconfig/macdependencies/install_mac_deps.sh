# A script to install mac deps in /usr/local
set -e -x

bash ./clean_usr_local.sh
sudo python3 install_mac_deps.py ${GITHUB_WORKSPACE}/pygame_mac_deps

# strangely somehow the built pygame links against the libportmidi.dylib here, so
# copy the dylib
cp /usr/local/lib/libportmidi.dylib ${GITHUB_WORKSPACE}/libportmidi.dylib
