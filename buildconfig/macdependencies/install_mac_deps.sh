# A script to install mac deps in /usr/local
set -e -x

bash ./clean_usr_local.sh
python3 -m pip install setuptools
sudo python3 install_mac_deps.py ${GITHUB_WORKSPACE}/pygame_mac_deps_${MAC_ARCH}
