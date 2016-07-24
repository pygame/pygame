#!/bin/bash
set -e -x

rpm --import /io/manylinux-build/RPM-GPG-KEY.dag.txt

if [ "$(uname -i)" = "x86_64" ]; then
    RPMFORGE_FILE="rpmforge-release-0.5.3-1.el5.rf.x86_64.rpm"
else
    RPMFORGE_FILE="rpmforge-release-0.5.3-1.el5.rf.i386.rpm"
fi

wget http://pkgs.repoforge.org/rpmforge-release/${RPMFORGE_FILE}
rpm -i ${RPMFORGE_FILE}
