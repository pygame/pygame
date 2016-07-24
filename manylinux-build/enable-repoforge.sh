#!/bin/bash
set -e -x

rpm --import /io/manylinux-build/RPM-GPG-KEY.dag.txt

if [ "$(uname -i)" = "x86_64" ]; then
    RPMFORGE_FILE="rpmforge-release-0.5.3-1.el5.rf.x86_64.rpm"
    RPMFORGE_URL="https://repoforge.cu.be/redhat/el5/en/x86_64/dag/RPMS/rpmforge-release-0.5.3-1.el5.rf.x86_64.rpm"
else
    RPMFORGE_FILE="rpmforge-release-0.5.3-1.el5.rf.i386.rpm"
    RPMFORGE_URL="https://repoforge.cu.be/redhat/el5/en/i386/dag/RPMS/rpmforge-release-0.5.3-1.el5.rf.i386.rpm"
fi

# wget http://pkgs.repoforge.org/rpmforge-release/${RPMFORGE_FILE}
wget ${RPMFORGE_URL}
rpm -i ${RPMFORGE_FILE}

