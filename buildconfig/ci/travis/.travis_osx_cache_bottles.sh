#!/usr/bin/env bash

source "buildconfig/ci/travis/.travis_osx_before_install.sh" --no-installs

# cache bottles with long build times

set +e
install_or_upgrade boost & prevent_stall
install_or_upgrade openssl
install_or_upgrade python@2
install_or_upgrade glib
install_or_upgrade cmake
set -e
