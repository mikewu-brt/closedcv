#!/bin/bash

set -e

cd $(git rev-parse --show-toplevel)
VERSION=$(cat version).${BUILD_NUMBER:-1}
ITERATION=$(git rev-parse --short=8 HEAD)

process_options() {
    # Parse opts:
    while [ $# -gt 0 ]; do
        opt=$(echo $1 | tr [:upper:] [:lower:])
        case $opt in
        "-version")
            shift
            value=$1
            VERSION=$value
            ;;
        "-iteration")
            shift
            value=$1
            ITERATION=$value
            ;;
        "-h")
            print_usage
            exit 0
            ;;
        "-help")
            print_usage
            exit 0
            ;;
        *)
            printf "ERROR: Unknown parameter %s\n" $opt
            print_usage
            exit 1
            ;;
        esac
        shift
    done
}

create_package() {
    printf "version:%s, iteration:%s\n" $1 $2

    make clean && make package

    fpm -s dir \
        -t deb \
        -C install \
        --after-install deb_package_scripts/after_install.sh \
        --deb-pre-depends python3-pip \
        --deb-pre-depends python3-numpy \
        --deb-pre-depends python3-yaml \
        --deb-pre-depends python3-pygments \
        --deb-pre-depends python3-cycler \
        --deb-pre-depends python3-decorator \
        --deb-pre-depends ipython3 \
        --deb-pre-depends python3-matplotlib \
        --deb-pre-depends python3-ipython-genutils \
        --deb-pre-depends python3-jsonschema \
        --deb-pre-depends python3-jedi \
        --deb-pre-depends python3-nose \
        --deb-pre-depends python3-opencv \
        --deb-pre-depends python3-parso \
        --deb-pre-depends python3-pexpect \
        --deb-pre-depends python3-pickleshare \
        --deb-pre-depends python3-pip \
        --deb-pre-depends python3-prompt-toolkit \
        --deb-pre-depends python3-ptyprocess \
        --deb-pre-depends python3-dateutil \
        --deb-pre-depends python3-setuptools \
        --deb-pre-depends python3-six \
        --deb-pre-depends python3-traitlets \
        --deb-pre-depends python3-wheel \
        --name depth_offline_calibration \
        --version $1 \
        --iteration $2 \
        --description "Light Depth Offline Calibration $1 Package" \
        --force
}

process_options "$@"
create_package "$VERSION" "$ITERATION"

exit 0
