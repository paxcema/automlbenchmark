#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="stable"
fi
REPO=${3:-"https://github.com/mindsdb/mindsdb"}
PKG=${4:-"mindsdb"}
RAWREPO=$(echo ${REPO} | sed "s/github\.com/raw\.githubusercontent\.com/")

# creating local venv
. $HERE/../shared/setup.sh $HERE

PIP install --no-cache-dir -r "${RAWREPO}/${VERSION}/requirements.txt"

# PIP install --no-cache-dir openml

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir ${PKG}
else
    PIP install --no-cache-dir ${PKG}==${VERSION}
fi

