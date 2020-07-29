#!/usr/bin/env bash
HERE=$(dirname "$0")  #/home/pcerdam/Documents/MindsDB/automlbenchmark/frameworks/MindsDB
AMLB_DIR="$1"  # /home/pcerdam/Documents/MindsDB/automlbenchmark
VERSION=${2:-"latest"} # 2.1.2
if [[ "$VERSION" == "latest" ]]; then
    VERSION="stable"
fi
REPO=${3:-"https://github.com/mindsdb/mindsdb"}  # https://github.com/mindsdb/mindsdb
PKG=${4:-"mindsdb"}  # mindsdb
RAWREPO=$(echo ${REPO} | sed "s/github\.com/raw\.githubusercontent\.com/")

# creating local venv
. $HERE/../shared/setup.sh $HERE

PIP install --no-cache-dir -U -r "${RAWREPO}/${VERSION}/requirements.txt"
PIP install --no-cache-dir ${PKG}==${VERSION}