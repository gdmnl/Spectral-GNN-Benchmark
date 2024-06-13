#!/bin/bash

# set -x
set -e

echo ::group:: Initialize various paths

repo_dir=$GITHUB_WORKSPACE/$INPUT_REPOSITORY_PATH
doc_dir=$repo_dir/$INPUT_DOCUMENTATION_PATH
# https://stackoverflow.com/a/4774063/4799273
action_dir=$GITHUB_ACTION_PATH

echo Action: $action_dir
echo Workspace: $GITHUB_WORKSPACE
echo Repository: $repo_dir
echo Documentation: $doc_dir

echo Adding ~/.local/bin to system path
PATH=$HOME/.local/bin:$PATH
if ! command -v sphinx-build &>/dev/null; then
    echo Sphinx is not successfully installed
    exit 1
else
    echo Everything goes well
fi

echo ::endgroup::

echo ::group:: Creating build directory
build_dir=/tmp/sphinxnotes-pages
mkdir -p $build_dir || true
echo Temp directory \"$build_dir\" is created

echo ::group:: Running Sphinx builder
if ! sphinx-build -b html $INPUT_SPHINX_BUILD_OPTIONS "$doc_dir" "$build_dir"; then
    # See: https://github.com/sphinx-notes/pages/issues/28
    # echo ::endgroup::
    # echo ::group:: Dumping Sphinx error log
    # for l in $(ls /tmp/sphinx-err*); do
    #     cat $l
    # done
    exit 1
fi
echo ::endgroup::

echo "artifact=$build_dir" >> $GITHUB_OUTPUT
