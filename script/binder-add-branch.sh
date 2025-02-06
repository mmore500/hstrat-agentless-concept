#!/bin/bash

cd "$(dirname "$0")/.."

# Fetch all remote branches
git fetch --all --jobs $(nproc)
git fetch origin

mkdir -p binder

# Loop through remote branches starting with 'binder'
for branch in $@; do
    echo "Processing branch $branch"

    if [ -d binder/$branch ]; then
        echo "Skipping branch $branch"
        continue
    fi

    # Add the branch as a submodule
    git clone --branch "$branch" --single-branch "https://github.com/mmore500/hypermutator-dynamics.git" "binder/$branch" --jobs $(nproc) --depth 1
    git submodule add --branch "$branch" "https://github.com/mmore500/hypermutator-dynamics.git" "binder/$branch"
    git config -f .gitmodules submodule."binder/$branch".branch "$branch"
    git config -f .gitmodules submodule."binder/$branch".shallow true

done
