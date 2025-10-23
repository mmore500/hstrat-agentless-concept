#!/usr/bin/bash

set -e

cd "$(dirname "$0")"
python3 -m uv pip compile requirements.in > requirements.txt
python3 -m uv pip compile --python-version "3.11" requirements_py311.in > requirements_py311.txt
