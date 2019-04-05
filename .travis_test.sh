#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# LOCAL TESTING:
# PYTHONPATH=$(pwd):PYTHONPATH SINGLET_CONFIG_FILENAME='example_data/config_example.yml' pytest -rxXs test

echo 'Running pytests...'

if [ "$TRAVIS_OS_NAME" == 'osx' ]; then
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
  $HOME/miniconda/bin/pytest -rxXs

else
  pytest -rxXs --cov=slmpy/test
fi

echo "Tests done!"
