#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# LOCAL TESTING:
# PYTHONPATH=$(pwd):PYTHONPATH SINGLET_CONFIG_FILENAME='example_data/config_example.yml' pytest -rxXs test

echo 'Running pytests...'

if [ "$TRAVIS_OS_NAME" == 'osx' ]; then
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
  #PYTHONPATH=$(pwd):PYTHONPATH $HOME/miniconda/bin/pytest -rxXs
  $HOME/miniconda/bin/pytest -rxXs

else
  #PYTHONPATH=$(pwd):PYTHONPATH pytest -rxXs --cov=slmpy/test
  pytest -rxXs --cov=slmpy/test
fi

echo "Tests done!"
