#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

if [ "$TRAVIS_OS_NAME" == 'osx' ]; then
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
  PYTHON="$HOME/miniconda/bin/python$PYTHON_VERSION"
  PYTEST="$HOME/miniconda/bin/pytest"
else
  PYTHON=${PYTHON:-python}
  #PYTEST=${PYTEST:-"pytest -rxXs --cov=slmpy/"}
  PYTEST=${PYTEST:-"pytest -rxXs"}
fi

echo "python: ${PYTHON}"

echo 'Running pytests...'
# LOCAL TESTING:
# PYTHONPATH=$(pwd):PYTHONPATH SINGLET_CONFIG_FILENAME='example_data/config_example.yml' pytest -rxXs test

${PYTEST} "test"
