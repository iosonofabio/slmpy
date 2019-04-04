set -euo pipefail
if [ $1 == "build" ]; then
  rm -rf build
  python setup.py build
fi

PYTHONPATH=$PYTHONPATH:$(pwd)/build/lib.linux-x86_64-3.7 pytest test/
