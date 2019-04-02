set -euo pipefail

rm -rf build
python setup.py build
PYTHONPATH=$PYTHONPATH:$(pwd)/build/lib.linux-x86_64-3.7 pytest test/
