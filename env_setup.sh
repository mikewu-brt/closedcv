# Add necessary search paths for Python:
CLOSEDCV=$(cd $(dirname "$BASH_SOURCE"); pwd)
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=$CLOSEDCV
else
    export PYTHONPATH=$CLOSEDCV:${PYTHONPATH}
fi
