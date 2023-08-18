#!/bin/bash

REUSE=false
SKIPSBI=false
config="config.cfg"
args="-f -d"
while getopts 'rsl' OPTION; do
  case "$OPTION" in
    r)
      REUSE=true
      args="${args} -r"
      ;;
    s)
      SKIPSBI=true
      args="${args} -s"
      ;;
    l)
      SKIPSBI=true
      args="${args} -l"
      ;;
    ?)
      REUSE=false
      ;;
  esac
done
shift "$(($OPTIND -1))"

if ! [ -z $1 ]; then
    config=$1
fi

echo $config
# echo $REUSE
# echo $SKIPSBI
echo $args
# exit 1
source job_factory.sh
factory $args $config
