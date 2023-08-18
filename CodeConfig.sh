#!/bin/bash

DIR="$(dirname "$(readlink -f "$0")")/"
template=$1
if [[ -z $template ]]; then
    echo "SPECIFY TEMPLATE CONFIG OR DIRECTORY!"
    echo "using config.cfg"
    template="config.cfg"

elif test -d "$DIR$template/"; then
    template=$template/$template.cfg
fi

echo "COPYING $template"
cp $template config.cfg

# if [[ "$PBS_JOBNAME" == "STDIN" ]]; then
if ! command -v code &> /dev/null ;
then
    echo "<the_command> could not be found"
    echo "VSCODE COMMAND LINE INTERFACE INACCESSIBLE, USING VIM (GOOD LUCK) ... "
    read -p ""
    vim config.cfg

else
    code config.cfg
fi


echo "HIT ENTER TO SEND JOB USING: config.cfg"
read -p ""

# source config.cfg &>/dev/null
# echo ""
# echo "SUFFIX READ FROM NEW CONFIG: $suffix"
# cp config.cfg $suffix.cfg

args=""
if [[ "$PBS_QUEUE" == "devel" ]]; then
    echo "ENTER ARGS FOR DEVEL?"
    read -p "" args
fi

echo ""
# echo "PRESS ENTER TO CONTINUE TO JOB FACTORY W/ ARGS: $args"
# read -p "" 
# bash JobFactory.sh $args config.cfg
source job_factory.sh
factory
