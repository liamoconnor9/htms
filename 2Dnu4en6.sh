#PBS -S /bin/bash
#PBS -j oe
#PBS -W group_list=s2276
#STOP

skip_target=false
local=false
if [[ $suffix == "$PBS_JOBNAME" ]]; then
    echo "JOB INITIATED FROM QSUB"
    devel=false
    local=false
    skip_target=false
    skip_sbi_overwrite=false
# elif [[ "STDIN" == "$PBS_JOBNAME" ]]; then
else
    echo "USING DEVEL OPTIONS IN shear_template.sh"
    devel=$1
    local=$2
    suffix=$3
    skip_target=$4
    skip_sbi_overwrite=$5
fi

echo "RESOURCES ALLOCATED: $RLARG"
# echo "RESOURCE_LIST: $resource_list"
# echo "ncpus: $ncpus"
# echo "model: $model"

export PATH=$HOME/scripts:$PATH
deactivate &>/dev/null
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus3

# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true
# alias True=$true
# alias False=$false

DIR="$(dirname "$(readlink -f "$0")")/"

echo "OGSUFFIX = $suffix"
if [[ -z $suffix ]]; then
    file=${0##*/}
    suffix="${file%.*}"
fi

CONFIG="$suffix/$suffix.cfg"
echo "config = $CONFIG"
skip_sbi=0
oldsuffix=$suffix

# source $CONFIG &>/dev/null
while read -r line; do
    eval $line &>/dev/null || continue
done <$CONFIG

if ! [[ "$suffix" == "$oldsuffix" ]]; then
    echo "CONFIG SUFFIX SHOULD MATCH ARG/JOBFILE. TERMINATING"
    exit 1
fi

if [[ "$local" == "True" ]]; then
    MPIPREFFIX="mpirun -n"
else
    MPIPREFFIX="mpiexec_mpt -np"
fi

echo "SUFFIX = $suffix"
if [[ "$solveEVP" == "True" ]]; then
    $MPIPREFFIX $MPIPROC python3 evp.py $CONFIG
else
    $MPIPREFFIX $MPIPROC python3 mri.py $CONFIG
fi

# if $PLOT_SCALARS; then
#     python plot_scalars.py $CONFIG
# fi