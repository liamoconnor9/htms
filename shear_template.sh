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

if $local; then
    MPIPREFFIX="mpirun -n"
else
    MPIPREFFIX="mpiexec_mpt -np"
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
alias True=$true
alias False=$false

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

echo "SUFFIX = $suffix"
echo "skip_targ = $skip_target"
echo "skip_sbi = $skip_sbi"
echo "skip_sbi_overwrite = $skip_sbi_overwrite"
# rm $suffix/purge
if $skip_target; then
    echo "SKIPPING TARGET SIMULATION ..."
elif [[ "$skip_target" == "True" ]] || [[ "$skip_target" == "1" ]]; then
    echo "SKIPPING TARGET SIMULATION ..."
else
    python3 make_dirs.py $CONFIG
    $MPIPREFFIX $MPIPROC python3 shear_flow.py $CONFIG
    if [[ "$show" == "True" ]] || [[ "$show" == "1" ]]; then
        echo "PERFORMING TARGET SNAPSHOT PROCESSING ..."
        mpirun -n $MPIPROC python3 plot_snapshots_og.py $suffix snapshots_target frames_target
        source ~/png2mp4.sh
        png2mp4 $suffix/frames_target/ $suffix/movie_target.mp4 60 
        echo "$DIR$suffix/movie_target.mp4"
    else
        echo "SKIPPING TARGET SNAPSHOT PROCESSING ..."
    fi
fi

if [[ "$skip_sbi" == "True" ]] || [[ "$skip_sbi" == "1" ]]; then
    echo "SKIPPING SIMPLE BACKWARD INTEGRATION (SBI) ..."
elif $skip_sbi_overwrite; then
    echo "DEVEL OPTION: SKIPPING SIMPLE BACKWARD INTEGRATION (SBI) ..."
else
    echo "PERFORMING SIMPLE BACKWARD INTEGRATION (SBI) ..."
    $MPIPREFFIX $MPIPROC python3 shear_abber.py $CONFIG SBI.cfg
fi

$MPIPREFFIX $MPIPROC python3 shear_abber.py $CONFIG
