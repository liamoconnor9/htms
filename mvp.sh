#PBS -S /bin/bash
#PBS -l select=1:ncpus=128:mpiprocs=128:model=rom_ait
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -W group_list=s2276
file=${0##*/}
job_name="${file%.*}"
#PBS -N ${job_name}

shopt -s expand_aliases
alias mpiexec_mpt="mpirun"
alias ffmpeg3="ffmpeg"

export PATH=$HOME/scripts:$PATH
deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus3
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true

source ~/png2mp4.sh

CONFIG="mri_options.cfg"
while read -r line; do
    eval $line &>/dev/null || continue
done <$CONFIG
echo $suffix

FILE="$(readlink -f "$0")"
DIR="$(dirname "$(readlink -f "$0")")/"
MAIN="mri.py"


mkdir $suffix
cp $FILE $suffix
cp $CONFIG $suffix
cp $MAIN $suffix
cd $suffix

mpiexec_mpt -np $MPIPROC python3 $MAIN $CONFIG $DIR $suffix
exit 1
cd ..
python plot_scalars.py
# mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs scalars_${suffix} --cleanup
# exit 1
# python3 plotting_scripts/plot_kebe.py scalars_${suffix}/*.h5 --dir=$DIR --config=$CONFIG --suffixix=$suffix
# python3 plotting_scripts/plot_ke.py scalars_${suffix}/*.h5 --dir=$DIR --config=$CONFIG --suffixix=$suffix
# python3 plotting_scripts/plot_be.py scalars_${suffix}/*.h5 --dir=$DIR --config=$CONFIG --suffixix=$suffix

mpiexec_mpt -np $MPIPROC python3 ../plotting_scripts/plot_slicepoints_xy.py slicepoints_${suffix}/*.h5 --output=frames_xy_${suffix} --dir=$DIR --config=$CONFIG --suffixix=$suffix
mpiexec_mpt -np $MPIPROC python3 ../plotting_scripts/plot_slicepoints_xz.py slicepoints_${suffix}/*.h5 --output=frames_xz_${suffix} --dir=$DIR --config=$CONFIG --suffixix=$suffix
mpiexec_mpt -np $MPIPROC python3 ../plotting_scripts/plot_slicepoints_yz.py slicepoints_${suffix}/*.h5 --output=frames_yz_${suffix} --dir=$DIR --config=$CONFIG --suffixix=$suffix
mpiexec_mpt -np $MPIPROC python3 ../plotting_scripts/plot_kebe_profiles.py slicepoints_${suffix}/*.h5 --output=kebe_profiles_${suffix} --dir=$DIR --config=$CONFIG --suffixix=$suffix
png2mp4 frames_xy_${suffix}/ mri_${suffix}_xy.mp4 60
png2mp4 frames_xz_${suffix}/ mri_${suffix}_xz.mp4 60
png2mp4 frames_yz_${suffix}/ mri_${suffix}_yz.mp4 60
png2mp4 kebe_profiles_${suffix}/ kebe_profiles_${suffix}.mp4 60
# # mpiexec_mpt -np $MPIPROC python3 -m dedalus merge_procs checkpoints_${suffix} --cleanup