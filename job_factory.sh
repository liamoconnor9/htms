# !/bin/bash

factory () {
    # flag defaults
    FORCE=false
    RUN_DEVEL=false
    SKIP_TARGET=false
    SKIP_SBI_OVER=false
    DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/"
    # cd $SCRIPT_DIR
    # SCRIPT_DIR=$(dirname "$0")
    LOCAL=false
    # JOB="${}/submission.sh"
    JOB="submission.sh"
    CONFIG="config.cfg"

    while getopts 'fdrsal:' OPTION; do
    case "$OPTION" in
        f)
        FORCE=true
        ;;
        d)
        RUN_DEVEL=true
        ;;
        r)
        SKIP_TARGET=true
        echo "SKIP_TARGET = $SKIP_TARGET"
        ;;
        s)
        SKIP_SBI_OVER=true
        ;;
        l)
        LOCAL=true
        ;;
        a)
        avalue="$OPTARG"
        echo "The value provided is $OPTARG"
        ;;
        ?)
        echo "script usage: $(basename \$0) [-l] [-h] [-a somevalue]" >&2
        # exit 1
        ;;
    esac
    done
    shift "$(($OPTIND -1))"

    # DIR="$(dirname "$(readlink -f "$0")")/"

    echo "FLAGS: FORCE=$FORCE, RUN_DEVEL=$RUN_DEVEL"
    echo "ARG: $1"
    echo "PREPARING JOB SUBMISSION: ##########################################"
    # should be able to overwrite something like $suffix with optional cmd arg

    # if test -f "$CONFIG" ; then
    #     echo "config supplied with specificity"
    # elif test -d "$DIR$1/" ; then
    #     CONFIG="$1.cfg"
    # else
    #     CONFIG="default.cfg"
    # fi
    echo "CONFIG: $CONFIG"

    if test -f "$DIR$JOB"; then

        if test -f "$DIR$CONFIG"; then
            # echo "READING CONFIG $DIR$CONFIG..."

            # defaults can be overwritten w/ config or optional cmd args
            suffix="temp"
            MPIPROC=0
            nodes=1
            walltime="02:00:00"
            
            # source $CONFIG &>/dev/null
            while read -r line; do
                eval $line &>/dev/null || continue
            done <$CONFIG
            
            echo "SUFFIX: $suffix"
            echo "PARENT DIRECTORY: $DIR"
            echo "JOB SCRIPT: $JOB"
            echo "MPIPROC: $MPIPROC"
            if [[ $suffix == *"load"* ]]; then
                SKIP_TARGET=true
                echo "suffix contains load indicator; SKIP_TARGET=true"
            elif [[ $load_state == "True" ]]; then
                SKIP_TARGET=true
                echo "config has load_state=True; SKIP_TARGET=true"
            fi
            echo ""
            if $SKIP_TARGET && test -d "$DIR$suffix/"; then
                echo "EXISTING RUN IDENTIFIED"
            elif test -d "$DIR$suffix/"; then
                echo "RUN DIRECTORY ALREADY EXISTS!"
                if ! $FORCE; then
                    read -p "PRESS ANY KEY TO OVERWRITE EXISTING DIRECTORY: $DIR$suffix/"
                    echo "KILLING JOB IF NOT COMPLETED."
                fi
                echo "REMOVING DIRECTORY AND FILES: $DIR$suffix/ $DIR$suffix.o* $DIR$suffix.sh*"
                $DIR/$suffix/killjob
                rm -rf $DIR$suffix/
                rm $DIR$suffix.o*
                rm $DIR$suffix.sh*
            fi
            
            if ! $SKIP_TARGET; then
                echo "CREATING NEW RUN DIRECTORY: $DIR$suffix/"
                mkdir $DIR$suffix
            fi

            # echo "copying $CONFIG to $DIR$suffix.cfg and $DIR$suffix/$suffix.cfg"
            echo "PERMANENT COPY $CONFIG --> $suffix/$suffix.cfg"
            cp $CONFIG $suffix/$suffix.cfg
            # cp $CONFIG $suffix.cfg
            
            # echo "copying $DIR$JOB to $DIR$suffix.sh and $DIR$suffix/$suffix.sh"
            echo "TEMPORARY COPY $JOB --> $suffix.sh"
            cp $JOB $suffix.sh
            # cp $JOB $suffix/$suffix.sh


            echo ""
            echo "CONFIG OPTIONS SNIPET:"
            LINENUM=1
            while read -r line; do
                name="$line"
                if (( LINENUM > 9 )); then
                    echo "$LINENUM)   $name"
                else
                    echo "$LINENUM)    $name"
                fi
                ((LINENUM++))
                if (( LINENUM > 15 )); then
                    break
                fi
            done < "$DIR$suffix/$suffix.cfg"

            echo ""


            if (( MPIPROC == -32 )); then
                echo "MPIPROC NOT SET IN CONFIG!"
                read -p "INPUT DESIRED MPIPROC:  " MPIPROC
                if [ -z $MPIPROC ]; then
                    MPIPROC=32
                    echo "USING DEFAULT: MPIPROC=$MPIPROC"
                fi
            fi

            if [ -z $model ]; then
                if (( MPIPROC < 28 )); then
                    model="bro"
                    ncpus=28
                elif (( MPIPROC < 40 )); then
                    model="sky_ele"
                    ncpus=40
                elif (( MPIPROC < 129 )); then
                    model="rom_ait"
                    ncpus=128
                fi
            elif [ -z $ncpus ]; then
                ncpus=$MPIPROC
            fi

            while [[ 1 ]]; do
                resource_list="select=$nodes:ncpus=$ncpus:mpiprocs=$ncpus:model=$model -l walltime=$walltime"
                RLARG="MPIPROC=$MPIPROC, nodes=$nodes, ncpus=$ncpus, model=$model, walltime=$walltime"
                if ! $FORCE; then
                    echo "INPUT A DEFINITION LIKE SO TO CHANGE RESOURCE: 'MPIPROC=$MPIPROC', 'nodes=$nodes', 'ncpus=$ncpus', 'model=$model', 'walltime=$walltime'"
                    # echo "INPUT NEW DEFINITION TO CHANGE RESOURCE: 'nodes=$nodes', 'ncpus=$ncpus', 'model=$model', 'walltime=$walltime'"
                    echo "OR PRESS ENTER TO SUBMIT JOB W/ ARGS: -N $suffix -o $suffix/logger.txt -l $resource_list"
                    echo ""
                    read -p "PRESS ENTER TO SUBMIT: "  EVALARGS
                fi
                if [[ -z $EVALARGS ]]; then

                    # creating executables in new directory: purge, tweak, code

                    echo "bash ${DIR}CodeConfig.sh $suffix" > $DIR$suffix/tweak
                    chmod +x $DIR$suffix/tweak

                    echo "code ${DIR}$suffix/$suffix.cfg" > $DIR$suffix/code
                    chmod +x $DIR$suffix/code

                    if ! $RUN_DEVEL; then
                        OUT_QSUB=$(qsub -o $suffix/logger.txt -v "suffix=$suffix" -l select=$nodes:ncpus=$ncpus:mpiprocs=$ncpus:model=$model -l walltime=$walltime -N "$suffix" $DIR$suffix.sh)
                        TIMESTAMP=$(bash Timestamp.sh)
                        echo ""
                        echo "JOBID:    ${OUT_QSUB}"
                        echo "JOBNAME:  ${suffix}"

                        echo "TIMESTAMP:  ${TIMESTAMP}"   >>  ~/JOBLOG.txt
                        echo "JOBID:      ${OUT_QSUB}"    >>  ~/JOBLOG.txt
                        echo "JOBNAME:    ${suffix}"      >>  ~/JOBLOG.txt
                        cp ~/JOBLOG.txt $DIR/JOBLOG.txt

                        echo "echo "$OUT_QSUB" " > $DIR$suffix/jobid
                        chmod +x $DIR$suffix/jobid
                        
                        jobid=$($DIR$suffix/jobid)
                        echo "qdel $jobid" > $DIR$suffix/killjob
                        chmod +x $DIR$suffix/killjob
                        
                        echo "$DIR$suffix/killjob" > $DIR$suffix/purge
                        echo "bash ${DIR}Purge.sh $suffix" >> $DIR$suffix/purge
                        chmod +x $DIR$suffix/purge

                        echo ""

                        echo "SUBMITTED. MOVING JOB SCRIPT TO NEW DIRECTORY $DIR$suffix/$suffix.sh"
                        mv $DIR$suffix.sh $DIR$suffix/$suffix.sh
                        echo "JOB SHOULD BE LISTED IN USER'S QUEUE:"
                        qstat -u $USER
                        echo ""
                    else
                        echo $"RUNNING JOB SCRIPT WITH BASH (YOU SHOULD HAVE CORES)"
                        bash $DIR$suffix.sh $RUN_DEVEL $LOCAL $suffix $SKIP_TARGET $SKIP_SBI_OVER
                        # exit 1
                    fi
                    # qsub -o $suffix/logger.txt -v "suffix=$suffix" -l select=$nodes:ncpus=$ncpus:mpiprocs=$ncpus:model=$model -l walltime=$walltime -N "$suffix" $DIR$suffix.sh
                    echo ""
                    # qsub -v "suffix=$suffix nodes=$nodes" -N "$suffix" $DIR$suffix.sh
                    break
                else
                    eval "$EVALARGS"
                fi
            done

            read -p "HIT ENTER TO TWEAK SUBMITTED JOB: $suffix"
            bash CodeConfig.sh $suffix

            # qsub -v "suffix=$suffix nodes=$nodes ncpus=$ncpus model=$model walltime=walltime" -N "$suffix" $DIR$suffix.sh


        else
            echo "config file DNE!!!"
            echo "$DIR$CONFIG not found. Aborting job prep and submission..."
            # exit 1
        fi
    else
        echo "bash script DNE!!!"
        echo "$DIR$JOB not found. Aborting job prep and submission..."
        # exit 1
    fi
}