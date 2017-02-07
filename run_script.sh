#! /bin/bash
# Execute one of the analysis scripts with the extremely simple
# parallelization method of launching a bunch of backgrounded processes

script=${1}.py
exp=$2
subjects=`cat lyman/${exp}_subjects.txt`
if [ -z "$3" ]; then roi=ifs; else roi=$3; fi

for subj in $subjects; do
    python $script $subj $exp $roi &
done
