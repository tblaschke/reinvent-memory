#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --in_file)
    NONSCORED="$2"
    shift # past argument
    shift # past value
    ;;
    --out_file)
    SCORED="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


echo ${SCORED}

echo ${NONSCORED}

#now call your script, e.g. another python script.
#the variable ${NONSCORED} contains the smiles which you have to score
#the variable ${SCORED} contains the file path where the results should be writting to

#python new_fancy_score ${NONSCORED} > ${SCORED}
