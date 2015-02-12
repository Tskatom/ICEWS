#!/bin/bash

# $Id: $
folder=$1
files=$(ls $folder)
for f in $files
do 
    echo $f
    python /home/tskatom/scratch/prototype/s3_put.py -p ICEWS_MENA_AuditTrail files $folder/$f
done

