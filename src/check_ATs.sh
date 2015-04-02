#!/bin/bash

# $Id: $
folder=$1
files=$(ls $folder)
for f in $files
do
    atf=$1/$f
    python /raid/home/nwself/schema/warning_check.py $atf
done

