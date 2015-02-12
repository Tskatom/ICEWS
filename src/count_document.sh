#!/bin/bash
t_start=$1
t_end=$2

python multi_task.py --task documentCount --core 40 --inFolder /raid/tskatom/raw_arabia_inform --outFolder /raid/tskatom/arabia_handbook_document_count --keywordsFile ../data/keywords/handbook_protest_keywords.txt --start $t_start --end $t_end

python multi_task.py --task documentCount --core 40 --inFolder /raid/tskatom/raw_arabia_inform --outFolder /raid/tskatom/coerce_document_count --keywordsFile ../data/keywords/coerce_keywords.txt --start $t_start --end $t_end

python multi_task.py --task documentCount --core 40 --inFolder /raid/tskatom/raw_arabia_inform --outFolder /raid/tskatom/assault_document_count --keywordsFile ../data/keywords/assault_keywords.txt --start $t_start --end $t_end

