#!/bin/bash
t_start=$1
t_end=$2

python multi_task.py --task keywordsCount --core 40 --inFolder /raid/tskatom/raw_arabia_inform --outFolder /raid/tskatom/arabia_mouna_keywords_count --keywordsFile ../data/keywords/mounah_protest_keywords.txt --start $t_start --end $t_end

python multi_task.py --task keywordsCount --core 40 --inFolder /raid/tskatom/raw_arabia_inform --outFolder /raid/tskatom/coerce_keywords_count --keywordsFile ../data/keywords/coerce_keywords.txt --start $t_start --end $t_end

python multi_task.py --task keywordsCount --core 40 --inFolder /raid/tskatom/raw_arabia_inform --outFolder /raid/tskatom/assault_keywords_count --keywordsFile ../data/keywords/assault_keywords.txt --start $t_start --end $t_end


