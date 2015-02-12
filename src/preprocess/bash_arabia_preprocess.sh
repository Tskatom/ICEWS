#!/bin/bash

start=$1
end=$2
#preprocess protests
python multi_task.py --task topDocumentCount --core 40 --inFolder /raid/tskatom/raw_arabia_inform/ --outFolder /raid/tskatom/seed_protest_document_count --keywordsFile /home/tskatom/workspace/icews_model/data/keywords/seed_protest_keywords.txt  --start $start --end $end

#preprocess coerce
python multi_task.py --task topDocumentCount --core 40 --inFolder /raid/tskatom/raw_arabia_inform/ --outFolder /raid/tskatom/seed_coerce_document_count --keywordsFile /home/tskatom/workspace/icews_model/data/keywords/seed_coerce_keywords.txt  --start $start --end $end

#preprocess protests
python multi_task.py --task topDocumentCount --core 40 --inFolder /raid/tskatom/raw_arabia_inform/ --outFolder /raid/tskatom/seed_assault_document_count --keywordsFile /home/tskatom/workspace/icews_model/data/keywords/seed_assault_keywords.txt  --start $start --end $end
