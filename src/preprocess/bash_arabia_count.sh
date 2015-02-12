#!/bin/bash
# process assault events
python arabia_count.py -d /raid/tskatom/seed_assault_document_count -t assault -o /home/tskatom/workspace/icews_model/data/features/ArabiaInform/related_doc/

#preprocess coerce events
python arabia_count.py -d /raid/tskatom/seed_coerce_document_count -t coerce -o /home/tskatom/workspace/icews_model/data/features/ArabiaInform/related_doc/

#preprocess protest events
python arabia_count.py -d /raid/tskatom/seed_protest_document_count -t protest -o /home/tskatom/workspace/icews_model/data/features/ArabiaInform/related_doc/



