warn=$1
surr=$2
python /home/tskatom/scratch/prototype/s3_put.py -p incoming/predictions/civilunrest/icews files $1
python /home/tskatom/scratch/prototype/s3_put.py -p incoming/predictions/civilunrest/icews files $2
