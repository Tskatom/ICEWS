path=$1/$2
echo $path
files=$(find $1/$2 -name '*.json')
date=$2
for f in $files
do
    python /home/tskatom/scratch/prototype/s3_put.py -p ICEWS_MENA_AuditTrail/$2 files $f
    echo $f
done
