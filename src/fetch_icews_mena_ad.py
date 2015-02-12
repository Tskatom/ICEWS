#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import boto
import argparse
import codecs

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--prefix', type=str, default='ICEWS_MENA_AuditTrail',
            help='the prefix of path in S3')
    ap.add_argument('-k', '--key', type=str, default=os.environ['AWS_ACCESS_KEY_ID'],
            help='The AWS KEY')
    ap.add_argument('-s', '--secret', type=str, default=os.environ['AWS_SECRET_ACCESS_KEY'])
    ap.add_argument('-w', '--warn_ids', type=str, nargs="+")
    ap.add_argument('-b', '--bucket', type=str, default='embers-osi-data')
    return ap.parse_args()

def fetch(args):
    """
    Download the audit trail from s3 according to warning embersID
    """
    conn_s3 = boto.connect_s3(args.key, args.secret)
    bucket = conn_s3.get_bucket(args.bucket)
    current_folder = args.prefix.split("/")[-1]
    if not os.path.exists(current_folder):
        os.mkdir(current_folder)

    for eid in args.warn_ids:
        ad_name = os.path.join(args.prefix, "%s.json" % eid)
        ad_s3 = bucket.list(prefix=ad_name)
        for k in ad_s3:
            outfile = os.path.join(current_folder, "%s.json" % eid)
            with open(outfile, 'w') as otf:
                with codecs.getwriter('utf-8')(otf, 'replace') as f:
                    for b in k:
                        f.write(b.decode('utf-8', 'replace'))

def main():
    args = parse_args()
    fetch(args)

if __name__ == "__main__":
    main()

