#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Generate the Audit Trail for ICEWS MENA Warning
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import os
from etool import message
import json
import re
from math import ceil
from datetime import datetime, timedelta
from dateutil import parser
import argparse

CAPITAL_COUNTRY = {
        "Argentina": "Buenos Aires",
        "Brazil": "Distrito Federal",
        "Chile": "Santiago",
        "Colombia": "Bogota",
        "Ecuador": "Quito",
        "El Salvador": "San Salvador",
        "Mexico": "Mexico City",
        "Paraguay": "Asuncion",
        "Uruguay": "Montevideo",
        "Venezuela": "Caracas",
        "Iraq": "Baghdad",
        "Egypt": "Cairo",
        "Libya": "Tripoli",
        "Jordan": "Amman",
        "Bahrain": "Manama",
        "Syria": "Damascus",
        "Saudi Arabia": "Riyadh"
        }


def wrapper_fusion(warning):
    """
    Generating a fake fusion message for original ICEWS warning.
    """
    fusion_mess = {}
    fusion_mess["classification"] = None
    fusion_mess["comments"] = warning["comments"]
    fusion_mess["confidence"] = warning["confidence"]
    fusion_mess["confidenceIsProbability"] = warning["confidenceIsProbability"]
    fusion_mess["coordinates"] = None
    fusion_mess["date"] = warning["date"]
    fusion_mess["distance"] = None
    fusion_mess["eventDate"] = warning["eventDate"]
    fusion_mess["eventType"] = warning["eventType"]
    fusion_mess["feed"] = None
    fusion_mess["feedPath"] = None
    fusion_mess["location"] = warning["location"]
    fusion_mess["location_popln_size"] = None
    fusion_mess["model"] = warning.get("model", "")
    fusion_mess["old_location"] = warning["location"]
    fusion_mess["old_location_popln_size"] = None
    fusion_mess["nearbyWarnings_count"] = None
    fusion_mess["population"] = warning["population"]
    fusion_mess["qs_prediction"] = None

    fusion_mess["derivedFrom"] = {
            "comments": None,
            "derivedMessages": [warning],
            "end": None,
            "start": None,
            "source": None,
            "fusionMessage": None
            }

    fusion_mess = message.add_embers_ids(fusion_mess)

    return fusion_mess


def extend_surr(surr, warn):
    """
    Devide the current surr message into three surrogate message:
        ICEWS historical surrogate,
        Arabia Inform surrogate,
        Gnip twitter surrogate
    """
    def gene_surr(summary_count, derived_from, mess_type, warn):
        sub_surr = {}
        sub_surr["summaryCounts"] = summary_count
        sub_surr["comments"] = None
        sub_surr["version"] = warn["version"]
        sub_surr["date"] = warn["date"]
        sub_surr["messageType"] = mess_type
        sub_surr["derivedFrom"] = derived_from
        sub_surr = message.add_embers_ids(sub_surr)
        return sub_surr

    icews_daily = surr["icews_daily"]
    coerce_doc_daily = surr["coerce_doc_daily"]
    protest_doc_daily = surr["protest_doc_daily"]
    assault_doc_daily = surr["assault_doc_daily"]

    #generate ICEWS historical sub surrogate
    code2name = {"0614": "protest", "0617": "coerce", "0618": "assault"}
    icews_key_name = code2name[warn["eventType"]]
    icews_summary_count = {icews_key_name: icews_daily}
    icews_mess_type = "ICEWS"
    #get past 1 month icews raw message
    event_date = warn["eventDate"]
    event_types = [warn["eventType"]]
    location = warn["location"]
    if type(location) == list:
        country = location[0]
    else:
        location = json.loads(location)
        country = location[0]
    city = location[2]
    if city == "-":
        city = None
    else:
        city = CAPITAL_COUNTRY[country]

    #use past one week's data as real data
    end_day = (parser.parse(event_date) - timedelta(days=3)).strftime("%Y-%m-%d")
    start_day = (parser.parse(event_date) - timedelta(days=9)).strftime("%Y-%m-%d")
    icews_der_msgs = get_raw_icews(start_day, end_day, event_types, country, city)
    icews_derived_from = {"derivedMessages": icews_der_msgs}
    icews_sub_surr = gene_surr(icews_summary_count, icews_derived_from, icews_mess_type, warn)

    #generate Arabia Inform sub surrogate
    arabia_summary_count = {
            "protest": protest_doc_daily,
            "coerce": coerce_doc_daily,
            "assault": assault_doc_daily
            }
    arabia_mess_type = "ARABIA"
    arabia_der_msgs = get_raw_arabia(start_day, end_day, country)
    arabia_derived_from = {"derivedMessages": arabia_der_msgs}
    arabia_sub_surr = gene_surr(arabia_summary_count, arabia_derived_from, arabia_mess_type, warn)

    return icews_sub_surr, arabia_sub_surr

def get_raw_icews(start, end, event_types, country, city, icews_folder="../data/icews_gsr_events"):
    """
    Ingest the raw icews message according to the start and end day
    for that location
    """
    #find the lastest version of the ICEWS
    folders = os.listdir(icews_folder)
    latest_ver = str(max(map(int, folders)))
    version_folder = os.path.join(icews_folder, latest_ver)
    results = []
    for e_type in event_types: #e_type format 06xx
        event_folder = os.path.join(version_folder, e_type[2:])
        if city is not None: #city level messages
            location = city
        else:
            location = country
        event_file = os.path.join(event_folder, location.replace(" ", "_"))
        with open(event_file) as ef:
            for line in ef:
                day_events = json.loads(line)
                event_date = day_events["day"]
                if event_date > end or event_date < start:
                    continue
                else:
                    results += day_events["events"]
    return results


def get_raw_arabia(start, end, country):
    protest_folder = "/raid/tskatom/arabia_handbook_document_count_filter"
    coerce_folder = "/raid/tskatom/coerce_document_count_filter"
    assault_folder = "/raid/tskatom/assault_document_count_filter"

    def get_data(folder):
        day_files = []
        for f in os.listdir(folder):
            searched = re.search("(\d{4}-\d{2}-\d{2})", f)
            if searched:
                day = searched.group()
                if day <= end and day >= start:
                    day_files.append(f)
        #event_folder = "/raid/tskatom/raw_arabia_inform"
        events = []
        folder_type = os.path.basename(folder).split("_")[0]
        if folder_type == "arabia":
            ratio = 1.0
        else:
            ratio = 0.1

        for f in day_files:
            #index_file = os.path.join(folder, f)
            #json_index = json.load(open(index_file))
            #country_set = json_index.get(country, {})
            #load the real events according to embersId
            #embersIds = {}
            #for day, ids in country_set.items():
            #    embersIds.update({i:1 for i in ids})

            event_file = os.path.join(folder, f)
            event_set = json.load(open(event_file))
            country_set = event_set.get(country, {})

            for day, day_set in country_set.items():
                #choose 10% of the events
                m_index = int(ceil(len(day_set) * ratio))
                print m_index
                events += day_set[:m_index]
        return events

    results = []
    folders = [protest_folder]
    #folders = [protest_folder, coerce_folder, assault_folder]
    for folder in folders:
        results += get_data(folder)

    return results

class AuditTrailFactory():
    def gene_audit_trail(self, warn, surr):
        icews_surr, arabia_surr = extend_surr(surr, warn)
        #modify the current warning structure
        del warn["derivedFrom"]["derivedIds"]
        warn["derivedFrom"]["derivedMessages"] = [icews_surr, arabia_surr]

        audit_trail = wrapper_fusion(warn)
        warn_id = warn["embersId"]
        return warn_id, audit_trail

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--warn', type=str, help="warn file")
    ap.add_argument('--surr', type=str, help="surr file")
    ap.add_argument('--out', type=str, help='out folder')
    args = ap.parse_args()

    outfolder = args.out
    current_day = datetime.now().strftime('%Y-%m-%d')
    #current_day = "2014-12-16"
    outfolder = os.path.join(outfolder, current_day)

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    factory = AuditTrailFactory()
    with open(args.warn) as wf, open(args.surr) as sf:
        warnings = [json.loads(l) for l in wf]
        surrs = [json.loads(l) for l in sf]
        surr_dict = {s["embersId"]:s for s in surrs}
        for w in warnings:
            surr_id = w["derivedFrom"]["derivedIds"][0]
            surr = surr_dict[surr_id]
            warn_id, audit_trail = factory.gene_audit_trail(w, surr)
            out_f = os.path.join(outfolder, "%s.json" % warn_id)
            with open(out_f, 'w') as otf:
                json.dump(audit_trail, otf)


