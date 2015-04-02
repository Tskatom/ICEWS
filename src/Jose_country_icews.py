#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ICEWS related operation
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import config
import argparse
import glob
from etool import message
import json
from dateutil import parser

class Icews(object):

    def __init__(self, version, raw_folder, out_folder):
        """
        Initiate the ICEWS parameters
        version: the latest version of icews data file
        raw_folder: the raw icews data file
        out_folder: the output folder of the daily count
        """
        self.version = version
        self.raw_folder = raw_folder
        self.out_folder = out_folder
        self.country_list = {"Argentina":1, "Brazil":1, "Chile":1, "Colombia":1, "Ecuador":1,
                "El Salvador":1, "Mexico":1, "Paraguay":1, "Uruguay":1, "Venezuela":1,
                "Iraq":1, "Egypt":1, "Libya":1, "Jordan":1, "Bahrain":1,
                "Syria":1, "Saudi Arabia":1}

        self.capital_city = {
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
            "Saudi Arabia": "Riyadh"}

        self.daily_count = {}
        self.daily_events = {}

    def read_events(self):
        """
        Read the required events in raw data files
        """
        files = glob.glob(os.path.join(self.raw_folder, "*"))

        for f in files:
            with open(f) as icews_f:
                keys = icews_f.readline().strip().split("\t")[:-2]
                for line in icews_f:
                    infos = line.strip().split("\t")
                    event = {keys[i]:infos[i] for i in range(len(keys))}
                    country = event.get("Country", "")
                    if self.country_list.get(country, 0) == 0: # country not in interested
                        continue
                    target_city = self.capital_city.get(country, "")
                    eventText = event['Event Text'].lower().replace(",","")
                    eventDate = parser.parse(event['Event Date']).strftime("%Y-%m-%d")
                    if eventDate > '2015-03-30' or eventDate < '2015-03-23':
                        continue

                    event = message.add_embers_ids(event)

                    self.daily_events.setdefault(country, {})
                    self.daily_events[country].setdefault(eventDate, [])
                    self.daily_events[country][eventDate].append(event)

                    #city event count

                    self.daily_events.setdefault(target_city, {})
                    found = False
                    if country == "Brazil":
                        city = event["Province"]
                        if city == target_city:
                            found = True
                    elif country == "Egypt":
                        city = event["City"]
                        if city == target_city or city == "Tahrir Square":
                            found = True
                    else:
                        city = event["City"]
                        if city == target_city:
                            found = True

                    if found:
                        self.daily_events[target_city].setdefault(eventDate, [])
                        self.daily_events[target_city][eventDate].append(event)


    def write_out_events(self):
        event_folder = self.out_folder if self.out_folder[-1]!='/' else self.out_folder[:-1]
        event_folder = event_folder + "_region"
        if not os.path.exists(event_folder):
            os.mkdir(event_folder)

        version_folder = os.path.join(event_folder, self.version)
        if not os.path.exists(version_folder):
            os.mkdir(version_folder)

        for region, daySet in self.daily_events.items():
            days = sorted(daySet.keys())
            event_file = os.path.join(version_folder, region.replace(" ", "_"))
            with open(event_file, "w") as outf:
                for day in days:
                    outf.write(json.dumps({"day": day, "events": daySet[day]}) + "\n")

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inFolder", type=str,
                    default="/home/tskatom/workspace/icews_data")
    ap.add_argument("--outFolder", type=str, default="../data/icews_gsr")
    ap.add_argument("--version", type=str)
    args = ap.parse_args()
    return args


def main():
    args = parse()
    inFolder = args.inFolder
    outFolder = args.outFolder
    version = args.version

    icews = Icews(version, inFolder, outFolder)
    icews.read_events()
    icews.write_out_events()

if __name__ == "__main__":
    sys.exit(main())

