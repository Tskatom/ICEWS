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
from dateutil import parser

class Icews(object):

    def __init__(self, version, raw_folder, out_folder, args):
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
        self.args = args

        self.events = {}

    def read_events(self):
        """
        Read the required events in raw data files
        """
        files = glob.glob(os.path.join(self.raw_folder, "*"))
        cameoCode = config.CAMEO_CODE
        text2code = {}
        for code, subset in cameoCode.items():
            for k in subset:
                num_code = str(subset[k])
                if num_code[:2] in self.args.limit:
                    text2code[k.lower().replace(",","")] = code

        for f in files:
            with open(f) as icews_f:
                keys = icews_f.readline().strip().split("\t")[:-2]
                for line in icews_f:
                    infos = line.strip().split("\t")
                    event = {keys[i]:infos[i] for i in range(len(keys))}
                    country = event.get("Country", "")
                    eventText = event['Event Text'].lower().replace(",","")
                    eventDate = parser.parse(event['Event Date']).strftime("%Y-%m-%d")
                    if self.args.start and eventDate < self.args.start:
                        continue
                    if self.args.end and eventDate > self.args.end:
                        continue

                    eventCode = text2code.get(eventText, None)
                    if not eventCode:
                        continue
                    event["Event Sentence"] = "%s\t%s" % (eventDate, event["Event Sentence"])
                    #country event records
                    self.events.setdefault(eventCode, {})
                    self.events[eventCode].setdefault(country, [])
                    self.events[eventCode][country].append(event["Event Sentence"])


    def write_out(self):
        #write the result to specific folder
        version_folder = os.path.join(self.out_folder, self.version)
        if not os.path.exists(version_folder):
            os.mkdir(version_folder)

        for eventCode, eventSet in self.events.items():
            eventFolder = os.path.join(version_folder, eventCode)
            if not os.path.exists(eventFolder):
                os.mkdir(eventFolder)

            for location, daySet in eventSet.items():
                event_file = os.path.join(eventFolder, location.replace(" ", "_"))
                with open(event_file, "w") as outf:
                    for event in daySet:
                        outf.write("%s\n" % event)

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inFolder",
            default="/home/tskatom/workspace/icews_data", type=str)
    ap.add_argument("--outFolder",
            default="../data/icews_gsr_text",type=str)
    ap.add_argument("--version", type=str)
    ap.add_argument("--limit", type=str, nargs='+')
    ap.add_argument("--start", type=str)
    ap.add_argument("--end", type=str)
    args = ap.parse_args()
    return args


def main():
    args = parse()
    inFolder = args.inFolder
    outFolder = args.outFolder
    version = args.version

    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    icews = Icews(version, inFolder, outFolder, args)
    icews.read_events()
    icews.write_out()

if __name__ == "__main__":
    sys.exit(main())

