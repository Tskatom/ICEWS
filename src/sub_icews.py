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

    def read_events(self):
        """
        Read the required events in raw data files
        """
        files = glob.glob(os.path.join(self.raw_folder, "*"))
        cameoCode = config.CAMEO_CODE
        text2code = {}
        for code, subset in cameoCode.items():
            for k in subset:
                text2code[k.lower().replace(",","")] = str(subset[k])[:3]

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
                    eventDate = event['Event Date']
                    eventCode = text2code.get(eventText, None)
                    if not eventCode:
                        continue

                    #country event count
                    self.daily_count.setdefault(eventCode, {})
                    self.daily_count[eventCode].setdefault(country, {})
                    count = self.daily_count[eventCode][country].setdefault(eventDate, 0)
                    self.daily_count[eventCode][country][eventDate] = count + 1

                    #city event count
                    self.daily_count[eventCode].setdefault(target_city, {})
                    count = self.daily_count[eventCode][target_city].setdefault(eventDate, 0)
                    if country == "Brazil":
                        city = event["Province"]
                        if city == target_city:
                            self.daily_count[eventCode][target_city][eventDate] = count + 1
                    elif country == "Egypt":
                        city = event["City"]
                        if city == target_city or city == "Tahrir Square":
                            self.daily_count[eventCode][target_city][eventDate] = count + 1
                    else:
                        city = event["City"]
                        if city == target_city:
                            self.daily_count[eventCode][target_city][eventDate] = count + 1
    def write_out(self):
        #write the result to specific folder
        version_folder = os.path.join(self.out_folder, self.version)
        if not os.path.exists(version_folder):
            os.mkdir(version_folder)

        for eventCode, eventSet in self.daily_count.items():
            eventFolder = os.path.join(version_folder, eventCode)
            if not os.path.exists(eventFolder):
                os.mkdir(eventFolder)

            for location, daySet in eventSet.items():
                days = sorted(daySet.keys())
                count_file = os.path.join(eventFolder, location.replace(" ", "_"))
                with open(count_file, "w") as outf:
                    for day in days:
                        outf.write("%s\t%d\n" % (day, daySet[day]))

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inFolder", type=str,
                    default="/home/tskatom/workspace/icews_data")
    ap.add_argument("--outFolder", type=str, default="../data/icews_sub_gsr")
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
    icews.write_out()

if __name__ == "__main__":
    sys.exit(main())

