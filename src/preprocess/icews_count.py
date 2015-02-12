#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ICEWS related operation
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import argparse
import glob
import json
from dateutil import parser
from util import logs

logs.init()
logger = logs.getLogger(__name__)


class Icews(object):
    def __init__(self, version, files,
                 out_folder, cameo_file="../cameo.json",
                 load=False):
        """
        Initiate the ICEWS parameters
        version: the latest version of icews data file
        raw_folder: the raw icews data file
        out_folder: the output folder of the daily count
        """
        self.version = version
        self.files = files
        self.out_folder = out_folder
        self.country_list = {"Argentina": 1, "Brazil": 1, "Chile": 1,
                             "Colombia": 1, "Ecuador": 1, "El Salvador": 1,
                             "Mexico": 1, "Paraguay": 1, "Uruguay": 1,
                             "Venezuela": 1, "Iraq": 1, "Egypt": 1,
                             "Libya": 1, "Jordan": 1, "Bahrain": 1,
                             "Syria": 1, "Saudi Arabia": 1}

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
        self.cameocode = json.load(open(cameo_file))
        self.files = files
        self.load = load
        if self.load:
            self.load_status()

    def load_status(self):
        # initiate daily count and daily_events count from files
        last_version = str(int(self.version) - 1)
        logger.info("Initiate dailycount from last status %s" % last_version)
        count_folder = os.path.join(self.out_folder, last_version)
        event_types = os.listdir(count_folder)
        for event_type in event_types:
            self.daily_count.setdefault(event_type, {})
            event_folder = os.path.join(count_folder, event_type)
            locations = os.listdir(event_folder)
            for location in locations:
                real_loc = location.replace("_", " ")
                self.daily_count[event_type].setdefault(real_loc, {})
                with open(os.path.join(event_folder, location)) as df:
                    for line in df:
                        day, count = line.strip().split('\t')
                        count = int(count)
                        self.daily_count[event_type][real_loc][day][count]
        logger.info("Initiate daily events from last status %s" % last_version)
        event_set_folder = os.path.join(self.out_folder, last_version)
        for event_type in event_types:
            self.daily_events.setdefault(event_type, {})
            event_folder = os.path.join(event_set_folder, event_type)
            locations = os.listdir(event_folder)
            for location in locations:
                real_loc = location.replace("_", " ")
                self.daily_events[event_type].setdefault(real_loc, {})
                with open(os.path.join(event_folder, location)) as df:
                    for line in df:
                        self.daily_events[
                            event_type][real_loc] = json.loads(line)
        logger.info("Finish loading from last status")

    def read_events(self):
        """
        Read the required events in raw data files
        """
        logger.info("Start to process the files.")

        for f in self.files:
            logger.info("Processing file %s." % f)
            with open(f) as icews_f:
                for line in icews_f:
                    event = json.loads(line)
                    country = event.get("Country", "")
                    if self.country_list.get(country, 0) == 0:
                        continue
                    target_city = self.capital_city.get(country, "")
                    eventDate = parser.parse(event['Event Date'])
                    eventDate = eventDate.strftime("%Y-%m-%d")
                    eventCode = event["Root Code"]

                    # country event count
                    self.daily_count.setdefault(eventCode, {})
                    self.daily_count[eventCode].setdefault(country, {})
                    self.daily_count[eventCode][country]\
                        .setdefault(eventDate, 0)
                    self.daily_count[eventCode][country][eventDate] += 1

                    self.daily_events.setdefault(eventCode, {})
                    self.daily_events[eventCode].setdefault(country, {})
                    self.daily_events[eventCode][country]\
                        .setdefault(eventDate, [])
                    self.daily_events[eventCode][country][eventDate]\
                        .append(event["embersId"])

                    # city event count
                    self.daily_count[eventCode].setdefault(target_city, {})
                    self.daily_events[eventCode].setdefault(target_city, {})
                    if country == "Brazil":
                        city = event["Province"]
                        if city == target_city:
                            self.daily_count[eventCode][target_city]\
                                .setdefault(eventDate, 0)
                            self.daily_events[eventCode][target_city]\
                                .setdefault(eventDate, [])
                            self.daily_count[
                                eventCode][target_city][eventDate] += 1
                            self.daily_events[eventCode][target_city][eventDate]\
                                .append(event["embersId"])
                    elif country == "Egypt":
                        city = event["City"]
                        if city == target_city or city == "Tahrir Square":
                            self.daily_count[eventCode][target_city]\
                                .setdefault(eventDate, 0)
                            self.daily_events[eventCode][target_city]\
                                .setdefault(eventDate, [])
                            self.daily_count[
                                eventCode][target_city][eventDate] += 1
                            self.daily_events[
                                eventCode
                            ][target_city][eventDate].append(event["embersId"])
                    else:
                        city = event["City"]
                        if city == target_city:
                            self.daily_count[eventCode][target_city]\
                                .setdefault(eventDate, 0)
                            self.daily_events[eventCode][target_city]\
                                .setdefault(eventDate, [])
                            self.daily_count[
                                eventCode][target_city][eventDate] += 1
                            self.daily_events[
                                eventCode
                            ][target_city][eventDate].append(event["embersId"])

    def write_out(self):
        # write the result to specific folder
        version_folder = os.path.join(self.out_folder, self.version)
        logger.info("Write out the result to folder %s" % version_folder)
        if not os.path.exists(version_folder):
            os.mkdir(version_folder)

        for eventCode, eventSet in self.daily_count.items():
            eventFolder = os.path.join(version_folder, eventCode)
            if not os.path.exists(eventFolder):
                os.mkdir(eventFolder)

            for location, daySet in eventSet.items():
                days = sorted(daySet.keys())
                count_file = os.path.join(eventFolder,
                                          location.replace(" ", "_"))
                with open(count_file, "w") as outf:
                    for day in days:
                        outf.write("%s\t%d\n" % (day, daySet[day]))

    def write_out_events(self):
        if self.out_folder[-1] != '/':
            event_folder = self.out_folder
        else:
            event_folder = self.out_folder[:-1]
        event_folder = event_folder + "_events"
        if not os.path.exists(event_folder):
            os.mkdir(event_folder)
        version_folder = os.path.join(event_folder, self.version)
        logger.info("Write the ICEWS GSR events to folder %s." % version_folder)
        if not os.path.exists(version_folder):
            os.mkdir(version_folder)

        for eventCode, eventSet in self.daily_events.items():
            eventFolder = os.path.join(version_folder, eventCode)
            if not os.path.exists(eventFolder):
                os.mkdir(eventFolder)

            for location, daySet in eventSet.items():
                days = sorted(daySet.keys())
                event_file = os.path.join(eventFolder,
                                          location.replace(" ", "_"))
                with open(event_file, "w") as outf:
                    for day in days:
                        outf.write(json.dumps(
                            {"day": day, "events": daySet[day]}) + "\n")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inFolder", type=str,
                    default="/home/tskatom/workspace/icews_data_labeled")
    ap.add_argument("--outFolder", type=str,
                    default="../../data/features/ICEWS/count")
    ap.add_argument("--version", type=str)
    ap.add_argument("--files", type=str, nargs='+',
                    help="the files to process")
    ap.add_argument("--load", action='store_true',
                    help='whether load the hisotrical data')
    ap.add_argument('--cameo', type=str, default='../cameo.json')
    args = ap.parse_args()
    return args


def main():
    args = parse()
    inFolder = args.inFolder
    outFolder = args.outFolder
    version = args.version

    if version is None:
        logger.error("Please enter the version number")
        sys.exit()

    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    # first consider input files
    if args.files and len(args.files) > 0:
        files = args.files
    else:
        # read all the files from folder
        files = glob.glob(os.path.join(inFolder, "*"))

    icews = Icews(version, files, outFolder,
                  cameo_file=args.cameo, load=args.load)
    try:
        icews.read_events()
        icews.write_out()
        icews.write_out_events()
    except:
        logger.error("Error araised.", exc_info=True)

if __name__ == "__main__":
    sys.exit(main())
