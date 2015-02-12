# -*- coding: utf-8 -*-
import sys
from dateutil import parser

filename=sys.argv[1]
start = filename.split('_')[-2]
end = filename.split('_')[-1][:3]

lines = [l.strip().split('\t') for l in open(filename) if l.strip().split('\t')[1] != 'Event Date']
for line_obj in lines:
    line_obj[1] = parser.parse(line_obj[1]).strftime("%Y-%m-%d")
# <codecell>
lines.insert(0, "Fake Head Line")
print lines[0]
print lines[1]

# <codecell>

categories_from_data = set([l[5] for l in lines[1:]])
categories_from_data = [c.lower().replace(",",'') for c in categories_from_data]
print "total categories from ICEWS data", len(categories_from_data)
#sorted(categories_from_data)

# <codecell>

icews_cameo_categories = {
14: "PROTEST", 
140: "Engage in political dissent",
1401: "Engage in civilian protest", 
141: "Demonstrate or rally",
1411: "Demonstrate or rally for leadership change",
1412: "Demonstrate or rally for policy change",
1413: "Demonstrate or rally for rights",
1414: "Demonstrate for change in institutions, regime",
1411: "Demonstrate for leadership change",
1412: "Demonstrate for policy change",
1413: "Demonstrate for rights",
142: "Conduct hunger strike",
1421: "Conduct hunger strike for leadership change",
1422: "Conduct hunger strike for policy change",
1423: "Conduct hunger strike for rights",
1424: "Conduct hunger strike change in institutions, regime",
143: "Conduct strike or boycott",
1431: "Strike or boycott for leadership change",
1432: "Strike or boycott for policy change",
1433: "Strike or boycott for rights",
1434: "Strike or boycott for change in institutions, regime",
1431: "Conduct strike or boycott for leadership change",
1432: "Conduct strike or boycott for policy change",
1433: "Conduct strike or boycott for rights",
1434: "Conduct strike or boycott for change in institutions, regime",
144: "Obstruct passage or block", 
1441: "Obstruct passage for leadership change",
1442: "Obstruct passage for policy change",
1443: "Obstruct passage for rights",
1444: "Obstruct passage for change in institutions, regime",
1441: "Obstruct passage to demand leadership change",
1442: "Obstruct passage to demand policy change",
1443: "Obstruct passage to demand rights",
1444: "Obstruct passage to demand change in institutions, regime",
145: "Engage in violent protest, riot",
145: "Protest violently, riot",
1451: "Violently protest for leadership change",
1452: "Violently protest for policy change",
1453: "Violently protest for rights",
1454: "Violently protest for change in institutions, regime",
1451: "Engage in violent protest for leadership change",
1452: "Engage in violent protest for policy change",
1453: "Engage in violent protest for rights",
1454: "Engage in violent protest for change in institutions, regime",
"01": "MAKE PUBLIC STATEMENT",
"010": "Make statement",
"011": "Decline comment",
"012": "Make pessimistic comment",
"013": "Make optimistic comment",
"014": "Consider policy option",
"015": "Acknowledge or claim responsibility",
"016": "Reject accusation, deny responsibility", 
"017": "Engage in symbolic act",
"018": "Make empathetic comment",
"019": "Express accord",
"020": "Appeal", 
"021": "Appeal for material cooperation",
"0211": "Appeal for economic cooperation", 
"0212": "Appeal for military cooperation",
"0213": "Appeal for judicial cooperation", 
"0214": "Appeal for intelligence cooperation", 
"0215": "Appeal for military cooperation",
"022": "Appeal for diplomatic cooperation or policy support", 
"023": "Appeal for material aid",
"0231": "Appeal for economic aid",
"0232": "Appeal for military aid",
"0233": "Appeal for humanitarian aid",
"0234": "Appeal for military protection or peacekeeping",
"024": "Appeal for political reform",
"0241": "Appeal for leadership change",
"0242": "Appeal for policy change",
"0243": "Appeal for rights",
"0244": "Appeal for change in institutions, regime", 
"025": "Appeal to yield",
"0251": "Appeal for easing of administrative sanctions", 
"0252": "Appeal for easing of political dissent",
"0253": "Appeal for release of persons or property",
"0254": "Appeal for easing of economic sanctions, boycott", 
"0255": "Appeal for international involvement (not mediat.)", 
"0256": "Appeal for target to deescalate military engage",
"026": "Appeal to others to meet or negotiate",
"027": "Appeal to others to settle dispute",
"028": "Appeal to others to engage in or accept mediation",
"030": "Express intent to cooperate",
"031": "Express intent to engage in material cooperation",
"0311": "Express intent to cooperate economically",
"0312": "Express intent to cooperate militarily",
"0313": "Express intent to cooperate judicially",
"0314": "Express intent to cooperate on intelligence",
"032": "Express intent to engage in diplomatic cooperation such as policy support",
"033": "Express intent to provide material aid",
"0331": "Express intent to provide economic aid",
"0332": "Express intent to provide military aid",
"0333": "Express intent to provide humanitarian aid",
"0334": "Express intent to provide military protection or peacekeeping",
"034": "Express intent to institute political reform",
"0341": "Express intent to change leadership",
"0342": "Express intent to change policy",
"0343": "Express intent to provide rights",
"0344": "Express intent to change institutions, regime",
"035": "Express intent to yield",
"0351": "Express intent to ease administrative sanctions",
"0352": "Express intent to ease popular dissent",
"0353": "Express intent to release of persons or property",
"0354": "Express intent to ease economic sanctions, boycott",
"0355": "Express intent to allow international involvement (not mediat.)",
"0356": "Express intent to deescalate military engagement",
"036": "Express intent to meet or negotiate",
"037": "Express intent to settle dispute",
"038": "Express intent to accept mediation",
"039": "Express intent to mediate",
"040": "Consult",
"041": "Discuss by telephone",
"042": "Make a visit",
"043": "Host a visit",
"044": "Meet at a 'third' location",
"045": "Engage in mediation",
"046": "Engage in negotiation",
"050": "Engage in diplomatic cooperation",
"051": "Praise or endorse",
"052": "Defend verbally",
"053": "Rally support on behalf of",
"054": "Grant diplomatic recognition",
"055": "Apologize",
"056": "Forgive",
"057": "Sign formal agreement",
"060": "Engage in material cooperation",
"061": "Cooperate economically",
"062": "Cooperate militarily",
"063": "Engage in judicial cooperation",
"064": "Share intelligence or information",
"070": "Provide aid",
"071": "Provide economic aid",
"072": "Provide military aid",
"073": "Provide humanitarian aid",
"074": "Provide military protection or peacekeeping",
"075": "Grant asylum",
"080": "Yield",
"081": "Ease administrative sanctions",
"0811": "Ease restrictions on political freedoms",
"0812": "Ease ban on parties or politicians",
"0813": "Ease curfew",
"0814": "Ease state of emergency or curfew",
"082": "Ease political dissent",
"083": "Accede to requests or demands for political reform",
"0831": "Accede to demands to change leadership",
"0832": "Accede to demands to change policy",
"0833": "Accede to demands to provide rights",
"0834": "Accede to demands to change institutions, regime",
"084": "Return, release",
"0841": "Return, release persons",
"0842": "Return, release property",
"085": "Ease economic sanctions, boycott or embargo",
"086": "Allow international involvement",
"0861": "Receive deployment of peacekeepers",
"0862": "Receive inspectors",
"0863": "Allow for humanitarian access",
"087": "De-escalate military engagement",
"0871": "Declare truce, ceasefire",
"0872": "Ease military blockade",
"0873": "Demobilize armed forces",
"0874": "Retreat or surrender militarily",
"090": "Investigate",
"091": "Investigate crime, corruption",
"092": "Investigate human rights abuses",
"093": "Investigate military action",
"094": "Investigate war crimes",
"100": "Demand",
"101": "Demand material cooperation",
"1011": "Demand economic cooperation",
"1012": "Demand military cooperation",
"1013": "Demand judicial cooperation",
"1014": "Demand intelligence cooperation",
"1015": "Demand military cooperation",
"102": "Demand diplomatic cooperation such as policy support",
"103": "Demand material aid",
"1031": "Demand economic aid",
"1032": "Demand military aid",
"1033": "Demand humanitarian aid",
"1034": "Demand military protection or peacekeeping",
"104": "Demand political reform",
"1041": "Demand leadership change",
"1042": "Demand policy change",
"1043": "Demand rights",
"1044": "Demand change in institutions, regime",
"105": "Demand target yield",
"1051": "Demand easing of administrative sanctions",
"1052": "Demand easing of political dissent",
"1053": "Demand release of persons or property",
"1054": "Demand easing of economic sanctions, boycott",
"1055": "Demand international involvement (not mediat.)",
"1056": "Demand de-escalation of military engage",
"106": "Demand meeting, negotiation",
"107": "Demand settling of dispute",
"108": "Demand meditation",
"110": "Disapprove",
"111": "Criticize or denounce",
"112": "Accuse",
"1121": "Accuse of crime, corruption",
"1122": "Accuse of human rights abuses",
"1123": "Accuse of aggression",
"1124": "Accuse of war crimes",
"1125": "Accuse of espionage, treason",
"113": "Rally opposition against",
"114": "Complain officially",
"115": "Bring lawsuit against",
"116": "Find legally guilty or liable",
"120": "Reject",
"121": "Reject material cooperation",
"1211": "Reject economic cooperation",
"1212": "Reject military cooperation",
"1213": "Reject judicial cooperation",
"1214": "Reject intelligence cooperation",
"1215": "Reject military cooperation",
"122": "Reject request for material aid",
"1221": "Reject request for economic aid",
"1222": "Reject request for military aid",
"1223": "Reject request for humanitarian aid",
"1224": "Reject request for military protection, peacekeeping",
"123": "Reject demand for political reform",
"1231": "Reject request for leadership change",
"1232": "Reject request for policy change",
"1233": "Reject request for rights",
"1234": "Reject request for change in institutions, regime",
"124": "Refuse to yield",
"1241": "Refuse to ease administrative sanctions",
"1242": "Refuse ease popular dissent",
"1243": "Refuse to release of persons or property",
"1244": "Refuse to ease economic sanctions, boycott",
"1245": "Refuse to allow international involvement (not mediation)",
"1246": "Refuse to de-escalate military engagement",
"125": "Reject proposal to meet, discuss, negotiate",
"126": "Reject mediation",
"127": "Reject plan, agreement to settle dispute",
"128": "Defy norms, law",
"129": "Veto",
"130": "Threaten",
"131": "Threaten non-force",
"1311": "Threaten to reduce or stop aid",
"1312": "Threaten to boycott, embargo, or sanction",
"1313": "Threaten to reduce or break relations",
"132": "Threaten with administrative sanctions",
"1321": "Threaten with restrictions on political freedoms",
"1322": "Threaten to ban political parties or politicians",
"1323": "Threaten to impose curfew",
"1324": "Threaten to impose state of emergency or martial law",
"133": "Threaten political dissent",
"134": "Threaten to halt negotiations",
"135": "Threaten to halt mediation",
"136": "Threaten to halt international involvement (not medit.)",
"137": "Threaten with repression",
"138": "Threaten with military force",
"1380": "Threaten force",
"1381": "Threaten blockade",
"1382": "Threaten occupation",
"1383": "Threaten unconventional violence",
"1384": "Threaten conventional attack",
"1385": "Threaten attack with WMD",
"137": "Give ultimatum",
"15": "EXHIBIT MILITARY POSTURE",
"150": "Exhibit military or police power",
"151": "Increase police alert status",
"152": "Increase military alert status",
"153": "Mobilize or increase police power",
"154": "Mobilize or increase armed forces",
"155": "Mobilize or increase cyber-forces",
"160": "Reduce relations",
"161": "Reduce or break diplomatic relations",
"162": "Reduce or stop material aid",
"1621": "Reduce or stop economic assistance",
"1622": "Reduce or stop military assistance",
"1623": "Reduce or stop humanitarian assistance",
"163": "Impose embargo, boycott or sanctions",
"164": "Halt negotiations",
"165": "Halt mediation",
"166": "Expel or withdraw",
"1661": "Expel or withdraw peacekeepers",
"1662": "Expel or withdraw inspectors, observers",
"1663": "Expel or withdraw aid agencies",
"170": "Coerce",
"171": "Seize or damage property",
"1711": "Confiscate property",
"1712": "Destroy property",
"172": "Impose administrative sanctions",
"1721": "Impose restrictions on political freedoms",
"1722": "Ban political parties or politicians",
"1723": "Impose curfew",
"1724": "Impose state of emergency or martial law",
"173": "Arrest, detain",
"174": "Expel or deport individuals",
"176": "Attack cybernetically",
"175": "Use tactics of violent repression",
"18": "ASSAULT",
"180": "Use unconventional violence",
"181": "Abduct, hijack, take hostage",
"182": "Physically assault",
"1821": "Sexually assault",
"1822": "Torture",
"1823": "Kill by physical assault",
"183": "Conduct suicide, car, or other non-military bombing",
"1831": "Carry out suicide bombing",
"1832": "Carry out vehicular bombing",
"1833": "Carry out roadside bombing (IED)",
"184": "Use as human shield",
"185": "Attempt to assassinate",
"186": "Assassinate",
"19": "FIGHT",
"190": "Use conventional military force",
"191": "Impose blockade, restrict movement",
"192": "Occupy territory",
"193": "Fight with small arms and light weapons",
"194": "Fight with artillery and tanks",
"195": "Employ aerial weapons",
"1951": "Employ precision-guided aerial munitions",
"1952": "Employ remotely piloted aerial munitions",
"196": "Violate ceasefire",
"200": "Engage in unconventional mass violence",
"201": "Engage in mass expulsion",
"202": "Engage in mass killings",
"203": "Engage in ethnic cleansing",
"204": "Use weapons of mass destruction",
"2041": "Use chemical, biological, or radiological weapons",
"2042": "Detonate nuclear weapons"

}
icews_event_codes = [str(k) for k in icews_cameo_categories.keys()]
icews_cameo_categories = {v.lower().strip().replace(',',''): str(k) for k, v in icews_cameo_categories.items()}
print "total CAMEO categories", len(icews_cameo_categories)

# <codecell>

print icews_event_codes

# <codecell>

mapped_categories = {
"protest":"14",
"engage in political dissent":140, 
"engage in civilian protest":1401,
"demonstrate or rally":141,
"present demonstrate or rally":141,
"demonstrate for leadership change":1411,
"demonstrate or rally for leadership change":1411,
"demonstrate for policy change":1412,
"demonstrate or rally for policy change":1412,
"demonstrate for rights":1413,
"demonstrate or rally for rights":1413,
"demonstrate for change in institutions regime":1414,
"conduct hunger strike":142,
"present conduct hunger strike":142,
"conduct hunger strike for leadership change":1421,
"conduct hunger strike for policy change":1422,
"conduct hunger strike for rights":1423,
"hunger strike change in institutions regime":1424,
"conduct hunger strike change in institutions regime":1424,
"conduct hunger strike for change in institutions regime":1424,
"conduct strike or boycott":143,
"present conduct strike or boycott":143,
"conduct strike or boycott for leadership change":1431, 
"strike or boycott for leadership change":1431,
"conduct strike or boycott for policy change":1432,
"strike or boycott for policy change":1432,
"conduct strike or boycott for policy change":1432,
"strike or boycott for rights":1433,
"conduct strike or boycott for rights":1433,
"strike or boycott for change in institutions regime":1434,
"conduct strike or boycott for change in institutions regime":1434,
"obstruct passage block":144,
"obstruct passage or block":144,
"obstruct passage to demand leadership change":1441,
"obstruct passage for leadership change":1441,
"obstruct passage to demand policy change":1442,
"obstruct passage for policy change":1442,
"obstruct passage to demand rights":1443,
"obstruct passage for rights":1443,
"obstruct passage to demand change in institutions, regime":1444,
"obstruct passage to demand change in institutions regime" :1444,
"obstruct passage for change in institutions regime":1444,
"protest violently riot":145,
"engage in violent protest riot":145,
"engage in violent protest for leadership change":1451,
"violently protest for leadership change":1451,
"engage in violent protest for policy change":1452,
"engage in violent protest for rights":1453,
"violently protest for rights":1453,
"violently protest for policy change":1452,
"engage in violent protest for change in institutions regime":1454,
"violently protest for change in institutions regime":1454,
"appeal for intelligence cooperation":"0214",
"appeal for intelligence":"0214",
"appeal for intelligence cooperation":"0214",
"appeal for intelligence":"0214",
"reject accusation deny responsibility":"016",
"deny responsibility":"016",
"appeal for international involvement (not mediat.)":"0255",
"appeal for target to allow international involvement (non-mediation)":"0255",
"appeal to others to engage in or accept mediation":"028",
"appeal to engage in or accept mediation":"028",
"appeal for diplomatic cooperation (such as policy support)":"022",
"appeal for diplomatic cooperation or policy support":"022",
"appeal for change in leadership":"0241",
"appeal for leadership change":"0241",
"appeal for de-escalation of military engagement":"0256",
"appeal for target to deescalate military engage":"0256",
"appeal for easing of economic sanctions boycott or embargo":"0254",
"appeal for easing of economic sanctions boycott":"0254",
"appeal for aid":"023",
"appeal for material aid":"023",
"appeal for easing of political dissent":"0252",
"appeal": "020",
"make an appeal or request": "020",
"make public statement":"01",
"express intent to engage in diplomatic cooperation (such as policy support)": "032",
"express intent to engage in diplomatic cooperation such as policy support": "032",
"express intent to de-escalate military engagement": "0356",
"express intent to deescalate military engagement": "0356",
"express intent to cooperate on judicial matters": "0313",
"express intent to cooperate judicially": "0313",
"express intent to release persons or property": "0353",
"express intent to release of persons or property": "0353",
"express intent to allow international involvement (non-mediation)": "0355",
"express intent to allow international involvement (not mediat.)": "0355",
"express intent to change policy": "0342",
"express intent to provide rights":"0343",
"express intent to ease economic sanctions boycott": "0354",
"express intent to ease popular dissent": "0352",
"express intent to ease economic sanctions boycott or embargo": "0354",
"engage in mediation": "045",
"mediate": "045",
"accede to demands for change in policy":"0832",
"accede to demands to change policy":"0832",
"accede to demands for change in leadership":"0831",
"accede to demands to change leadership":"0831",
"allow humanitarian access":"0863",
"allow for humanitarian access":"0863",
"accede to demands for rights":"0833",
"accede to demands to provide rights":"0833",
"return release person(s)":"0841",
"return release persons":"0841",
"accede to demands for change in institutions regime":"0834",
"accede to demands to change institutions regime":"0834",
"ease ban on political parties or politicians":"0812",
"ease ban on parties or politicians":"0812",
"ease curfew":"0814",
"ease state of emergency or curfew":"0814",
"return release":"084",
"ease economic sanctions boycott embargo":"085",
"ease economic sanctions boycott or embargo":"085",
"allow international involvement":"086",
"de-escalate military engagement":"087",
"reject request for material aid":"122",
"reject request or demand for material aid":"122",
"refuse to release of persons or property":"1243",
"refuse to release persons or property":"1243",
"refuse to allow international involvement (not mediation)":"1245",
"refuse to allow international involvement (non mediation)":"1245",
"refuse to ease economic sanctions boycott":"1244",
"refuse to ease economic sanctions boycott or embargo":"1244",
"reject request for military protection peacekeeping":"1224",
"reject request for military protection or peacekeeping":"1224",
"reject proposal to meet discuss negotiate":"125",
"reject proposal to meet discuss or negotiate":"125",
"reject request for leadership change":"1231",
"reject request for change in leadership":"1231",
"reject demand for political reform":"123",
"reject request or demand for political reform":"123",
"find legally guilty or liable":"116",
"find guilty or liable (legally)":"116",
"demand international involvement (not mediat.)":"1055",
"demand that target allows international involvement (non-mediation)":"1055",
"demand de-escalation of military engage":"1056",
"demand de-escalation of military engagement":"1056",
"demand easing of economic sanctions boycott":"1054",
"demand easing of economic sanctions boycott or embargo":"1054",
"demand diplomatic cooperation such as policy support":"102",
"demand diplomatic cooperation (such as policy support)":"102",
"demand leadership change":"1041",
"demand change in leadership":"1041",
"demand that target yields":"105",
"demand target yield": "105",
"demand mediation": "108",
"demand meditation": "108",
"demand material aid": "103", 
"demand military aid": "1032",
"demand economic cooperation": "1011",
"demand easing of political dissent": "1052",
"disapprove": "110",
"reject intelligence cooperation": "1214",
"refuse ease popular dissent": "1242",
"refuse to ease popular dissent": "1242",
"threaten with military force": "138",
"threaten with repression": "137",
"ease state of emergency or martial law": "0814",
"carry out vehicular bombing": "1832",
"carry out car bombing" : "1832",
"abduct hijack take hostage": "181",
"abduct hijack or take hostage" : "181",
"arrest detain": "173",
"arrest detain or charge with legal action" : "173",
"carry out roadside bombing (ied)" :"1833",
"carry out roadside bombing": "1833",
"exhibit military or police power": "150",
"demonstrate military or police power" : "150",
"threaten to boycott embargo or sanction": "1312",
"threaten with sanctions boycott embargo": "1312",
"threaten political dissent" : "133",
"threaten with political dissent protest": "133",
"threaten to impose state of emergency or martial law": "1324",
"threat to impose state of emergency or martial law": "1324",
"threaten to halt international involvement (non-mediation)": "136",
"threaten to halt international involvement (not medit.)": "136",
"exhibit military posture": "15",
"threaten conventional attack": "1384",
"threaten blockade": "1381",
"employ precision-guided aerial munitions": "1951",
"threaten attack with wmd": "1385",
"use tactics of violent repression":"175",
"attack cybernetically": "176",
"threaten force": "1380",
"threaten with military force":"138",
"fight": "19",
"expel or withdraw inspectors observers": "1662",
"engage in unconventional mass violence": "200",
"expel or withdraw aid agencies": "1663",
"mobilize or increase cyber-forces": "155",
"threaten unconventional violence": "1383",
"threaten occupation": "1382",
"assault": "18",
"employ remotely piloted aerial munitions": "1952"
}
mapped_categories = {k:str(v) for k, v in mapped_categories.items()}
print "mapped categories", len(mapped_categories)
#print "updating the newly mapped categories in the CAMEO categories"
#icews_cameo_categories.update({k:str(v) for k, v in mapped_categories.items()})
#print "updated categories", len(icews_cameo_categories)

# <codecell>

for k, v in icews_cameo_categories.items():
    if k in mapped_categories:
        continue
    if k in set(categories_from_data):
        pass #print "present", k, "[",v, "]"
    else:
        print "not present ==>", k, "[",v, "]"

# <codecell>

for k in sorted(categories_from_data):
    if k.lower().replace(',','') in set([ v.lower().replace(",","") for v in mapped_categories]) or\
       k.lower().replace(',','') in set([ u.lower().replace(",","") for u in icews_cameo_categories]):
        continue
    else:
        print k

# <codecell>

all_categories = {}
all_categories.update(icews_cameo_categories)
print "updating with cameo categories", len(all_categories)
all_categories.update(mapped_categories)
print "updating with mapped categories", len(all_categories)

# <codecell>

def group_by(data, key=None, value=None):
    grps = {}
    for d in data:
        try:
            k = key(d)
        except KeyError as ke:
            raise Exception("KeyError %s\nValue: %s" % (ke,d))
        if k is not None:
            if k not in grps:
                grps[k] = []
            if value is None:
                grps[k].append(d)
            else:
                grps[k].append(value(d))
    return grps

# <codecell>

# group data by country
data_co = group_by(lines[1:], key=lambda x:x[-3])

# <codecell>

from datetime import datetime
from time import strptime
dates = sorted([l[1] for l in lines[1:]], key=lambda x: datetime(*strptime(x, "%Y-%m-%d")[0:6]))
print dates[0], dates[-1]

# <codecell>

#osi_countries = set(["argentina", "brazil", "chile", "colombia", "ecuador", "el salvador","mexico","paraguay","uruguay","venezuela","turkey"])
osi_countries = set(["argentina", "bahrain", "brazil", "chile", "colombia", 
                     "ecuador", "el salvador", "egypt", "iraq", "jordan",
                     "libya", "mexico", "paraguay", "saudi arabia","syria", 
                     "uruguay", "venezuela"])

# <codecell>

data_co_osi = {k:v for k, v in data_co.items() if k.lower() in osi_countries}

# <codecell>

data_co_osi.keys()

# <codecell>

# group data by day
for k in data_co_osi.keys():
    data_co_osi[k] = group_by(data_co_osi[k], key=lambda x:x[1])

# <codecell>

# group data by all sub-events
for co in data_co_osi:
    for d in data_co_osi[co]:
        data_co_osi[co][d] = group_by([i for i in data_co_osi[co][d] if i[5].lower().replace(',', '') in all_categories], 
                                       key=lambda x:all_categories[x[5].lower().replace(',', '')])

# <codecell>

cols = ['country', 'date'] + sorted(set([v for v in all_categories.values()]))
category_index = {c:i+2 for i, c in enumerate(sorted(set([v for v in all_categories.values()]))) }

# <codecell>

print cols
print category_index

# <codecell>

# write country wise counts to file
import math
for co in data_co_osi:
    with open("%s_icews_sub_event_counts_%s_%s.csv" % ("_".join(co.lower().split()),start,end), 'w') as out:
        out.write("\t".join([str(c) for c in cols]) + '\n')
        for d in data_co_osi[co]:
            if len(data_co_osi[co][d]) > 0:
                row = [0] * len(cols)
                row[0] = co
                row[1] = d
                for c in data_co_osi[co][d]:
                    row[category_index[c]] = len(data_co_osi[co][d][c])
                out.write("\t".join([str(r) for r in row]) + '\n')
        for d in [dates[0], dates[-1]]:
            if d not in data_co_osi[co]:
                row = [0] * len(cols)
                row[0] = co
                row[1] = d
                out.write("\t".join([str(r) for r in row]) + '\n')

for co in data_co_osi:
    with open("%s_icews_sub_event_intensities_%s_%s.csv" %("_".join(co.lower().split()),start,end), 'w') as out:
        out.write("\t".join([str(c) for c in cols]) + '\n')
        for d in data_co_osi[co]:
            if len(data_co_osi[co][d]) > 0:
                row = [0] * len(cols)
                row[0] = co
                row[1] = d
                for c in data_co_osi[co][d]:
                    row[category_index[c]] = math.fsum([float(i[6]) for i in data_co_osi[co][d][c]])
                out.write("\t".join([str(r) for r in row]) + '\n')
        for d in [dates[0], dates[-1]]:
            if d not in data_co_osi[co]:
                row = [0] * len(cols)
                row[0] = co
                row[1] = d
                out.write("\t".join([str(r) for r in row]) + '\n') 

# <codecell>


# <codecell>

# sum intensities
data_co_intensities = {k:v for k, v in data_co.items() if k.lower() in osi_countries}

# <codecell>

# group data by day
for k in data_co_intensities.keys():
    data_co_intensities[k] = group_by(data_co_intensities[k], key=lambda x:x[1])

# <codecell>

# group data by all parent-events 'XY'
grp_by_parent_event = lambda x: all_categories[x[5].lower().replace(',', '')][:2]
for co in data_co_intensities:
    for d in data_co_intensities[co]:
        grpd_co_d = [i for i in data_co_intensities[co][d] if i[5].lower().replace(',', '') in all_categories]
        data_co_intensities[co][d] = group_by(grpd_co_d, key=grp_by_parent_event)

# <codecell>

import math
intensity_cols = ['country', 'date'] + sorted(set([v[:2] for v in all_categories.values()]))
intensity_category_index = {c:i+2 for i, c in enumerate(sorted(set([v[:2] for v in all_categories.values()]))) }
print intensity_cols
print intensity_category_index
# write country wise counts to file
for co in data_co_intensities:
    with open("%s_icews_parent_event_counts_%s_%s.csv" % ("_".join(co.lower().split()),start,end), 'w') as out:
        out.write("\t".join([str(c) for c in intensity_cols]) + '\n')
        for d in data_co_intensities[co]:
            if len(data_co_intensities[co][d]) > 0:
                row = [0] * len(intensity_cols)
                row[0] = co
                row[1] = d
                for c in data_co_intensities[co][d]:
                    row[intensity_category_index[c]] = len(data_co_intensities[co][d][c])
                out.write("\t".join([str(r) for r in row]) + '\n')
        for d in [dates[0], dates[-1]]:
            if d not in data_co_intensities[co]:
                row = [0] * len(intensity_cols)
                row[0] = co
                row[1] = d
                out.write("\t".join([str(r) for r in row]) + '\n') 
                
for co in data_co_intensities:
    with open("%s_icews_parent_event_intensities_%s_%s.csv" %("_".join(co.lower().split()),start,end), 'w') as out:
        out.write("\t".join([str(c) for c in intensity_cols]) + '\n')
        for d in data_co_intensities[co]:
            if len(data_co_intensities[co][d]) > 0:
                row = [0] * len(intensity_cols)
                row[0] = co
                row[1] = d
                for c in data_co_intensities[co][d]:
                    row[intensity_category_index[c]] = math.fsum([float(i[6]) for i in data_co_intensities[co][d][c]])
                out.write("\t".join([str(r) for r in row]) + '\n')
        for d in [dates[0], dates[-1]]:
            if d not in data_co_intensities[co]:
                row = [0] * len(intensity_cols)
                row[0] = co
                row[1] = d
                out.write("\t".join([str(r) for r in row]) + '\n') 


