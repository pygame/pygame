"""
reads a svn log, and generates output for the WHATSNEW.

Should still edit the WHATSNEW to make it more human readable.


TODO: 
- group the changes on one day into a group.
- for many changes on one day list the svn as ranges [SVN 2000-2005]



"""
import sys,os,glob
lines = sys.stdin.readlines()

out= []
state = "none"
for l in lines:
    if "-------" in l:
        state = "revision date line"
        continue
    if state == "revision date line":
        parts = l.split("|")
        out.append({})
        out[-1]['revision'] = parts[0].strip().replace("r", "")
        out[-1]['username'] = parts[1].strip()
        out[-1]['date'] = parts[2].strip()
        date_parts = parts[2].strip().split("(")
        out[-1]['day_month_year'] = day, month, year = date_parts[1][:-1].split(",")[1].split()
        

        out[-1]['message'] = ""
        state = "message"
    elif state == "message":
        out[-1]['message'] += l + "\n"

import pprint
#pprint.pprint(out)


for o in out:
    day, month, year = o['day_month_year']
    print "[SVN %s] %s %s, %s" % (o['revision'], day, month, year)
    msg = o['message'][2:]

    for i in range(4):
        if msg and msg[-1] == "\n":
            msg = msg[:-1]

    print msg + "\n"



