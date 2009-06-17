"""
reads a svn log, and generates output for the WHATSNEW.

Should still edit the WHATSNEW to make it more human readable.



"""
import sys,os,glob,textwrap
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


# group revisions on the same day into one block.
previous = []
for o in (out + [None]):

    if o and previous and o['day_month_year'] == previous[-1]['day_month_year']:
        previous.append(o)
        continue
    else:

        if not previous:
            previous.append(o)
            continue

        day, month, year = previous[-1]['day_month_year']
        revs = [int(p['revision']) for p in previous]
        if len(revs) == 1:
            revisions = revs[0]
        else:
            revisions = "%s-%s" % (min(revs), max(revs))

        print "[SVN %s] %s %s, %s" % (revisions, month, day, year)

        # uniqify the messages, keep order.
        messages = [p['message'][2:] for p in previous]
        unique_messages = []
        for m in messages:
            if m not in unique_messages:
                unique_messages.append(m)
            


        for msg in reversed(unique_messages):
            #msg = p['message'][2:]

            for i in range(4):
                if msg and msg[-1] == "\n":
                    msg = msg[:-1]

            lines = textwrap.wrap(msg, 74)
            if lines:
                if lines[-1][-1:] != ".":
                    lines[-1] += "."

            for i, l in enumerate(lines):
                if i == 0:
                    print "    %s" % (l[:1].upper() + l[1:])
                if i != 0:
                    print "      %s" % l

        print ""

        previous  = [o]




