# Program getcab.py

"""Extract all cabinet embedded in an .msi file.

Usage:  python getcabs.py <.msi file>

Cab files are written to the current directory and are named after their
entries in the Media table.

This is a demonstation of the msidb module.

"""

import sys

import msidb

def main(msi_file_path):
    print ("Processing %s" % msi_file_path)
    con = msidb.connect(msi_file_path)
    try:
        cur = con.cursor()
        cur.execute("select Cabinet from Media")
        for cabinet, in cur:
            if cabinet.startswith('#'):
                name = cabinet[1:]
                cur.execute("select Data from _Streams where Name=?", (name,))
                cab_file_name = "%s.cab" % name
                data, = cur.fetchone()
                print ("Writing %s..." % cab_file_name)
                cab_file = open(cab_file_name, 'wb')
                try:
                    cab_file.write(data)
                finally:
                    cab_file.close()
    finally:
        con.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("usage: python getcabs.py <.msi file>")
    else:
        main(sys.argv[1])
