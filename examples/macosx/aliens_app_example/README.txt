*********************************************************************
 THESE INSTRUCTIONS ARE ONLY FOR MAC OS X 10.3, AND WILL ONLY CREATE 
 STANDALONE BUNDLES FOR MAC OS X 10.3.  THERE IS NO SUPPORT FOR 
 MAC OS X 10.2.
*********************************************************************

You will need the following packages installed to use this example:
    macholib v2.0a0 or later
        http://undefined.org/python/#macholib
        (or http://undefined.org/python/macholib-v2.0a0.tgz if not listed)

    PyProtocols v0.9.2 or later
        http://peak.telecommunity.com/PyProtocols.html
        
    Both of these should eventually be in the PackageManager repository at:
        http://undefined.org/python/pimp/

To create the bundle:
    pythonw ./buildapp.py --semi-standalone build
