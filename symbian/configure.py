""" Script to configure build_config.py 
  
== Build everything, including pys60 ==
configure.py pys60_ce_src=\projects\pys60ce\trunk\src

== Use existing python sis ==
configure.py build_python=False pys60_sis=python.sisx pythondll=python222.dll

"""

import os, sys

def start():
    
    try:
        import build_config as defaults
    except:
        import default_build_config as defaults
    
    args = [x.split("=") for x in sys.argv if "=" in x]
        
    vars = [ x for x in dir(defaults) if not x.startswith("_") ]
    values = [ getattr( defaults, x ) for x in vars ]
    oldvalues = zip( vars, values )
    
    print "=" * 79
    result = {}
    for name,value in args:
        if name not in vars:
            print "Error: no such configuration '%s'" % name
            print "Possible configuration values are:\n", " | ".join( vars )
            raise SystemExit( )
        
        old = getattr(defaults, name )
        try:
            # Evaluate booleans and integers
            result[name] = eval(value)
        except:
            result[name] = value
            
        print name, "reconfigured '%s' => '%s'" % ( str(old), str( value ))
    
    for name, value in oldvalues:
        if name not in result:
            result[name] = value
    
    
    # Create the module
    print
    print "-" * 79
    f=open("build_config.py",'w');
    keys = result.keys();keys.sort()
    for name in keys:
        value = result[name]
        line = "%-15s = %s\n" % ( name, repr(value))
        print line.strip()
        f.write(line)
    f.close()
    
    print "=" * 79
    
if __name__ == "__main__":
    start()    
