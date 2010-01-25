#!/bin/sh

# Path to the debug version of python
: ${PYDEBUG:="/opt/python-debug/bin/python"}
# Valgrind checker application and the default arguments to use.
: ${VALGRIND:="valgrind"}
: ${VALGRINDARGS:="--tool=memcheck --leak-check=summary"}

usage()
{
    echo "usage: `basename $0` [-f|-r|-s] file ..."
}

while getopts frs arg; do
    case $arg in
        f)
            DEFARGS="--tool=memcheck --leak-check=full"
            ;;
        r)
            DEFARGS="--tool=memcheck --leak-check=full --show-reachable=yes"
            ;;
        s)
            DEFARGS="--tool=memcheck --leak-check=summary"
            ;;
        \? | h)
            usage
            exit 2
    esac
done

shift $(expr $OPTIND - 1)
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

for f in $@; do
    $VALGRIND $DEFARGS $PYDEBUG -E $f
done
