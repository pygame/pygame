################################################################################
# Imports

# StdLib
import sys

# User Libs
import regexes 
import config
import update

#################################################################################

svn_blame = lambda x: (0, '\n'.join([regexes.SVN_BLAME_TEST.strip()] * 500))

get_and_brand_latest_svn = lambda x: 1000000

#################################################################################

if __name__ == '__main__':
    # Swap out some slow components for quick debugging
    update.svn_blame = svn_blame
    config.get_and_brand_latest_svn = get_and_brand_latest_svn
    
    if 'config' in sys.argv:     config.main()
    else:                        update.main()