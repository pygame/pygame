# coding: utf-8

################################################################################

import re
import htmlentitydefs

_escape_re = re.compile(eval(r'u"[&<>\"]|[\u0080-\uffff]+"'))

################################################################################

def _escape_sub(match):
    try:
        entity_code = ord(match.group(0))
    
    except Exception, e:
        print match.group(0)
        raise

    named_entitiy = htmlentitydefs.codepoint2name.get(entity_code)
    if named_entitiy: return '&%s;' % named_entitiy
    else: return '&#%d;' % entity_code

################################################################################

def escape(uni, codec=None):
    if codec:  uni = uni.decode(codec)
    return _escape_re.sub(_escape_sub, uni)

ehtml = escape

__all__ = ['escape', 'ehtml']

################################################################################

if __name__ == '__main__':
    print ehtml(
        '“Gross national happiness is more important”', 'utf-8'
    )
    
    print ehtml('&&', 'utf-8')
    
################################################################################