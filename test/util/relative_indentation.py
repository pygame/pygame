################ QUICK AND NASTY RELATIVE INDENTATION TEMPLATES ################

import re

################################################################################

def strip_common_preceding_space(selection):
    "Strips preceding common space so only relative indentation remains"

    preceding_whitespace = re.compile("^(?:(\s*?)\S)?")
    common_start = len(selection)

    split = selection.split("\n")
    for line in (l for l in split if l.strip()):
        for match in preceding_whitespace.finditer(line):
            common_start = min(match.span(1)[1], common_start)
    
    return "\n".join( [l[common_start:] for l in split] )

def pad_secondary_lines(input_str, padding):
    split = input_str.split('\n')
    return '\n'.join( [split[0]] + [(padding+l) for l in split[1:]] )

################################################################################

ph_re = re.compile("\${(.*?)}")
multi_line_re = re.MULTILINE | re.DOTALL

################################################################################

class Template(object):
    def __init__(self, template, strip_common = True, strip_excess = True):
        
        if strip_common: template = strip_common_preceding_space(template)
        if strip_excess: template = template.strip() + '\n'
        
        self.template = template
        
        self.find_ph_offsets()

    def find_ph_offsets(self):
        self.ph_offsets = dict()
    
        for i, lines in enumerate(self.template.split('\n')):
            for match in ph_re.finditer(lines):
                self.ph_offsets[match.group(1)] = match.span()[0]
    
    def render(self, replacements = None, **kw):
        if not replacements: replacements = kw
        
        if not len(self.ph_offsets) == len(replacements):
            raise ValueError( 
                'length of replacements not length of placeholders'
            )

        template = self.template[:]
        for ph_name, replacement in replacements.iteritems():
            ph_offset = self.ph_offsets[ph_name]
    
            ph_search = re.search("\${%s}" % ph_name, template, multi_line_re)
            ph_start, ph_end = ph_search.span()
            
            padded = pad_secondary_lines(replacement, ph_offset * ' ')
    
            template = template[0:ph_start] + padded + template[ph_end:]
    
        return template

if __name__ == "__main__":
    
    template = Template( '''
        def test_${test_name}(self):
            
            """
                ${docstring}
            """
                    ${comments}
            
            self.assert_(not_completed() ''' 
    )
    
    print template.render( test_name = 'this_please', 
                           docstring = 'Comments\nAre\nVital',
                           comments  = '# some comment\n# some other comment', )

################################################################################