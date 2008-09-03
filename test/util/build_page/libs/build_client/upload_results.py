import multi_part
import urllib2
import base64

def post_build(results_zip, url=None):
    print 'POSTING BUILD, PLEASE WAIT'

    auth = tuple(open('./config/auth.txt').read().split(':'))

    theurl = url or 'http://pygame-testify.net/upload_results/index.py'

    req = urllib2.Request(theurl)

    base64string = base64.encodestring('%s:%s' % auth)[:-1]
    authheader =  "Basic %s" % base64string

    req.add_header("Authorization", authheader)

    opener = urllib2.build_opener(multi_part.MultipartPostHandler)
    params = { "results_file" : open(results_zip, "rb") }

    return opener.open(req, params).read()

if __name__ == '__main__':    
    print post_build ('output/build.zip')