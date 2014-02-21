import socket
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--host", action="store", type="string", dest="host", default = "localhost")
parser.add_option("-p", "--port", action="store", type="int", dest="port", default = 4774)
(options, args) = parser.parse_args()

print "UDP target IP:", options.host
print "UDP target port:", options.port
print "message:", args[0]

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.sendto(args[0], (options.host, options.port))
