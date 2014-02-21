import socket

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--host", action="store", type="string", dest="host", default = "localhost")
parser.add_option("-p", "--port", action="store", type="int", dest="port", default = 4774)
(options, args) = parser.parse_args()

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((options.host, options.port))

while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    print "Received message:", data
