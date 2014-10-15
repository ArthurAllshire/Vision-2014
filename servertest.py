from server import udp_send
import time

while True:
    packet = [5, 8, 3]
    udp_send(packet)
    print "sent"
    time.sleep(0.1)
