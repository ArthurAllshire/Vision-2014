from socket import *

sock = False
PORT = 4775

def make_packet(to_send):
    as_string = str(to_send)
    as_string = as_string.replace("[", "")
    as_string = as_string.replace("]", "")
    as_string = as_string.replace(" ", "")
    return as_string

def make_socket():
    global sock
    sock = socket(AF_INET, SOCK_DGRAM)
    sock.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)

def udp_send(to_send):
    global sock, PORT
    if not sock:
        make_socket()

    packet = make_packet(to_send)

    sock.sendto(packet, ('<broadcast>', PORT))
