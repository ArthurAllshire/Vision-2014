import socket
import threading
import time
from optparse import OptionParser

from ball import BallFinder
from goal import GoalFinder

HOST = "" # Bind to all address on specified port
# Take command line options if they exist. Useful for testing
parser = OptionParser()
parser.add_option("-c", "--crio", action="store", type="string", dest="CRIO", default = "10.47.74.2")
parser.add_option("-p", "--port", action="store", type="int", dest="PORT", default = 4774)
(options, args) = parser.parse_args()
CRIO = options.CRIO
PORT = options.PORT

CRIO_TIMEOUT = 2.0 # Stop sending if we don't hear from the cRIO for this long

gf = GoalFinder()
bf = BallFinder(width = 160, height = 120)

ball_finding = threading.Event()
goal_finding = threading.Event()

lastrecv = time.time()

recvsock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
recvsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)                   
recvsock.bind((HOST, 4774)) # Any interface, port 4774
sendsock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

def find_goal():
    while True:
        goal_finding.wait()
        gf.find()
        # Output a UDP packet to whoever is listening
        response = 'g{} {} {}'.format(gf.grange, gf.angle, gf.dummy)
        print response
        sendsock.sendto(response, (CRIO, PORT))
       
def find_ball():
    while True:
        ball_finding.wait()
        bf.find()
        # Output a UDP packet to whoever is listening
        if bf._is_red:
            response = 'r{} {} {}'.format(bf.xbar, bf.ybar, bf.diam)
        else:
            response = 'b{} {} {}'.format(bf.xbar, bf.ybar, bf.diam)
        print response
        sendsock.sendto(response, (CRIO, PORT))
        
def get_mode():
    '''Get the desired mode from the cRIO - whether to search for
    the goal or a particular colour of ball.'''
    while True:
        data, addr = recvsock.recvfrom(8) # buffer size is 8 bytes
        print "Recieved from cRIO:", data
        lastrecv = time.time()
        # The recvfrom call will block until data received, so we just wait
        if data[0].lower() == 'g':
            # We want the goal tracking thread
            ball_finding.clear()
            goal_finding.set()
        else:
            bf.setColour(data[0].lower())
            #Catching thread
            goal_finding.clear()
            ball_finding.set()

def watchdog():
    '''Stop sending packets if we haven't heard from the cRIO in the last 2 seconds.'''
    if time.time() - lastrecv > CRIO_TIMEOUT:
        # We haven't heard from the cRIO - stop the sending threads
        goal_finding.clear()
        ball_finding.clear()
        print "Halted on lost cRIO communication." 
    # Respawn the thread
    watchdog_thread = threading.Timer(CRIO_TIMEOUT, watchdog)
    watchdog_thread.daemon = True
    watchdog_thread.start()
    
if __name__ == "__main__":
    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=get_mode, name="udpserver")
    # Exit the server thread when the main thread terminates
    # Makes breaking out of program with Ctrl-C easier
    server_thread.daemon = True
    server_thread.start()
    # A watchdog thread to stop sending data if the cRIO isn't asking
    # The cRIO seems to get upset if the packets don't get consumed
    # (even though it's UDP - weird).
    watchdog_thread = threading.Timer(CRIO_TIMEOUT, watchdog)
    watchdog_thread.daemon = True
    watchdog_thread.start()
    
    
    goal_thread = threading.Thread(target=find_goal, name = 'finding goal')
    goal_thread.daemon = True
    goal_finding.clear()
    goal_thread.start()
    
    ball_thread = threading.Thread(target=find_ball, name = 'finding ball')
    ball_thread.daemon = True
    ball_finding.clear()
    ball_thread.start()
    
    while True:
        pass
        
        # Infinite loop to keep the main thread running
        # and waiting for connections
        # Makes breaking out of program with Ctrl-C easier
        #pass

