#######################################
#                                     #
#      Beta SEEFOOD SERVER            #
#                                     #
#######################################
import socket
import tensorflow as tf
import find_food_2 as find_food

# Setup the server socket
HOST = ""                # server ip
PORT = 2025              # SeeFood highway port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #create the socket
s.bind((HOST, PORT)) #bind the hostname and port number to the socket
print "Server running", HOST, PORT  #our server is up and running
s.listen(5)                #start listening for input

# Do all of the AI setup only once
sess = find_food.create_session()  #create an AI session
saver = tf.train.import_meta_graph('saved_model/model_epoch5.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('saved_model/'))
graph = tf.get_default_graph()
x_input = graph.get_tensor_by_name('Input_xn/Placeholder:0')
keep_prob = graph.get_tensor_by_name('Placeholder:0')
class_scores = graph.get_tensor_by_name("fc8/fc8:0")
print "We have created a SeeFood Session"

while True:
    #continuously listen for data
    conn, addr = s.accept()
    print'Connected by', addr

    #listen for data
    while True:
        print'Listening for data'
        #input stream
        data = conn.recv(1024)
        if not data: break

        #the data has been received
        print 'Data received'
        # Run the AI on the image
        tensor = find_food.analyze_image(data, sess, saver, graph, x_input, keep_prob, class_scores)
        
        #return the value
        conn.send(str(tensor))

        #close the connection and restart the listening process
        #NOTE: If we don't close the connection, then the next image request
        #      won't be processed until 1024 bits of data are sent in as a request
        #      which stalls the program. Connections are fast to form and we don't
        #      want to guess how long the input string will be so we just don't.
        print 'Closing connection'
        print '' # For formatting. Keeps each analysis nicely grouped in server console
        conn.close()
        break



