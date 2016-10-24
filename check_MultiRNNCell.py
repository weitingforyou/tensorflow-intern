#-*- coding: cp950 -*-
# ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

import matplotlib.pyplot as plt

H = 256 # number of hidden layer neurons
D = 4 # input dimensionality
C = 3 # ouput classes 
n_input = 2 # MNIST data input (img shape: 2*2)
n_steps = 2 # timesteps
num_layers = 2 # number of cell layers

tf.reset_default_graph()

def bias_variable(name, shape):
    return tf.Variable(tf.constant(0.01, shape = shape, name = name))

status = tf.placeholder(tf.float32, [None,D] , name="input_x")
input_y = tf.placeholder(tf.float32,[None,C], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

W2 = tf.get_variable("W2", shape=[H, C], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.constant(0.01, shape = [C], name = "B2"))

status1 = tf.reshape(status, [-1,n_steps, n_input])
status2 = tf.transpose(status1, [1, 0, 2])
status3 = tf.reshape(status2, [-1, n_input])
status4 = tf.split(0, n_steps, status3)

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=1.0, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers, state_is_tuple=True)
_init_state = cell.zero_state(10, tf.float32)

outputs = []
state = _init_state
for time_step in range (n_steps):
    if time_step > 0: tf.get_variable_scope().reuse_variables()
    (cell_output, state) = cell(status4[time_step], state)
    outputs.append(cell_output)
output = tf.reshape(outputs[-1], [-1, H])
score = tf.matmul(output, W2) + b2

#outputs, states = rnn.rnn(lstm_cell, status4, dtype=tf.float32)
#score = tf.matmul(outputs[-1], W) + b2

'''
W1 = tf.get_variable("W1", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
b1 = bias_variable("B1", [H]) 
layer1 = tf.nn.relu(tf.matmul(status,W1) + b1)

W2 = tf.get_variable("W2", shape=[H, H*2],
           initializer=tf.contrib.layers.xavier_initializer())
b2 = bias_variable("B2", [H*2]) 
layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)

W3 = tf.get_variable("W3", shape=[H*2, 3],
           initializer=tf.contrib.layers.xavier_initializer())        
         
score = tf.matmul(layer2, W3) 
'''

loss = tf.reduce_sum(tf.square(input_y*advantages - score)) 

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) 

all_status = [1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9],[7,8,9,10],[8,9,10,11],[9,10,11,12],[10,11,12,13]
all_action = [1,0,0],  [0,1,0],  [0,0,1],  [1,0,0],  [0,1,0],  [0,0,1],  [1,0,0],   [0,1,0],    [0,0,1],     [1,0,0]
all_reward = [1,        -2,        3,       -4,        5,       -6,       -7,        8,           -9,         10]
    
episode_number = 0
total_episodes = 3000
init = tf.initialize_all_variables()

all_status = np.array(all_status)
all_action = np.vstack(all_action)
all_reward = np.vstack(all_reward)
my_list = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    fig, ax = plt.subplots()
    plt.ion()
    plt.show()

    while episode_number < total_episodes:
        episode_number += 1
        
        print 'all_action*all_reward:', all_action*all_reward
      
        p1 = sess.run(score,feed_dict={status: all_status})
        print episode_number, 'p1:', p1
        
        sess.run(optimizer ,feed_dict={status: all_status, input_y: all_action, advantages: all_reward})  
        
        p2 = sess.run(score,feed_dict={status: all_status})        
        print episode_number, 'p2:', p2
        
        print 'p2-p1:', p2-p1
        
        print 'loss:', sess.run(loss ,feed_dict={status: all_status, input_y: all_action, advantages: all_reward})  
        
        print 'all_action*all_reward:', all_action*all_reward
        print 'abs(all_action*all_reward-p2):', abs(all_action*all_reward-p2)
        print 'np.sum(abs(all_action*all_reward-p2):', np.sum(abs(all_action*all_reward-p2))
        my_list.append(np.sum(abs(all_action*all_reward-p2)))
        
        if (episode_number % 20 == 0):
            # ref : https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow12_plut_result.py
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
                
            new_my_list = my_list[max(0, len(my_list)-100): len(my_list)]
            x_list = np.arange(len(new_my_list))
            
            plt.ylim(np.min(new_my_list)*0.95,np.max(new_my_list)*1.05)
            lines  = ax.plot(x_list, new_my_list)
            plt.pause(0.01)

plt.pause(60)
