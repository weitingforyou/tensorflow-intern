#-*- coding: cp950 -*-

import tensorflow as tf
import numpy as np

H = 256 # number of hidden layer neurons
D = 4 # input dimensionality
regularation_param = 0.0001

tf.reset_default_graph()

def bias_variable(name, shape):
    initial = tf.constant(0.01, shape = shape, name = name)
    return tf.Variable(initial)

status = tf.placeholder(tf.float32, [None,D] , name="input_x")
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

input_y = tf.placeholder(tf.float32,[None,3], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(score, input_y)**(1/advantages))
#loss = tf.reduce_sum(tf.square(input_y*advantages - score)) 
l2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
loss = tf.reduce_sum(tf.square(input_y*advantages - score)) + regularation_param *l2_loss
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
        
        print 'np.sum(abs(all_action*all_reward-p2):', np.sum(abs(all_action*all_reward-p2))
        my_list.append(np.sum(abs(all_action*all_reward-p2)))

import matplotlib.pyplot as plt
x_list = []
for i in xrange (0, 3000):
    x_list.append(i)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x_list, my_list)
plt.show()
        
        
