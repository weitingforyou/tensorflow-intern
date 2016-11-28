import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell


H = 256 # number of hidden layer neurons
D = 4 # input dimensionality
C = 2 # ouput classes 
#C = 1
n_input = 2 # MNIST data input (img shape: 2*2)
n_steps = 2 # timesteps


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

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=1.0)

outputs, states = rnn.rnn(lstm_cell, status4, dtype=tf.float32)

score = tf.matmul(outputs[-1], W2) + b2


loss = tf.reduce_sum(tf.square(input_y*advantages - score)) 
#optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
#optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss)
#optimizer = tf.train.FtrlOptimizer(learning_rate=0.001).minimize(loss)


all_status = [1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]
#all_action = [1,        1,        1,        1,        1,        1,        1,        1,        1,        1,     ]
all_action = [1,0],    [1,0],    [1,0],    [1,0],    [1,0],    [1,0],    [1,0],    [0,1],    [0,1],    [0,1]    
all_reward = [-1,       -1,       -1,       -1,       -1,       -1,       -1,       10,       10,       10,    ]


episode_number = 0
total_episodes = 5000
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
        
        #print 'all_action*all_reward:', all_action*all_reward
      
        p1 = sess.run(score,feed_dict={status: all_status})
        #print episode_number, 'p1:', p1
        
        sess.run(optimizer ,feed_dict={status: all_status, input_y: all_action, advantages: all_reward})  
        
        p2 = sess.run(score,feed_dict={status: all_status})        
        print episode_number
        print p2
        
        #print 'p2-p1:', p2-p1
        
        print 'loss:', sess.run(loss ,feed_dict={status: all_status, input_y: all_action, advantages: all_reward})  
        
        #print 'all_action*all_reward:', all_action*all_reward
        #print 'abs(all_action*all_reward-p2):', abs(all_action*all_reward-p2)
        #print 'np.sum(abs(all_action*all_reward-p2):', np.sum(abs(all_action*all_reward-p2))
