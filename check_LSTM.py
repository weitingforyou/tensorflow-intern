import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

H = 256 # number of hidden layer neurons
D = 4 # input dimensionality
C = 3 # ouput classes

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([D, H])),
    'out': tf.Variable(tf.random_normal([H, C]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([H,])),
    'out': tf.Variable(tf.random_normal([C,]))
}

status = tf.placeholder(tf.float32, [None,D] , name="input_x")
input_y = tf.placeholder(tf.float32,[None,C], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

def RNN(_status, _weights, _biases):
    _status = tf.reshape(_status, [-1,D])
    _status = tf.matmul(_status, _weights['hidden']) + _biases['hidden']
    _status = tf.reshape(_status, [-1,1,H])
    
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=1.0, state_is_tuple=True)
    '''
    If the version of your tensorflow is higher than r0.9, then your lstm_cell should try the following :
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=1.0, state_is_tuple=True)
    '''
    _init_state = lstm_cell.zero_state(10, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, _status, initial_state=_init_state, time_major=False)
    
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2])) 
    score = tf.matmul(outputs[-1], _weights['out']) + _biases['out']
    
    return score

score = RNN(status, weights, biases)
loss = tf.reduce_sum(tf.square(input_y*advantages - score)) 
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) 

all_status = [1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9],[7,8,9,10],[8,9,10,11],[9,10,11,12],[10,11,12,13]
all_action = [1,0,0],  [0,1,0],  [0,0,1],  [1,0,0],  [0,1,0],  [0,0,1],  [1,0,0],   [0,1,0],    [0,0,1],     [1,0,0]
all_reward = [1,        -2,        3,       -4,        5,       -6,       -7,        8,           -9,         10]

episode_number = 0
total_episodes = 10000
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
        #print episode_number, 'p1:', p1
        
        sess.run(optimizer ,feed_dict={status: all_status, input_y: all_action, advantages: all_reward})  
        
        p2 = sess.run(score,feed_dict={status: all_status})        
        print episode_number, 'p2:', p2
        
        #print 'p2-p1:', p2-p1
        
        print 'loss:', sess.run(loss ,feed_dict={status: all_status, input_y: all_action, advantages: all_reward})  

        #print 'all_action*all_reward:', all_action*all_reward
        #print 'abs(all_action*all_reward-p2):', abs(all_action*all_reward-p2)
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

