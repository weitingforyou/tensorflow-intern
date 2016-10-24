import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.python.ops import rnn, rnn_cell
from yahoo_finance import Share


day_len = 9
H = 256 # number of hidden layer neurons
D = 9 # input dimensionality
C = 1 # ouput classes 
n_input = 3 # MNIST data input (img shape: 2*2)
n_steps = 3 # timesteps

# get 2330 stock data from yahoo-finance
stock = Share('2330.TW')
today = datetime.date.today()
stock_data = stock.get_historical('2016-01-01', str(today))
stock_data.reverse()
print len(stock_data)

# delete the zero-volumn data
i = 0
while( i < len(stock_data)):
    if (int(stock_data[i].get('Volume')) <= 0):
        stock_data.remove(stock_data[i])
        i = -1
    i += 1
print 'data without zero volumn', len(stock_data)

# training and testing data
data_X = np.zeros((len(stock_data)-day_len,day_len), dtype=np.float)
data_Y = np.ones((len(stock_data)-day_len,1), dtype = np.float)

for i in xrange(0, len(data_X)):
    for j in xrange(0, day_len):
        data_X[i,j] = float(stock_data[i+j].get('Close'))
    data_Y[i,0] = float(stock_data[i+day_len].get('Close'))
    
len_train = int(len(data_X) * 0.8)
len_test = len(data_X) - len_train

train_X = data_X[0:len_train]
train_Y = data_Y[0:len_train]
test_X = data_X[len_train::]
test_Y = data_Y[len_train::]


# LSTM
# placeholder
input_x = tf.placeholder(tf.float32, [None,D], name="input_x")
input_y = tf.placeholder(tf.float32, [None,C], name="input_y")
# weights and biases
w1 = tf.get_variable("W1", shape=[H, C], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.constant(0.01, shape = [C], name = "B1"))

input_x1 = tf.reshape(input_x, [-1,n_steps, n_input])
input_x2 = tf.transpose(input_x1, [1, 0, 2])
input_x3 = tf.reshape(input_x2, [-1, n_input])
input_x4 = tf.split(0, n_steps, input_x3)

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=1.0)

outputs, states = rnn.rnn(lstm_cell, input_x4, dtype=tf.float32)

score = tf.matmul(outputs[-1], w1) + b1


loss = tf.reduce_mean(tf.reduce_sum(tf.square(score-input_y),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) 


init = tf.initialize_all_variables()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax2 = fig2.add_subplot(1,1,1)
    ax1.plot(train_Y, 'b.')
    ax2.plot(test_Y, 'b.')
    plt.ion()
    plt.show()

    for i in range(30000):
    
        # training
        sess.run(train_step, feed_dict={input_x: train_X, input_y: train_Y})
        
        if i % 50 == 0:
            # to visualize the result and improvement
            try:
                ax1.lines.remove(lines1[0])
            except Exception:
                pass
                
            print i
            print 'loss:', sess.run(loss, feed_dict={input_x:train_X, input_y: train_Y})
            
            train_prediction = sess.run(score, feed_dict={input_x: train_X})
            #print train_prediction.reshape([len(train_prediction)])
            
            # plot the prediction
            ax1.set_ylim(min(np.min(train_prediction),np.min(train_Y))*0.95,max(np.max(train_prediction), np.max(train_Y))*1.05)
            lines1 = ax1.plot(train_prediction, 'r-', lw=4)


            # testing
            try:
                ax2.lines.remove(lines2[0])
            except Exception:
                pass
                
            test_prediction = sess.run(score, feed_dict={input_x: test_X})
            #print test_prediction.reshape([len(test_prediction)]) 
            
            # plot the prediction
            ax2.set_ylim(min(np.min(test_prediction),np.min(test_Y))*0.95,max(np.max(test_prediction), np.max(test_Y))*1.05)
            lines2 = ax2.plot(test_prediction, 'r-', lw=4)
            plt.pause(0.1)
