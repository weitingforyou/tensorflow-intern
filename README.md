# tensorflow-intern
4th grade tensorflow intern <br>
There are two kinds of input data in this project. <br>
If the file name begins with "2330", it represents that the input data is the stock price of 2330(TSMC). <br>
<br>
<br>
## requirement
* tensorflow(r0.9)
* numpy
* matplotlib
* yahoo_finance (for downloading the stock price)
<br>
<br>

## Folder practice
Files in this folder is for practicing building the model. <br>
The input data includes state, action and rewards.
### check.py
Simple Tensorflow code for MLP model.
### check_LSTM.py
Simple Tensorflow code for LSTM(RNN) model.
### check_MultiRNNCell.py
Simple Tensorflow code for MultiLayer LSTM model.
<br>
<br>

## Folder stock
Files in this folder is for the implementation the various model. <br>
The input data is the stock price of 2330(TSMC), 2412(Chunghwa Telecom Co.) etc.
### 2330_LSTM.py
The input data is the stock price of 2330(TSMC). <br>
It can predict the 10th day's stock price with knowing the 1st~9th day's stock price. <br>
This file is training and testing with LSTM(RNN) model.
<br>
<br>
