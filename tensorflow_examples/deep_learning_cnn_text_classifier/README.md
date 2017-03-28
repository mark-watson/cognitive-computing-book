# CNN Deep Neural Network for text classification

        LICENSE: Apache 2

### Credits

Parts of this example are derived from example code at https://github.com/dennybritz/cnn-text-classification-tf (Apache 2 license) which in turn was partially derived from Yoon Kim's paper http://arxiv.org/abs/1408.5882


## Running the code

This code runs under TensorFlow version 0.11 and above using Python 2.7

install stemming library:

pip install stemming
 
source activate tensorflow27

./main.py

./main.py -test

tensorboard --logdir=runs/ --port=8080
