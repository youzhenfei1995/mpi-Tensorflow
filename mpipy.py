from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from mpi4py import MPI
import numpy as np
import tensorflow as tf
from six.moves import urllib
import os, sys
import time

from convolutional import extract_data, extract_labels, error_rate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MPI_OPTIMAL_PATH'] = '1'

DATA_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
iteration = 2
image_size = 28
batch_size = 64
num_channel = 10

class Cnn:
  def __init__(self, comm, rec_trd, rec_trl, rec_tsd, rec_tsl, rec_vd, rec_vl):
    # Receive test/training/validation dataset for each process
    self.comm = comm
    self.train_data = rec_trd
    self.train_label = rec_trl
    self.test_data = rec_tsd
    self.test_label = rec_tsl
    self.val_data = rec_vd
    self.val_label = rec_vl
    self.train_data_node = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size, image_size, 1))
    self.train_label_node = tf.placeholder(tf.int32, shape=(batch_size,))
    self.eval_data = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size, image_size, 1))
    self.conv1_weight = tf.Variable(tf.truncated_normal([5, 5, 1, 32],
                                    stddev=0.1,
                                    seed=1, dtype=tf.float32))
    self.conv1_bias = tf.Variable(tf.zeros([32]), dtype=tf.float32)
    self.conv2_weight = tf.Variable(tf.truncated_normal([5, 5, 32, 64], 
                                    stddev=0.1,
                                    seed=1, dtype=tf.float32))
    self.conv2_bias = tf.Variable(tf.constant(0.1, shape=[64]), dtype=tf.float32)
    self.fc1_weight = tf.Variable(tf.truncated_normal([image_size//4 * image_size//4 * 64, 512],
                                  stddev=0.1,
                                  seed=1,dtype=tf.float32))
    self.fc1_bias = tf.Variable(tf.constant(0.1, shape=[512]), dtype=tf.float32)
    self.fc2_weight = tf.Variable(tf.truncated_normal([512, num_channel],
                                  stddev=0.1,
                                  seed=1,dtype=tf.float32))
    self.fc2_bias = tf.Variable(tf.constant(0.1, shape=[num_channel]), dtype=tf.float32)
    self.logits = self.model(self.train_data_node)  
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	      labels=self.train_label_node, logits=self.logits))
    self.loss += 5e-4 * (tf.nn.l2_loss(self.fc1_weight) + tf.nn.l2_loss(self.fc1_bias)+
                         tf.nn.l2_loss(self.fc2_weight) + tf.nn.l2_loss(self.fc2_bias))
    self.iter_ = tf.Variable(0, dtype='float32')
    self.learning_rate = tf.train.exponential_decay(0.01, 
                                                   self.iter_ * batch_size,
                                                   self.train_data.shape[0],
                                                   0.95,
                                                   staircase=True)
    self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss, 
                                                                         global_step = self.iter_)
    self.train_prediction = tf.nn.softmax(self.logits)       
    self.eval_prediction = tf.nn.softmax(self.model(self.eval_data))
    self.config = tf.ConfigProto()
    self.config.gpu_options.allow_growth = True
    self.config.gpu_options.allocator_type = "BFC"
    with tf.Session(config=self.config) as self.sess:
      tf.global_variables_initializer().run()
      self.run_process(self.sess)

  def run_process(self, sess):
      print("Process ID:",self.comm.Get_rank()," training session starts!")
      #self.start_time = time.time()
      for step in xrange(iteration * self.train_label.shape[0] // batch_size):
        offset = (step * batch_size) % (self.train_label.shape[0] - batch_size)
        batch_data = self.train_data[offset:(offset + batch_size), ...]
        batch_label = self.train_label[offset:(offset + batch_size)]
        feed_dict = {self.train_data_node: batch_data,
                     self.train_label_node: batch_label}
        sess.run(self.optimizer, feed_dict=feed_dict)
        test_error = error_rate(self.eval_in_batches(self.test_data, sess), self.test_label)
        if step>0 and (step%50==0):
          print(self.comm.Get_rank(),' process at ', step, 'with test error: %.1f%%' % test_error)
          sess.run([self.loss, self.learning_rate, self.train_prediction], feed_dict=feed_dict)
          sys.stdout.flush()
          self.bcast_parameters(sess)
          #if step%10 == 0:
          #self.comm.Barrier()

  def  bcast_parameters(self,sess):
       recvbuf_conv1 = None
       #recvbuf_conv1_bias = None
       recvbuf_conv2 = None
       #recvbuf_conv2_bias = None
       recvbuf_fc1 = None
       #recvbuf_fc1_bias = None
       recvbuf_fc2 = None
       #recvbuf_fc2_bias = None
       if self.comm.Get_rank() == 0:
         recvbuf_conv1 = np.zeros([self.comm.Get_size()] +
                                    self.conv1_weight.get_shape().as_list(), dtype='f')
         #recvbuf_conv1_bias = np.zeros([self.comm.Get_size()] +
         #                           self.conv1_bias.get_shape().as_list(), dtype='f')
         recvbuf_conv2 = np.zeros([self.comm.Get_size()] +
                                    self.conv2_weight.get_shape().as_list(), dtype='f')
         #recvbuf_conv2_bias = np.zeros([self.comm.Get_size()] +
         #                           self.conv2_bias.get_shape().as_list(), dtype='f')
         recvbuf_fc1 = np.zeros([self.comm.Get_size()] +
                                  self.fc1_weight.get_shape().as_list(), dtype='f')
         #recvbuf_fc1_bias = np.zeros([self.comm.Get_size()] +
         #                         self.fc1_bias.get_shape().as_list(), dtype='f')
         recvbuf_fc2 = np.zeros([self.comm.Get_size()] +
                                  self.fc2_weight.get_shape().as_list(), dtype='f')
         #recvbuf_fc2_bias = np.zeros([self.comm.Get_size()] +
         #                         self.fc2_bias.get_shape().as_list(), dtype='f')
       self.comm.Gather(self.conv1_weight.eval(), recvbuf_conv1, root = 0)
       #self.comm.Gather(self.conv1_bias.eval(), recvbuf_conv1_bias, root = 0)
       self.comm.Gather(self.conv2_weight.eval(), recvbuf_conv2, root = 0)
       #self.comm.Gather(self.conv2_bias.eval(), recvbuf_conv2_bias, root = 0)
       self.comm.Gather(self.fc1_weight.eval(), recvbuf_fc1, root = 0)
       #self.comm.Gather(self.fc1_bias.eval(), recvbuf_fc1_bias, root = 0)
       self.comm.Gather(self.fc2_weight.eval(), recvbuf_fc2, root = 0)
       #self.comm.Gather(self.fc2_bias.eval(), recvbuf_fc2_bias, root = 0)
       if np.any(recvbuf_fc2):
         conv1_weight_mean = tf.convert_to_tensor(np.mean(recvbuf_conv1, 0), dtype=tf.float32)
         #conv1_bias_mean = tf.convert_to_tensor(np.mean(recvbuf_conv1_bias, 0), dtype=tf.float32)
         conv2_weight_mean = tf.convert_to_tensor(np.mean(recvbuf_conv2, 0), dtype=tf.float32)
         #conv2_bias_mean = tf.convert_to_tensor(np.mean(recvbuf_conv2_bias, 0), dtype=tf.float32)
         fc1_weight_mean = tf.convert_to_tensor(np.mean(recvbuf_fc1, 0), dtype=tf.float32)
         #fc1_bias_mean = tf.convert_to_tensor(np.mean(recvbuf_fc1_bias, 0), dtype=tf.float32)
         fc2_weight_mean = tf.convert_to_tensor(np.mean(recvbuf_fc2, 0), dtype=tf.float32)
         #fc2_bias_mean = tf.convert_to_tensor(np.mean(recvbuf_fc2_bias, 0), dtype=tf.float32)
         self.assign_conv1_oper = tf.assign(self.conv1_weight, conv1_weight_mean)
         #self.assign_conv1_bias_oper = tf.assign(self.conv1_bias, conv1_bias_mean)
         self.assign_conv2_oper = tf.assign(self.conv2_weight, conv2_weight_mean)
         #self.assign_conv2_bias_oper = tf.assign(self.conv2_bias, conv2_bias_mean)
         self.assign_fc1_oper = tf.assign(self.fc1_weight, fc1_weight_mean)
         #self.assign_fc1_bias_oper = tf.assign(self.fc1_bias, fc1_bias_mean)
         self.assign_fc2_oper = tf.assign(self.fc2_weight, fc2_weight_mean)
         #self.assign_fc2_bias_oper = tf.assign(self.fc2_bias, fc2_bias_mean)
         sess.run(self.assign_conv1_oper)
         #sess.run(self.assign_conv1_bias_oper)
         sess.run(self.assign_conv2_oper)
         #sess.run(self.assign_conv2_bias_oper)
         sess.run(self.assign_fc1_oper)
         #sess.run(self.assign_fc1_bias_oper)
         sess.run(self.assign_fc2_oper)
         #sess.run(self.assign_fc2_bias_oper)

  def model(self, data):
    self.conv = tf.nn.conv2d(data, self.conv1_weight, strides=[1,1,1,1], padding='SAME')
    self.relu = tf.nn.relu(tf.nn.bias_add(self.conv, self.conv1_bias))
    self.pool = tf.nn.max_pool(self.relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    self.conv = tf.nn.conv2d(self.pool, self.conv2_weight, strides=[1,1,1,1], padding='SAME')
    self.relu = tf.nn.relu(tf.nn.bias_add(self.conv, self.conv2_bias))
    self.pool = tf.nn.max_pool(self.relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    pool_shape = self.pool.get_shape().as_list()
    reshape = tf.reshape(self.pool,
                         [pool_shape[0], pool_shape[1]*pool_shape[2]*pool_shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weight) + self.fc1_bias)
    hidden = tf.nn.dropout(hidden, 0.5, seed=1)
    return tf.matmul(hidden, self.fc2_weight) + self.fc2_bias

  def eval_in_batches(self, data, sess):
    size = data.shape[0]
    if size < batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, num_channel), dtype=np.float32)
    for begin in xrange(0, size, batch_size):
      end = begin + batch_size
      if end <= size:
        predictions[begin:end, :] = sess.run(self.eval_prediction,
                                             feed_dict={self.eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(self.eval_prediction,
                                     feed_dict={self.eval_data: data[-batch_size:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

def data_exist_here(data_file_name):
  cur_path = os.getcwd()
  if not os.path.isdir(cur_path+'/data'):
    try:
       os.mkdir(cur_path+'/data')
    except OSError:
       print("Create data folder failed at current path!")
  # Ready for download dataset
  file_path = os.path.join(cur_path+'/data', data_file_name)
  if not os.path.exists(file_path):
    try:
       file_path, _ = urllib.request.urlretrieve(DATA_URL + data_file_name, file_path)
    except DownloadError:
       print(data_file_name + "download failed!")
  return file_path

def main():
  # Download Dataset
  tr_data_fname = data_exist_here('train-images-idx3-ubyte.gz')
  tr_label_fname = data_exist_here('train-labels-idx1-ubyte.gz')
  ts_data_fname = data_exist_here('t10k-images-idx3-ubyte.gz')
  ts_label_fname = data_exist_here('t10k-labels-idx1-ubyte.gz')
  # Ready to process data with MPI
  comm = MPI.COMM_WORLD
  rank_ = comm.Get_rank()
  size_ = comm.Get_size()
  tr_size = 55000//size_*size_
  ts_size = 10000//size_*size_
  val_size = 5000//size_*size_
  if rank_ == 0:
    tr_data = extract_data(tr_data_fname, 60000)
    tr_label = extract_labels(tr_label_fname, 60000)
    ts_data = extract_data(ts_data_fname, 10000//size_*size_)
    ts_label = extract_labels(ts_label_fname, 10000//size_*size_)
    val_data = tr_data[:5000//size_*size_,...]
    val_label = tr_label[:5000//size_*size_]
    tr_data = tr_data[5000//size_*size_:tr_size,...]
    tr_label = tr_label[5000//size_*size_:tr_size]
  else:
    tr_data = None
    tr_label = None
    ts_data = None
    ts_label = None
    val_data = None
    val_label = None
  recv_tr_data_buf = np.zeros([tr_size//size_,28,28,1], dtype='float32')
  recv_tr_label_buf = np.zeros([tr_size//size_,], dtype='uint64')
  recv_ts_data_buf = np.zeros([ts_size//size_,28,28,1], dtype='float32')
  recv_ts_label_buf = np.zeros([ts_size//size_,], dtype='uint64')
  recv_val_data_buf = np.zeros([val_size//size_,28,28,1], dtype='float32')
  recv_val_label_buf = np.zeros([val_size//size_,], dtype='uint64')
  comm.Scatter(tr_data, recv_tr_data_buf, root=0)
  comm.Scatter(tr_label, recv_tr_label_buf, root=0)
  comm.Scatter(ts_data, recv_ts_data_buf, root=0)
  comm.Scatter(ts_label, recv_ts_label_buf, root=0)
  comm.Scatter(val_data, recv_val_data_buf, root=0)
  comm.Scatter(val_label, recv_val_label_buf, root=0)
  train_proc = Cnn(comm, recv_tr_data_buf, recv_tr_label_buf,
                   recv_ts_data_buf, recv_ts_label_buf,
                   recv_val_data_buf, recv_val_label_buf)

if __name__ == "__main__":
  main()
