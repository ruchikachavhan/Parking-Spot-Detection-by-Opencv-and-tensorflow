import tensorflow as tf
import cv2
import numpy as np
import os
import tensorflow.contrib.slim as slim  # TensorFlow-Slim

# Number of classes=3(right, left,up)
# Defining training examples and labels
empty=[]
occupied=[]
examples=[]
error= 'error.txt'
error_file= open(error, 'w')
#Reading images
os.chdir('/home/ruchika/parking-spot/train set empty')
cur=os.getcwd() 
for image in os.listdir(cur):
	img= cv2.imread(image,0)
	re= cv2.resize(img, (50,50))
	empty.append(re)

os.chdir('/home/ruchika/parking-spot/train set occupied')
cur=os.getcwd() 
for image in os.listdir(cur):
	img= cv2.imread(image,0)
	re= cv2.resize(img, (50,50))
	occupied.append(re)

# examples=[]
labels= np.zeros([len(empty)+ len(occupied)-1,2])
# Labels
for it in range(0, int(len(empty)/2)):
	examples.append(empty[it])
	labels[it][0]=1

for it in range(0,int(len(occupied)/2)):
	examples.append(occupied[it])
	labels[it+ int(len(empty)/2)][1]=1

for it in range(0 ,int(len(empty)/2)):
	examples.append(empty[int(len(empty)/2)+it])
	labels[it+int(len(empty)/2)+int(len(occupied)/2)][0]=1

for it in range(0, int(len(occupied)/2)):
	examples.append(occupied[it+ int(len(occupied)/2)])
	labels[it+len(empty)+int(len(occupied)/2)][1]=1
	
learning_rate= 0.00001
epochs= 30
# Defining model of the neural network
def create_new_conv_layer(input, channels, filters, filter_shape, pool_shape, name):
	convolutional_shape= [filter_shape[0], filter_shape[1], channels, filters]
	weights= tf.get_variable(initializer= tf.truncated_normal(convolutional_shape, stddev= 0.03), name= name+'w')
	bias= tf.get_variable(initializer=tf.truncated_normal([filters]), name= name+'b')
	out_layer= tf.nn.conv2d(input, weights, [1,1,1,1], padding='SAME')
	out_layer=out_layer+bias
	out_layer= tf.nn.relu(out_layer)
	ksize=[1, pool_shape[0], pool_shape[1], 1]
	strides=[1,2,2,1]
	out_layer= tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
	return out_layer
# #Defining placeholders
x = tf.placeholder(tf.float32, [50,50])
input_x = tf.reshape(x, [-1, 50,50, 1])
y = tf.placeholder(tf.float32, [2])
####First Network 
#Calculating convolutional layer
layer1 = create_new_conv_layer(input_x, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
print(layer1)
#Flattening 
flattened = tf.reshape(layer2, [-1,13*13*64 ])
sess= tf.Session()
# Defining the rest of the model
# Layer1
weights1 = tf.get_variable(initializer=tf.truncated_normal([13*13*64, 1000], stddev=0.03), name='weights1')
b1= tf.get_variable(initializer= tf.truncated_normal([1000], stddev=0.01), name='b1')
dense_layer1 =tf.add( tf.matmul(flattened, weights1), b1)
dense_layer1 = tf.nn.relu(dense_layer1)
#Layer2
net = tf.nn.dropout(dense_layer1, 0.5)
weights2 = tf.get_variable(initializer=tf.truncated_normal([1000, 2], stddev=0.03), name='weights2')
b2 = tf.get_variable(initializer= tf.truncated_normal([2], stddev=0.01), name='b2')
dense_layer2 = tf.add(tf.matmul(net, weights2), b2)
y_ = tf.nn.softmax(dense_layer2)
#Calculating cross- entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
sess= tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver= tf.train.Saver()
for epoch in range(epochs):
    avg_cost = 0.0
    for i in range(0, len(examples)):
	    _, c= sess.run([optimiser, cross_entropy], feed_dict= {x: examples[i], y:labels[i]})
	    avg_cost=avg_cost+c
    print('Number of iterations', epoch+1)
    print('avg', avg_cost)
    #Validati0n set
    os.chdir('/home/ruchika/parking-spot/validation set empty')
    empty_val=[]
    occupied_val=[] 
    for image in os.listdir(os.getcwd()):
    	img= cv2.imread(image,0)
    	re= cv2.resize(img, (50,50))
    	empty_val.append(re)
    os.chdir('/home/ruchika/parking-spot/validation set occupied')
    for image in os.listdir(os.getcwd()):
    	img= cv2.imread(image,0)
    	re= cv2.resize(img, (50,50))
    	occupied_val.append(re)
    val_x=[]
    val_y=np.zeros([len(empty_val)+len(occupied_val),2])
    for it in range(0, len(empty_val)):
    	val_x.append(empty_val[it])
    	val_y[it][0]=1
    for ir in range(0, len(occupied_val)):
    	val_x.append(occupied_val[ir])
    	val_y[ir+len(empty_val)][1]=1
    cost_val=0.0
    for v in range(0, len(val_x)):
    	cost_val=cost_val+sess.run(cross_entropy,feed_dict= {x: val_x[v], y:val_y[v]} )
    print('cost on val set', cost_val/len(val_x))
    #Test Set
    test_cost=0.0
    print('testing on empty')
    os.chdir('/home/ruchika/parking-spot/test set empty')
    for test_image in os.listdir(os.getcwd()):
    	test= cv2.imread(test_image,0)
    	test= cv2.resize(test,(50,50))
    	print(sess.run(y_, feed_dict= {x: test, y:[1,0]}))
    print('testing on occupied')
    os.chdir('/home/ruchika/parking-spot/test set occupied')
    for test_image in os.listdir(os.getcwd()):
    	test= cv2.imread(test_image,0)
    	test= cv2.resize(test,(50,50))
    	print(sess.run(y_, feed_dict= {x: test, y:[0,1]}))
os.chdir('/home/ruchika/parking-spot/')
#Saving the model
save_path= saver.save(sess, '/home/ruchika/parking-spot/model.ckpt')
