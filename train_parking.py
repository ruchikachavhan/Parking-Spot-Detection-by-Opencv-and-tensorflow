import tensorflow as tf
import cv2
import numpy as np
import os

# Number of classes=3(right, left,up)
# Defining training examples and labels
empty=[]
occupied=[]
examples=[]
error= 'error.txt'
error_file= open(error, 'w')
#Reading images
os.chdir('/home/ruchika/parking-spot/dataset-empty')
cur=os.getcwd() 
for image in os.listdir(cur):
	img= cv2.imread(image,0)
	re= cv2.resize(img, (50,50))
	empty.append(re)

os.chdir('/home/ruchika/parking-spot/dataset-occupied')
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

#Dividing data in train and validation sets
validation_set = int(len(examples)*0.6)
train_x, val_x = examples[:validation_set], examples[validation_set:]
train_y, val_y = labels[:validation_set], labels[validation_set:]
# Defining hyper parameters of neural network
print(len(train_x))
print(len(val_x))
print(np.shape(train_x[0])) 
learning_rate= 0.00001
epochs= 30
#Defining model of the neural network
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

#Defining placeholders
x = tf.placeholder(tf.float32, [50,50])
input_x = tf.reshape(x, [-1, 50,50, 1])
y = tf.placeholder(tf.float32, [2])

#Calculating convolutional layer
layer1 = create_new_conv_layer(input_x, 1, 32, [5, 5], [2, 2], name='layer1')
# layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
# layer3 = create_new_conv_layer(layer2, 64, 128, [5, 5], [2, 2], name='layer3')
print(layer1)
#Flattening 
flattened = tf.reshape(layer1, [-1,25*25*32 ])

#Defining the rest of the model
#Layer1
weights1 = tf.get_variable(initializer=tf.truncated_normal([25*25*32, 100], stddev=0.03), name='weights1')
b1= tf.get_variable(initializer= tf.truncated_normal([100], stddev=0.01), name='b1')
dense_layer1 =tf.add( tf.matmul(flattened, weights1), b1)
dense_layer1 = tf.nn.relu(dense_layer1)
#Layer2
weights2 = tf.get_variable(initializer=tf.truncated_normal([100, 2], stddev=0.03), name='weights2')
b2 = tf.get_variable(initializer= tf.truncated_normal([2], stddev=0.01), name='b2')
dense_layer2 = tf.add(tf.matmul(dense_layer1, weights2), b2)
y_ = tf.nn.softmax(dense_layer2)
#Calculating cross- entropy
print("dense_layer2",tf.transpose(dense_layer2))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
sess= tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(epochs):
    avg_cost = 0.0
    for i in range(0, len(train_x)):
    	_, c= sess.run([optimiser, cross_entropy], feed_dict= {x: train_x[i], y:train_y[i]})
    	avg_cost=avg_cost+c
    print('avg', avg_cost)

# Testing in validation set
cost_val=0.0
for v in range(0, len(val_x)):
	cost_val=cost_val+sess.run(cross_entropy,feed_dict= {x: val_x[v], y:val_y[v]} )
print('cost on val set', cost_val/len(val_x))

#TESTSET
os.chdir('/home/ruchika/parking-spot/')
test_image=cv2.imread('test2.jpeg',0)
r= test_image[62:105, 70:123]
test_resized=cv2.resize(r, (50,50))
print('test',sess.run(y_ , feed_dict= {x: test_resized, y:[0,1] }))
