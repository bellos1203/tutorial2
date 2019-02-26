import tensorflow as tf
import numpy as np,sys
from  scipy.signal import convolve2d

np.random.seed(678)
tf.set_random_seed(6789)
sess = tf.Session()

# ========== Experiment Set Up ===========
# 0. Create a matrix we want to perform experiments
mat_size = 10
matrix = np.zeros((mat_size,mat_size)).astype(np.float32)
center = np.array([[1,0,1],[0,1,0],[-1,0,-1]])

for x in range(4,7):
    for y in range(4,7):
            matrix[y,x] = center[y-4,x-4]

# Create a Kernel 
kernel = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
]).astype(np.float32)
print("====== Original Set Up ======")
print("Matrix Shape : ",matrix.shape)
print(matrix)
print("kernel Shape : ",kernel.shape)
print(kernel)
# ========== Experiment Set Up ===========

# ========== EXAMPLE 1 - Dilation Factor 1 ===========
print("\n====== Dilated Kernel 1 ======")
print("========== Tensorfow Conv2D Results ===============")

##### Exercise 1-1 #####
matrix_tensor = tf.convert_to_tensor(matrix)
matrix_tensor = tf.expand_dims(matrix_tensor, axis=0)
matrix_tensor = tf.expand_dims(matrix_tensor, axis=3)

kernel_tensor = tf.convert_to_tensor(kernel)
kernel_tensor = tf.expand_dims(kernel_tensor, axis=2)
kernel_tensor = tf.image.rot90(kernel_tensor,k=2)
kernel_tensor = tf.expand_dims(kernel_tensor, axis=3)

strides = [1,1,1,1]
dilations = [1,1,1,1]

tf_operation1_1 = tf.nn.conv2d(matrix_tensor,
                            kernel_tensor,
                            strides=strides,padding="VALID",
                            dilations=dilations)
tf_result = sess.run(tf_operation1_1)
print("Tensorfow Conv2D Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))

print("========== Tensorfow Atrous Conv2D Results ===============")
tf_operation1_2 = tf.nn.atrous_conv2d(matrix_tensor,
                                    kernel_tensor,
                                    rate=1, padding="VALID")
tf_result = sess.run(tf_operation1_2)
print("Tensorfow Atrous Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))
# ========== EXAMPLE 1 - Dilation Factor 1 ===========



# ========== EXAMPLE 2 - Dilation Factor 2 ===========
print("\n====== Dilated Kernel 2 ======")
kernel2 = np.array([
    [1,0,2,0,3],
    [0,0,0,0,0],
    [4,0,5,0,6],
    [0,0,0,0,0],
    [7,0,8,0,9]
]).astype(np.float32)
print('Kernal 2: \n',kernel2)


##### Exercise 1-2 ######
kernel2_tensor = tf.convert_to_tensor(kernel2)
kernel2_tensor = tf.expand_dims(kernel2_tensor, axis=2)
kernel2_tensor = tf.image.rot90(kernel2_tensor,k=2)
kernel2_tensor = tf.expand_dims(kernel2_tensor, axis=3)

strides = [1,1,1,1]
dilations = [1,2,2,1]

print("========== Tensorfow Conv2D Results ===============")
tf_operation2_1 = tf.nn.conv2d(matrix_tensor,
                             kernel2_tensor,
                             strides=strides,padding="VALID",
                             dilations=dilations)
tf_result = sess.run(tf_operation2_1)
print("Tensorfow Conv2D Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))

print("========== Tensorfow Atrous Conv2D Results ===============")
tf_operation2_2 = tf.nn.atrous_conv2d(matrix_tensor,
                                    kernel2_tensor,
                                    rate=2,padding="VALID")
tf_result = sess.run(tf_operation2_2)
print("Tensorfow Atrous Results Shape: ",tf_result.shape)
print(np.squeeze(tf_result))
# ========== EXAMPLE 2 - Dilation Factor 2 ===========


# -- end code --
