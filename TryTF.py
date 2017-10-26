import tensorflow as tf
import numpy as np
import time

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

def sum_squares(x):
    sum = 0
    for i in range(x):
        sum = sum + i**2
    return sum

# start = time.clock()
# sum_squares(10**8)
# elapsed = (time.clock() - start)
#
# print(elapsed)


# N = tf.placeholder("int64", name = "input_to_your_function")
# result = tf.reduce_sum((tf.range(N)**2))
#
# start_tf = time.clock()
# print(result.eval({N:10**8}))
# elapsed_tf = (time.clock() - start_tf)
#
# print(elapsed_tf)

#Practice time

my_vector = tf.placeholder("float32", shape = (None, ))
my_vector_2 = tf.placeholder("float32", shape = (None, ))

my_transfor = my_vector * my_vector_2 / (tf.sin(my_vector) + 1)
print(my_transfor)


dummy = np.arange(5).astype('float32')
print(dummy)

my_transfor.eval({my_vector:dummy,my_vector_2:dummy[::-1]})
