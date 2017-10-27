import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# my_scalar = tf.placeholder('float32')
# scalar_squared = my_scalar**2
# derivative = tf.gradients(scalar_squared, my_scalar)[0]
#
# x = np.linspace(-3,3)
# x_squared, x_squared_der = s.run([scalar_squared,derivative],
#                                  {my_scalar:x})
#
# plt.plot(x, x_squared,label="x^2")
# plt.plot(x, x_squared_der, label="derivative")
# plt.legend()
# plt.show()

my_scalar = tf.placeholder('float32')
my_vector = tf.placeholder('float32',[None])
weird_psychotic_function = tf.reduce_mean((my_vector+my_scalar)**(1+tf.nn.moments(my_vector,[0])[1]) + 1./ tf.atan(my_scalar))/(my_scalar**2 + 1) + 0.01*tf.sin(2*my_scalar**1.5)*(tf.reduce_sum(my_vector)* my_scalar**2)*tf.exp((my_scalar-4)**2)/(1+tf.exp((my_scalar-4)**2))*(1.-(tf.exp(-(my_scalar-4)**2))/(1+tf.exp(-(my_scalar-4)**2)))**2

der_by_scalar = tf.gradients(weird_psychotic_function, my_scalar)[0]
der_by_vector = tf.gradients(weird_psychotic_function, my_vector)[0]

scalar_space = np.linspace(1, 7, 100)

y = [s.run(weird_psychotic_function, {my_scalar:x, my_vector:[1, 2, 3]})
     for x in scalar_space]

plt.plot(scalar_space, y, label='function')

y_der_by_scalar = [s.run(der_by_scalar, {my_scalar:x, my_vector:[1, 2, 3]})
     for x in scalar_space]

plt.plot(scalar_space, y_der_by_scalar, label='derivative')
plt.grid()
plt.legend()
plt.show()