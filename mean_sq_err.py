import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

vec_1 = tf.placeholder("float32", shape = (None, ))
vec_2 = tf.placeholder("float32", shape = (None, ))

mse = tf.metrics.mean_squared_error(vec_1, vec_2)

compute_mse = lambda v_1, v_2: mse.eval({vec_1: v_1, vec_2: v_2})
#Не пойму в чем ошибка =(

for n in [1, 5, 10, 10 ** 3]:

    elems = [np.arange(n), np.arange(n, 0, -1), np.zeros(n),
             np.ones(n), np.random.random(n), np.random.randint(100, size=n)]

    for el in elems:
        for el_2 in elems:
            true_mse = np.array(mean_squared_error(el, el_2))
            my_mse = compute_mse(el, el_2)
            if not np.allclose(true_mse, my_mse):
                print('Wrong result:')
                print('mse(%s,%s)' % (el, el_2))
                print("should be: %f, but your function returned %f" % (true_mse, my_mse))
                raise ValueError

print("All tests passed")
