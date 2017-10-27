from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

mnist = load_digits(2)

X,y = mnist.data, mnist.target

print("y [shape - %s]:" % (str(y.shape)), y[:10])
print("X [shape - %s]:" % (str(X.shape)))

print('X:\n',X[:3,:10])
print('y:\n',y[:10])
plt.imshow(X[0].reshape([8,8]))
plt.show()

weights = tf.Variable(initial_value=np.ones(5))
input_X = tf.placeholder("float32", shape=(None, None,))
input_y = tf.placeholder("float32", shape=(None, ))

predicted_y = tf.placeholder("float32", shape=(None, ))
loss = tf.reduce_mean(predicted_y - input_y)
optimizer = tf.train.MomentumOptimizer(0.01,0.9).minimize(loss, var_list=predicted_y)

train_function = tf.losses.log_loss(X, y, weights)
predict_function = tf.nn.top_k(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

for i in range(5):
    loss_i = loss(i)

    print("loss at iter %i:%.4f" % (i, loss_i))

    print("train auc:", roc_auc_score(y_train, predict_function(X_train)))
    print("test auc:", roc_auc_score(y_test, predict_function(X_test)))

print("resulting weights:")
plt.imshow(tf.contrib.keras.backend.get_value(weights).reshape(8, -1))
plt.colorbar()
plt.show()