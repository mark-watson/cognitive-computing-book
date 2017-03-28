import tensorflow as tf
import numpy as np

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename="train.csv", target_dtype=np.int, features_dtype=np.float32)
test_set =     tf.contrib.learn.datasets.base.load_csv_without_header(filename="test.csv",  target_dtype=np.int, features_dtype=np.float32)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=9)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[12, 12, 12], n_classes=2)

# note: to set L1 or L2 regularization (for overfitting), learning rate, etc.
#       then use DNNClassifier options described in:
# https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.learn.html#DNNClassifier

classifier.fit(x=training_set.data, y=training_set.target, steps=500)

accuracy = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy))

no_and_yes_samples = np.array(
        [[4,1,1,3,2,1,3,1,1], [3,7,7,4,4,9,4,8,1]], dtype=np.int)
y = classifier.predict(no_and_yes_samples)
for yval in y:
  print ('Predictions: {}'.format(str(yval)))


