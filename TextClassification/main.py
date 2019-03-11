#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# Download the IMDB dataset
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# The words from each review have been converted into integers that represent an index in some word dictionary
# print(train_data[0]) prints a list of integers that represent the first review

# dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_review(train_data[0])) Decodes the first review into a readable format

'''
Each review has different length and we need equal length to input it into the model so we can:
 1. Convert the arrays into vectors of 0s and 1s indicating word occurrence, similar to a one-hot encoding. 
 For example, the sequence [3, 5] would become a 10,000-dimensional vector that is all zeros except for 
 indices 3 and 5, which are ones. Then, make this the first layer in our network—a Dense layer—that can 
 handle floating point vector data. This approach is memory intensive, though, 
 requiring a num_words * num_reviews size matrix.

 2. Alternatively, we can pad the arrays so they all have the same length, then create an integer 
 tensor of shape max_length * num_reviews. We can use an embedding layer capable of handling this 
 shape as the first layer in our network.

 In this example I use the second approach 
'''

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# Build Model

# Input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# model.summary()  Prints how the NN its represended (layers, outputs, etc)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])



# Creating a validation set:
'''
When training, we want to check the accuracy of the model on data it hasn't seen before.
Creating a validation set by setting apart 10,000 examples from the original training data.
(Why not use the testing set now? Our goal is to develop and tune our model using only 
the training data, then use the test data ***just once*** to evaluate our accuracy).
'''

partial0_validation = train_data[:10000]
partial0_train = train_data[10000:]

partial1_validation = train_labels[:10000]
partial1_train = train_labels[10000:]


# Train
history = model.fit(partial0_train,
                    partial1_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(partial0_validation, partial1_validation),
                    verbose=1)



# Evaluate
results = model.evaluate(test_data, test_labels)
print(results)


# Generate Graph
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('demo.png', bbox_inches='tight')
