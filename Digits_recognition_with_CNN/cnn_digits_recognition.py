
# ## 1. Load MNIST Database

from keras.datasets import mnist

( x_train, y_train),( x_test, y_test ) = mnist.load_data()

# ## 2. Normalize the data 

# rescale [ 0, 255] -> [ 0, 1 ]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# 3. Encode categorical integer labels using a one-hot code



from keras.utils import np_utils

# print first ten (integer-valued) training labels
print('Integer-valued labels:')
print(y_train[:10])
print('\n\n\n')

# one-hot encode the labels
y_train = np_utils.to_categorical( y_train, 10 )
y_test = np_utils.to_categorical( y_test, 10 )

# print first ten (one-hot) training labels
print('One-hot labels:')
print(y_train[:10])



from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout 

model = Sequential()

# turn the matrix to vector
model.add( Flatten( input_shape = x_train.shape[ 1: ] ) )

# two hidden layers
model.add( Dense( 512 , activation = 'relu' ) )
model.add( Dropout( 0.2 ) )
model.add( Dense( 512 , activation = 'relu' ) )
model.add( Dropout( 0.2 ))

# output layer
model.add( Dense( 10 ) )
model.add( Activation('softmax' ) )

model.summary()

# ## 5. Compile the model

model.compile( loss = 'categorical_crossentropy', optimizer = 'rmsprop', metircs = ['accuracy'])


# 6. Evaluate the model

score = model.evaluate( x_test, y_test, verbose = 0)
accuracy = 100 * score[ 1 ]
print( 'Test accuracy: %.4f%%' % accuracy )



# 7. Train the model
# 
#  Validation set: certain numbers of datas splited from the training set, used to evaluate the model after each epoch, in order not to be overfitting

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint( filepath = 'mnist.model.best.hdf5',
                              verbose  = 1, save_best_only = True)

hist = model.fit( x_train, y_train, batch_size = 128, epochs=10,
                validation_split = 0.2, callbacks = [checkpointer],
                verbose = 1, shuffle = True)


# 8. Load the model with the best classification accuracy on the validation set


model.load_weights('mnist.model.best.hdf5')


# 9. Calculate the classification accuracy on the test set


score = model.evaluate( x_test, y_test, verbose=0)
accuracy = 100 * score[ 1 ]

print('Test accuracy: %.4f%%' %accuracy )




