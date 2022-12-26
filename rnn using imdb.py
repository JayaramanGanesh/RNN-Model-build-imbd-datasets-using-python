
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import imdb


max_features = 1000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32



# modify the default parameters of np.load
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


#np.load = np_load_old
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


#Visualize the data
INDEX_FROM=3   # word index offset
word_to_id = imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id[""] = 0
word_to_id[""] = 1
word_to_id[""] = 2
id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in x_train[10] ))


#Build model...
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 8))
model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Write the training input and output, batch size, and testing input and output
model.fit(x_train, y_train, batch_size=batch_size, epochs=1,validation_data=(x_test, y_test))


#testing the code
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

#prediction 
prediction = model.predict(x_test[22220:22221])
print('Prediction value:',prediction[0])
print('Test Label:',y_test[22220:22221])

