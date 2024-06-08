from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(oov_token='<nothing>')

file = open('Data.txt', 'r', encoding = 'utf-8')
text = file.read()
file.close()

tokenizer.fit_on_texts([text])
tok_len = len(tokenizer.word_index)

input_sequences = []
for sentences in text.split('\n'):
  tokenized_sen = tokenizer.texts_to_sequences([sentences])[0]
  for i in range(1,len(tokenized_sen)):
    input_sequences.append(tokenized_sen[:i+1])

max_len = max([len(x) for x in input_sequences])
max_len

from keras.preprocessing.sequence import pad_sequences
padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')

X = padded_input_sequences[:,:max_len-1]
y = padded_input_sequences[:,-1:]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes = tok_len + 1)

from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential

model = Sequential()
model.add(Embedding(tok_len + 1, 200, input_length = max_len - 1))
model.add(LSTM(250))
model.add(Dense(tok_len + 1, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

model.fit(X, y, epochs = 100)

model.save('The_Verdict.h5')

import pickle
with open('tokenizer.pickle', 'wb') as handle:
  pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
