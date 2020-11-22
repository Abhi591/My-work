import pandas as pd 
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os 

vocab_size=88584
max_len=250
batch_size=64

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=vocab_size)

#print(len(train_data[0]))   #looking at one review

"""as the data contains different length so we can fed the data to neural network 
so if length is more then 250 then we trim if less than 250 then we add 0's to left side of it to make 
it equal 
keras sequence does this for us :)"""

train_data=sequence.pad_sequences(train_data,max_len)
test_data=sequence.pad_sequences(test_data,max_len)
#print(len(train_data[0]))

#creating the model
model=tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size,32),      
	tf.keras.layers.LSTM(32),
	tf.keras.layers.Dense(1,activation='sigmoid')])
#model.summary()

#training the model
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
history=model.fit(train_data,train_labels,epochs=10,validation_split=0.2)
result=model.evaluate(test_data,test_labels)
print(result)

"""since our reviews are encoded we'll need to convert any review that we write into that from so the network
can understand it .To do that we'll load the encoding from the dataset and use them to encode our own data """

word_index=imdb.get_word_index()

def encode_text(text):
	tokens=tf.keras.preprocessing.text.text_to_word_sequence(text)
	tokens=[word_index[word] if word in word_index else 0 for word in tokens]
	return sequence.pad_sequences([tokens],max_len)[0]

text="that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

#lets make a decode function
reverse_word_index={value:key for (key,value) in word_index.items()}
def decode_integers(integers):
	pad=0
	text=""
	for num in integers:
		if num!=pad:
			text+=reverse_word_index[num]+ " "
	return text[:-1]

print(decode_integers(encoded))

def predict(text):
	encoded_text=encode_text(text)
	pred=np.zeros((1,250))
	pred[0]=encoded_text
	result=model.predict(pred)
	print(result[0])

positive_review="that movie was so awesome , I really loved it and would watch it again \
cause it was amazingly great"
predict(positive_review)

negative_review="That movie was so bad . I would not watch it again , i didn't liked the movie , \
it was really bad"
predict(negative_review)

#output greater than 0.5 is a positive review and less than 0.5 is negative review



