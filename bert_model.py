import os
import math
import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class Label_Detection_Data:
    data_column = 'text'
    label_column = 'label'

    def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=64): # 192
        self.tokenizer = tokenizer
        self.max_seq_len = 0
        self.classes = classes

        train, test = map(lambda df: df.reindex(df[Label_Detection_Data.data_column].str.len().sort_values().index), [train, test])

        ((self.x_train, self.y_train), (self.x_test, self.y_test)) = map(self._prepare, [train, test])

        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        self.x_train, self.x_test = map(self._pad, [self.x_train, self.x_test])

    def _prepare(self, df):
        x, y = [], []
        for _, row in tqdm(df.iterrows()):
            text, label = row[Label_Detection_Data.data_column], row[Label_Detection_Data.label_column]
            tokens = self.tokenizer.tokenize(text)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_len = max(self.max_seq_len, len(token_ids))
            x.append(token_ids)
            y.append(self.classes.index(label))

        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))

        return np.array(x)


def create_model(max_seq_len, bert_checkpoint_file):
    with tf.io.gfile.GFile(bert_config_file, 'r') as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name='bert')

    input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name='input_ids')
    bert_output = bert(input_ids)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation='tanh')(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(classes), activation='softmax')(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_checkpoint_file)

    return model

def remove_datapoints(x_train, y_train, x_test, y_test):
    percentage = 0.75

    def slash_datapoints(*args):
        labels, counts = np.unique(args[1], return_counts=True)
        bloodborne = []
        for label in labels:
            indices = np.array(np.where(args[1] == label)).tolist()[0]
            num_indices = round(len(indices) * percentage)
            dead_indices = np.random.choice(indices, num_indices, replace=False)
            bloodborne.append(dead_indices)

        bloodborne = np.concatenate(bloodborne).ravel()
        temp0 = np.delete(args[0], bloodborne, axis=0)
        temp1 = np.delete(args[1], bloodborne, axis=0)

        return temp0, temp1

    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)

    x_train, y_train = slash_datapoints(x_train, y_train)
    x_test, y_test = slash_datapoints(x_test, y_test)

    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_test.shape)
    #print(y_test.shape)

    return x_train, y_train, x_test, y_test

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.drop(['id'], 1)
test = test.drop(['id'], 1)

bert_checkpoint_file = 'bert_model.ckpt'
bert_config_file = 'bert_config.json'

tokenizer = FullTokenizer(vocab_file='vocab.txt')

classes = train.label.unique().tolist()

data = Label_Detection_Data(train, test, tokenizer, classes, max_seq_len=128)

data.x_train, data.y_train, data.x_test, data.y_test = remove_datapoints(data.x_train, data.y_train, data.x_test, data.y_test)

#print(data.max_seq_len)

print(data.x_train.shape)
print(data.x_test.shape)

print(data.y_train.shape)
print(data.y_test.shape)

model = create_model(data.max_seq_len, bert_checkpoint_file)

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

history = model.fit(data.x_train, data.y_train, validation_split=0.2, batch_size=16, epochs=5)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
#plt.show()
plt.savefig('bert_model-accuracy.png')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
#plt.show()
plt.savefig('bert_model-loss.png')

_, accuracy = model.evaluate(data.x_test, data.y_test)
print('Accuracy: ', accuracy)

with open('bert_model_results.txt', 'a') as file:
    file.write('{}'.format(accuracy * 100))
