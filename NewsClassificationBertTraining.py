import pandas as pd
import tensorflow_hub as hub
import tokenization
module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)
df1 = pd.read_csv('data.csv', usecols = ['text', 'category'])
df2 = pd.read_csv('transp.csv', usecols = ['text', 'category'])
data = pd.concat([df1, df2])
data = data.replace('transp', 13)


data = data.dropna(how='any',axis=0)


from bert import bert_tokenization
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=100):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=100):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(14, activation='softmax')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2,shuffle = True)


import keras
import tensorflow as tf
import numpy as np
max_len = 100
train_input = bert_encode(train.text.values, tokenizer, max_len=max_len)
test_input = bert_encode(test.text.values, tokenizer, max_len=max_len)
train_labels = tf.keras.utils.to_categorical(train.category.values, num_classes=14)


model = build_model(bert_layer, max_len=max_len)
model.summary()



train_history = model.fit(
    train_input, train_labels, 
    validation_split=0.5,
    epochs=3,
    batch_size=16)

model.save_weights("E:/Bertmodel/FineTunedBert_News.h5")

test_pred = model.predict(test_input)
computed_predictions = np.argmax(test_pred, axis=1)
from sklearn.metrics import classification_report, confusion_matrix
y = test.category.values
print(confusion_matrix(y, computed_predictions))

print(classification_report(y,computed_predictions))