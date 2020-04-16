import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import logging
import sys
import re
import spacy
import json
import numpy as np
import copy
import string
import contractions

from bert.tokenization import FullTokenizer
from keras.layers import Layer
from keras.models import load_model
from keras import backend as K
from globals import *
from spacy import displacy
from IPython.core.display import display, HTML
from copy import deepcopy

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 205
tag2idx = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, -1: 14, '-PAD-': 0}
n_tags = len(tag2idx)
shape2idx = json.loads(open("shape2idx.json","r").read())
shapeEmbeddings = np.identity(len(shape2idx), dtype='float32')
nlp = spacy.load("en_core_web_md")

# Initialize session
sess = tf.Session()
graph = tf.get_default_graph()

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, shapetags=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.shapetags=shapetags

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokens_a = tokenizer.tokenize(example.text_a)
    print(tokens_a)
    new_shapetags = copy.deepcopy(example.shapetags)

    for idx, t in enumerate(tokens_a):
        try:
            dummy = new_shapetags[idx]
        except IndexError as e:
            new_shapetags.insert(idx, new_shapetags[idx-1])
        if t[:2] == "##":
            new_shapetags.insert(idx, new_shapetags[idx-1])

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]
        new_shapetags = new_shapetags[0 : (max_seq_length - 2)]


    tokens = []
    shapetags = []
    segment_ids = []

    tokens.append("[CLS]")
    shapetags.append(shape2idx['-PAD-'])
    segment_ids.append(0)
    for i, token in enumerate(tokens_a):
        tokens.append(token)
        shapetags.append(new_shapetags[i])
        segment_ids.append(0)
    tokens.append("[SEP]")
    shapetags.append(shape2idx['-PAD-'])
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    while len(shapetags) < max_seq_length:
        shapetags.append(shape2idx["-PAD-"])

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(shapetags) == max_seq_length

    return input_ids, input_mask, segment_ids, shapetags

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, shapetags_arr = [], [], [], []
    for example in examples:
        input_id, input_mask, segment_id, shapetags = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        shapetags_arr.append(shapetags)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(shapetags_arr),
    )

def convert_text_to_examples(texts, shapetags_arr):
    """Create InputExamples"""
    InputExamples = []
    for text, shapetags in zip(texts, shapetags_arr):
        InputExamples.append(
            InputExample(guid=None, text_a=text, text_b=None, shapetags=shapetags)
        )
    return InputExamples

class BertLayer(Layer):
    def __init__(self, n_fine_tune_layers=10, mask_zero=False, trainable=True, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = trainable
        self.output_size = 768
        self.mask_zero=mask_zero
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print('Loading TF-Hub module...')
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        print('TF-Hub module loaded!')
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask,
                           segment_ids=segment_ids)
        result = self.bert(inputs=bert_inputs, signature="tokens",
                           as_dict=True)["sequence_output"]
        result = K.reshape(result, (-1,inputs[0].shape[1],768))
        return result

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return (None, input_shape[0][1], self.output_size)
      
    def compute_mask(self, inputs, mask=None):
      input_ids, input_mask, segment_ids = inputs
      if not self.mask_zero:
          return None
      return K.not_equal(input_ids, 0)

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

def create_shape_features(text):
    doc = nlp(text) 
    shape_tags = [token.shape_ for token in doc]
    return shape_tags

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from keras.models import Model, Input
from keras.layers import Embedding, Dense, Bidirectional, Dropout, LSTM, TimeDistributed, concatenate

# Build model
def build_model(max_seq_length):
    #Shape embeddings
    shape_input = Input(shape=(max_seq_length,), dtype='int32', name='shape_input')
    shape_embed = Embedding(output_dim=shapeEmbeddings.shape[1], input_dim=shapeEmbeddings.shape[0], weights=[shapeEmbeddings], trainable=False, name = 'shape_embed')(shape_input)

    # Bert Embeddings
    in_id = Input(shape=(max_seq_length,), name="input_ids")
    in_mask = Input(shape=(max_seq_length,), name="input_masks")
    in_segment = Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]
    bert_output = BertLayer(n_fine_tune_layers=3, mask_zero=True, trainable=False)(bert_inputs)
    
    output = concatenate([bert_output,shape_embed])

    lstm = Bidirectional(LSTM(units=128, return_sequences=True))(output)
    drop = Dropout(0.4)(lstm)
    dense = TimeDistributed(Dense(128, activation="relu"))(drop)
    crf = CRF(n_tags)
    out = crf(dense)
    model = Model(inputs=bert_inputs + [shape_input], outputs=out)
    model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    model.summary()
        
    return model
  
def initialize_vars(sess):
    K.get_session().run(tf.local_variables_initializer())
    K.get_session().run(tf.global_variables_initializer())
    K.get_session().run(tf.tables_initializer())

def load_model(filepath, max_seq_length):
    model = build_model(max_seq_length)
    model.load_weights(filepath)
    return model
    
def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

tokenizer = create_tokenizer_from_hub_module()
model = load_model("model.h5", max_seq_length)
logger.info("Model done!")

def predict(text):
    global graph

    text = text.strip()

    with graph.as_default():
        pre_shapetags = create_shape_features(text)
        pre_shapetags_ids = list(map(lambda t: shape2idx.get(t,0), pre_shapetags))
        examples = convert_text_to_examples([text], [pre_shapetags_ids])
        ids, masks, segments, shapetags = convert_examples_to_features(tokenizer, examples,max_seq_length=max_seq_length)
        predictions = model.predict([ids, masks, segments, shapetags])
        annotations = logits_to_tokens(predictions, {i: t for t, i in tag2idx.items()})
        print(annotations)
    return annotations

def preprocess(message):
    # Remove some punctuation
    filtered = []
    for c in message.lower():
        if not c in string.punctuation or c in '.$£@,-':
            filtered.append(c)
    return contractions.fix(''.join(filtered))


def align_tokenization_with_labels(tokens, annotations):
    packed_test_docs = []
    packed_test_labels = []
    print(tokens)
    print(annotations)
    for tid, t in enumerate([tokens]):
        tokens = []
        labels = []
        labels_idx = 0
        for i in t:
            ct = []
            for tok in tokenizer.tokenize(i):
                ct.append(tok.replace("##", ""))            
            if len(ct) != 0 and labels_idx+len(ct) <=205:
                tokens.append(''.join(ct))
                possible_labels = annotations[tid][labels_idx:labels_idx+len(ct)] 
                max_l = -1  
                for l in possible_labels:
                    if type(l)==int and l > max_l:
                        max_l = l
                labels.append(max_l)
                labels_idx  += len(ct)
        packed_test_labels.append(labels)
        packed_test_docs.append(tokens)

    return packed_test_docs, packed_test_labels

def generate_sens_viz_cont(doc, annotations):
    tag_in = {'text':'','ents':[], 'title':None}
    cursor = 0
    ent = {}
    current_annotation = None
    for i in range(len(doc)):
        tag_in['text'] += doc[i] + ' '
        if annotations[i] != current_annotation:
            if "start" in ent:
                ent["end"] = cursor
                tag_in['ents'].append(deepcopy(ent))
                ent = {}
                
        if annotations[i] in REVERSE_SENSITIVITY_TABLE:
            if "start" not in ent:
                current_annotation = annotations[i]
                ent["start"] = cursor
                ent["label"] = REVERSE_SENSITIVITY_TABLE[annotations[i]]['label']
            
        cursor += len(doc[i])+1
        
    if "start" in ent:
        ent["end"] = cursor
        tag_in['ents'].append(deepcopy(ent))

    return tag_in