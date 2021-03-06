from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dropout, Conv1D, GRU, Attention, Dense, Input, concatenate, Flatten, Layer, LayerNormalization, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask=None):
    # Q K MatmMul
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # Q K Scale by sqrt dk
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(dk)

    # Do mask if included. add the mask zero out padding tokens.
    if mask is not None:
        logits += (mask * -1e9)

    # softmax Q K
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # matmul Q K with V
    Attention = tf.matmul(attention_weights, value)

    return Attention

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention", regularizer = None):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model, activation='linear', kernel_regularizer=regularizer)
        self.key_dense = tf.keras.layers.Dense(units=d_model, activation='linear', kernel_regularizer=regularizer)
        self.value_dense = tf.keras.layers.Dense(units=d_model, activation='linear', kernel_regularizer=regularizer)
        self.output_dense = tf.keras.layers.Dense(units=d_model, activation='linear', kernel_regularizer=regularizer)

    def split_heads(self, inputs, batch_size):
        reshape = tf.reshape(inputs, shape=(batch_size, self.num_heads, int(inputs.shape[-1]/self.num_heads)))
        return reshape

    def __call__(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot product attention
        scaled_attention = scaled_dot_product_attention(query, key, value)

        # Concat
        concat_attention = tf.reshape(scaled_attention, (batch_size, self.d_model))

        # Linear
        outputs = self.output_dense(concat_attention)
        return outputs

def encoder_layer(d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=d_model, name="inputs")

    # multi head attention
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs
    })
    # dropout
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)

    # add and norm
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    # feed forward
    outputs = tf.keras.layers.Dense(units=d_model, activation='relu')(attention)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)

    # add and norm
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    # assemble layer
    layer = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return layer

def encoder(num_layers, d_model, num_heads, dropout, units = 1, embedding_size=500, name="encoder"):
    # create input
    inputs = tf.keras.Input(shape=(units, d_model), name="inputs")
    flatten = tf.keras.layers.Flatten()(inputs)

    # embedding
    embedding_enc = tf.keras.layers.Dense(units=embedding_size, name='embedding_layer')(flatten)

    # create layers
    for i in range(num_layers):
        outputs = encoder_layer(
            d_model=embedding_size,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )(embedding_enc)

    # assemble model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model

def decoder(num_layers, d_model, num_heads, dropout, embedding_size = 500, name="decoder"):
    # create input
    inputs = tf.keras.Input(shape=(d_model), name="inputs")
    prev_outputs = tf.keras.Input(shape=(1), name="prev_outputs")

    # no embeddings but if there were put them here.
    embedding_dec = tf.keras.layers.Dense(units=embedding_size, name='embedding_layer')(prev_outputs)

    # create layers
    for i in range(num_layers):
        outputs = decoder_layer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="decoder_layer_{}".format(i),
        )([inputs, embedding_dec])

    # assemble model
    model = tf.keras.Model(inputs=[inputs, prev_outputs], outputs=outputs, name=name)

    return model

def decoder_layer(d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=d_model, name="inputs")
    prev_outputs = tf.keras.Input(shape=d_model, name="prev_outputs")

    # multi head attention
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs
    })
    # dropout
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    # add and norm
    add_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    attention2 = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': add_norm,
        'key': add_norm,
        'value': prev_outputs
    })
    # dropout
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)

    # add and norm
    add_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(add_norm + attention2)

    # feed forward
    outputs = tf.keras.layers.Dense(units=d_model, activation='relu')(add_norm2)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)

    # add and norm
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    # assemble layer
    layer = tf.keras.Model(inputs=[inputs,prev_outputs], outputs=outputs, name=name)
    return layer

def transformer_network(output_size, units, num_layers, d_model, num_heads, dropout, embedding_size = 500, name="transformer"):
    # inputs
    inputs = tf.keras.Input(shape=(units, d_model), name="inputs")
    prev_outputs = tf.keras.Input(shape=(1), name="prev_outputs")

    # encoder
    enc_outputs = encoder(
        num_layers=num_layers,
        d_model=d_model,
        units = units,
        embedding_size = embedding_size,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs])

    dec_outputs = decoder(
        num_layers=num_layers,
        d_model=embedding_size,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[enc_outputs, prev_outputs])

    outputs = tf.keras.layers.Dense(units=output_size, activation = 'relu')(dec_outputs)

    # build model
    model = tf.keras.Model(inputs=[inputs, prev_outputs], outputs=outputs, name=name)

    return model

def modified_encoder(output_size, units, num_layers, d_model, num_heads, dropout, embedding_size = 500, name="transformer"):
    # inputs
    inputs = tf.keras.Input(shape=(units, d_model), name="inputs")

    # encoder
    enc_outputs = encoder(
        num_layers=num_layers,
        d_model=d_model,
        units = units,
        embedding_size = embedding_size,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs])

    outputs = tf.keras.layers.Dense(units=output_size, activation = 'relu')(enc_outputs)

    # build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model