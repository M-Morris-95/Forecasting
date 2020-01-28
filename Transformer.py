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
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

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
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concat
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Linear
        outputs = self.output_dense(concat_attention)
        return outputs

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(units, d_model), name="inputs")

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
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    # dropout
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)

    # add and norm
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    # assemble layer
    layer = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return layer

def encoder(num_layers, units, d_model, num_heads, dropout, name="encoder"):
    # create input
    inputs = tf.keras.Input(shape=(units,d_model), name="inputs")

    # no embeddings but if there were put them here.

    # add dropout
    outputs = tf.keras.layers.Dropout(rate=dropout)(inputs)

    # create layers
    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )(outputs)

    # assemble model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model

def encoder_network(output_size, num_layers, units, d_model, num_heads, dropout, name="transformer"):
    # inputs
    inputs = tf.keras.Input(shape=(units,d_model), name="inputs")

    # encoder
    enc_outputs = encoder(
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs])

    outputs = tf.keras.layers.Dense(units=output_size, activation = 'relu')(enc_outputs)
    # output dense layer
    outputs = Flatten()(outputs)
    outputs = tf.keras.layers.Dense(units=output_size, name="outputs")(outputs)
    # build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model

def transformer_network(output_size, num_layers, units, d_model, num_heads, dropout, embedding_size = 500, name="transformer"):
    # inputs
    inputs = tf.keras.Input(shape=(units,d_model), name="inputs")
    flatten = tf.keras.layers.Flatten()(inputs)
    embedding_layer = tf.keras.layers.Dense(units=embedding_size, name='embedding_layer')(flatten)

    # encoder
    enc_outputs = encoder(
        num_layers=1,
        units=1,
        d_model=flatten.shape[1],
        num_heads=7,
        dropout=dropout,
    )(inputs=[flatten])

    outputs = tf.keras.layers.Dense(units=output_size, activation = 'relu')(enc_outputs)
    # output dense layer
    outputs = Flatten()(outputs)
    outputs = tf.keras.layers.Dense(units=output_size, name="outputs")(outputs)
    # build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model