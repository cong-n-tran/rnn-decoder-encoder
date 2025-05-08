# Khanh Nguyen Cong Tran
# 1002046419

import tensorflow as tf
import keras
from keras import layers, models
import random
import np


def train_enc_dec(train_sentences, validation_sentences, epochs):
    # create the input and output data
    train_inputs, train_targets = create_input_output(train_sentences)
    val_inputs, val_targets = create_input_output(validation_sentences)

    # create the vectorization layers
    source_vec_layer = make_vectorization_layer(train_inputs)
    target_vec_layer = make_vectorization_layer(["startseq " + s + " endseq" for s in train_targets])

    # vectorized the data and return the formatted dataset
    train_source, train_target_in, train_target_out = format_dataset(train_inputs, train_targets, source_vec_layer, target_vec_layer)
    val_source, val_target_in, val_target_out = format_dataset(val_inputs, val_targets, source_vec_layer, target_vec_layer)

    # build the model
    model = build_encoder_decoder_model(
        source_vocab_size=len(source_vec_layer.get_vocabulary()),
        target_vocab_size=len(target_vec_layer.get_vocabulary())
    )

    # train the model
    model.fit(
        [train_source, train_target_in],
        tf.expand_dims(train_target_out, -1),
        validation_data=([val_source, val_target_in], tf.expand_dims(val_target_out, -1)),
        epochs=epochs,
        batch_size=32,
        verbose=2
    )

    return model, source_vec_layer, target_vec_layer

# create the vectorization layer
def make_vectorization_layer(texts, vocab_size=250, max_len=10):
    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_len,
        standardize='lower_and_strip_punctuation',
    )
    text_ds = tf.data.Dataset.from_tensor_slices(texts).batch(32)
    vectorize_layer.adapt(text_ds)
    return vectorize_layer

# apply the vectorization and add the [start] and [end]
def format_dataset(source_texts, target_texts, source_vectorizer, target_vectorizer):
    source = source_vectorizer(source_texts)
    target_in = target_vectorizer(["startseq " + t for t in target_texts])
    target_out = target_vectorizer([t + " endseq" for t in target_texts])
    return (source, target_in, target_out)

# build the model
def build_encoder_decoder_model(source_vocab_size, target_vocab_size, embedding_dim=64, latent_dim=64):
    # encoder
    encoder_inputs = layers.Input(shape=(None,), name='encoder_input')
    x = layers.Embedding(source_vocab_size, embedding_dim)(encoder_inputs)
    encoder_outputs, state_h, state_c = layers.LSTM(latent_dim, return_state=True)(x)
    encoder_states = [state_h, state_c]

    # decoder
    decoder_inputs = layers.Input(shape=(None,), name='decoder_input')
    x = layers.Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
    decoder_dense = layers.Dense(target_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # model
    model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 # Create inputs/targets
    
def create_input_output(sentences):
    input_texts, target_texts = [], []
    for sentence in sentences:
        words = sentence.strip().split()
        if random.random() < 0.5:
            input_text = ' '.join(reversed(words))
        else:
            input_text = ' '.join(words)
        target_text = ' '.join(words)
        input_texts.append(input_text)
        target_texts.append(target_text)
    return input_texts, target_texts



def get_enc_dec_results(model, test_sentences, source_vectorizer, target_vectorizer, max_decoded_length=50):
    # vectorize input
    source_sequences = source_vectorizer(test_sentences)
    
    # get the batch size
    batch_size = tf.shape(source_sequences)[0]

    # get start and end tokens
    vocab = target_vectorizer.get_vocabulary()
    start_token = vocab.index('startseq') # for some reason when i switch to "START" it can't be found???
    end_token = vocab.index('endseq')

    # start the rnn sequence with the start token
    decoded_sequences = tf.fill([batch_size, 1], tf.constant(start_token, dtype=tf.int64))

    # feed start token into rnn and reuse output as input until we get end token or max iterations
    for _ in range(max_decoded_length):
        
        #predict the output
        predictions = model([source_sequences, decoded_sequences])

        # get last time step's prediction
        next_token_logits = predictions[:, -1, :]  # (batch_size, vocab_size)
        next_tokens = tf.argmax(next_token_logits, axis=-1, output_type=tf.int64)  # (batch_size,)
        next_tokens = tf.expand_dims(next_tokens, axis=-1)  # (batch_size, 1)

        # concat last time step to sequence
        decoded_sequences = tf.concat([decoded_sequences, next_tokens], axis=-1)

        # check if we reached end token
        end_token_tensor = tf.constant(end_token, dtype=tf.int64)
        finished = tf.reduce_all(tf.reduce_any(decoded_sequences == end_token_tensor, axis=1))
        if finished:
            break

    # detokenized the entire sequences 
    detokenized = []
    for seq in decoded_sequences.numpy():
        words = []
        for idx in seq[1:]:  # skip start token
            word = vocab[idx]

            #once we reach teh end seq, we know we are done. 
            if word == 'endseq':
                break
            words.append(word)
        detokenized.append(' '.join(words))

    # return the entire detokenized and predicted sentences. 
    return detokenized






#best model where you can do whatever the tuck you want
# 99.40% 
# 99.15%
# 99.13%
# 99.20%
# 99.13%
# 99.20%
# 99.15%
# 99.14%
# 99.15%
# 99.09%


# positional embedding
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
        )
        self.pos_emb = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.seq_len = sequence_length

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        return self.token_emb(x) + self.pos_emb(positions)

# transformer encoder layer
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout_rate)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop2 = layers.Dropout(dropout_rate)

    def call(self, x, training):
        attn_out = self.self_attn(x, x)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop2(ffn_out, training=training))

# transformer decoder layer
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout_rate)
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop2 = layers.Dropout(dropout_rate)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.drop3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_outputs, training):
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]
        attn1 = self.self_attn(query=x, value=x, key=x, attention_mask=causal_mask)
        x = self.norm1(x + self.drop1(attn1, training=training))
        attn2 = self.cross_attn(x, enc_outputs, enc_outputs)
        x = self.norm2(x + self.drop2(attn2, training=training))
        ffn_out = self.ffn(x)
        return self.norm3(x + self.drop3(ffn_out, training=training))


def train_best_model(train_sentences, validation_sentences):
    # required parameters
    vocab_size = 250
    seq_len    = 8

    # my hyperparameters
    embed_dim  = 64
    ff_dim     = 64
    num_heads  = 4
    batch_size = 64
    epochs     = 75 # 20-50 epoch gives me 98% while 75+ gives me 99%

    vec = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_len,
    )
    vec.adapt(train_sentences + validation_sentences)
    vocab = vec.get_vocabulary()
    tv_size = len(vocab)
    start_tok = tv_size
    end_tok   = tv_size + 1
    full_vocab = tv_size + 2

    # build data ???
    def build_data(sent_list):
        N = len(sent_list)
        sources, dec_in, dec_out = [], [], []
        for s in sent_list:
            rev = " ".join(reversed(s.split()))
            for src in (s, rev):
                sources.append(src)
                dec_in.append(s)
                dec_out.append(s)
        S = vec(np.array(sources)).numpy()
        T = vec(np.array(dec_in)).numpy()
        M = vec(np.array(dec_out)).numpy()
        Din = np.zeros((2*N, seq_len+1), dtype="int32")
        Dout= np.zeros((2*N, seq_len+1), dtype="int32")
        for i in range(2*N):
            Din[i,0]      = start_tok
            Din[i,1:]     = T[i]
            Dout[i,:seq_len] = M[i]
            Dout[i,seq_len]   = end_tok
        return S, Din, Dout

    train_S, train_Din, train_Dout = build_data(train_sentences)
    val_S,   val_Din,   val_Dout   = build_data(validation_sentences)

    # using encoder decoder transformer
    enc_inputs = keras.Input(shape=(seq_len,), dtype="int32")
    x = PositionalEmbedding(seq_len, full_vocab, embed_dim)(enc_inputs)
    for _ in range(2):
        x = TransformerEncoder(embed_dim, ff_dim, num_heads)(x, training=True)
    enc_outputs = x

    # decoder
    dec_inputs = keras.Input(shape=(seq_len+1,), dtype="int32")
    y = PositionalEmbedding(seq_len+1, full_vocab, embed_dim)(dec_inputs)
    for _ in range(2):
        y = TransformerDecoder(embed_dim, ff_dim, num_heads)(y, enc_outputs, training=True)
    outputs = layers.Dense(full_vocab, activation="softmax")(y)

    # the regular complie and train model shit
    model = keras.Model([enc_inputs, dec_inputs], outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        [train_S, train_Din],
        train_Dout[..., np.newaxis],
        validation_data=([val_S, val_Din], val_Dout[..., np.newaxis]),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
    )

    return model, vec, vec


def get_best_model_results(model, test_sentences, source_vec_layer, target_vec_layer):
    seq_len = 8
    vocab = target_vec_layer.get_vocabulary()
    tv_size = len(vocab)
    start_tok = tv_size
    end_tok = tv_size + 1

    # vectorize the input
    S = source_vec_layer(np.array(test_sentences))  # (N, seq_len)
    N = S.shape[0]

    # initalize using start token
    dec_input = np.zeros((N, seq_len+1), dtype="int32")
    dec_input[:, 0] = start_tok

    finished = np.zeros(N, dtype=bool)

    # man idek at this point
    for t in range(seq_len):
        preds = model.predict([S, dec_input], verbose=0)  # (N, seq_len+1, V)
        next_tokens = np.argmax(preds[:, t, :], axis=-1)
        dec_input[:, t+1] = next_tokens
        # mark finished sequences
        finished |= (next_tokens == end_tok)
        if finished.all():
            break

    # remove all special tokens and get rsults
    results = []
    for i in range(N):
        # take decoded IDs (excluding the start token)
        token_ids = dec_input[i, 1:]
        # filter out padding and end token
        token_ids = [int(t) for t in token_ids if t != 0 and t != end_tok]
        words = [vocab[t] for t in token_ids]
        results.append(" ".join(words))

    return results