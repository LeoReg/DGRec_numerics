import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

import tensorflow.keras.backend as K

from tensorflow.keras.utils import register_keras_serializable

EPS = 1e-8

# Define your custom metric class again (same as before)
class MaskedAccuracy(keras.metrics.Metric):
    def __init__(self, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        mask = tf.cast(tf.reduce_any(tf.not_equal(y_true, 0.0), axis=-1), tf.float32)
        matches = tf.cast(tf.equal(y_true_labels, y_pred_labels), tf.float32)
        masked_matches = matches * mask
        self.total.assign_add(tf.reduce_sum(masked_matches))
        self.count.assign_add(tf.reduce_sum(mask))
    
    def result(self):
        return self.total / tf.maximum(self.count, 1e-8)
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
    
# One-hot encoding function for DNA sequences (A, C, G, T)
def one_hot_encode(sequence, vocab_size=4):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    integer_encoded = [mapping[base] for base in sequence]
    onehot_encoded = to_categorical(integer_encoded, num_classes=vocab_size)
    return onehot_encoded

def one_hot_decode(encoded_sequence):
    # Reverse the mapping used in encoding
    reverse_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    # Get the index of the max value in each one-hot vector (which corresponds to the base index)
    integer_decoded = np.argmax(encoded_sequence, axis=1)
    
    # Convert the integer sequence back to the DNA bases using the reverse mapping
    decoded_sequence = ''.join([reverse_mapping[i] for i in integer_decoded])
    
    return decoded_sequence

def separate_model(model):
    """
    Disentangle the two-stage masked LSTM model into two parts:
      1. First BiLSTM + TimeDistributed Dense layers
      2. Second LSTM + Final Dense (with initial states for generation)
    """
    lstm_units = 16
    vocab_size = 4  # A, C, G, T

    # --- Part 1: First BiLSTM + Dense(s) ---
    inputs = layers.Input(shape=(None, vocab_size+1), name="DNA_input_sep")

    masked_seq = layers.Masking(mask_value=0.0)(inputs)
    bilstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units//2, return_sequences=True),
        name="bilstm_1_sep"
    )(masked_seq)

    inter_dense_out = layers.TimeDistributed(
        layers.Dense(vocab_size+2, activation="softmax"),
        name="inter_dense_sep"
    )(bilstm_out)

    inter_dense_relu_out = layers.TimeDistributed(
        layers.Dense(2, activation="relu"),
        name="inter_dense_relu_sep"
    )(bilstm_out)

    # First part outputs both intermediate representations
    first_part_model = models.Model(
        inputs=inputs,
        outputs=[inter_dense_out, inter_dense_relu_out]
    )

    # Transfer weights by name
    first_part_model.get_layer("bilstm_1_sep").set_weights(
        model.get_layer("bilstm_1").get_weights()
    )
    first_part_model.get_layer("inter_dense_sep").set_weights(
        model.get_layer("intermediate_dense").get_weights()
    )
    first_part_model.get_layer("inter_dense_relu_sep").set_weights(
        model.get_layer("intermediate_dense_relu").get_weights()
    )

    # --- Part 2: Second LSTM + Final Dense ---
    inputs_seq = layers.Input(shape=(None, vocab_size+1), name="DNA_input2_sep")
    inputs_mut = layers.Input(shape=(None, 2 * vocab_size + 1), name="mutation_input2_sep")
    inter_out = layers.Input(shape=(None, vocab_size+2), name="inter_dense_out_sep")
    inter_relu_out = layers.Input(shape=(None, 2), name="inter_dense_relu_out_sep")
    initial_h = layers.Input(shape=(lstm_units,), name="init_h")
    initial_c = layers.Input(shape=(lstm_units,), name="init_c")

    concat_inputs = layers.Concatenate(name="concat_sep")(
        [inputs_seq, inputs_mut, inter_out, inter_relu_out]
    )
    masked_concat = layers.Masking(mask_value=0.0)(concat_inputs)

    lstm_out, state_h, state_c = layers.LSTM(
        lstm_units, return_sequences=True, return_state=True, name="lstm_2_sep"
    )(masked_concat, initial_state=[initial_h, initial_c])

    final_out = layers.TimeDistributed(
        layers.Dense(vocab_size, activation="softmax"),
        name="final_dense_sep"
    )(lstm_out)

    second_part_model = models.Model(
        inputs=[inputs_seq, inputs_mut, inter_out, inter_relu_out, initial_h, initial_c],
        outputs=[final_out, state_h, state_c]
    )

    # Transfer weights
    second_part_model.get_layer("lstm_2_sep").set_weights(
        model.get_layer("lstm_2").get_weights()
    )
    second_part_model.get_layer("final_dense_sep").set_weights(
        model.get_layer("final_dense").get_weights()
    )

    return first_part_model, second_part_model
    
model_TR_to_VR = load_model("LSTM_model_8_16.keras",
    custom_objects={'MaskedAccuracy': MaskedAccuracy}) #the LSTM model
firstmodel,secondmodel=separate_model(model_TR_to_VR)

def generate_sequence_from_onehot(X,firstmodel= firstmodel, secondmodel=secondmodel):
    """
    Generate a sequence using separated masked LSTM models.
    
    X: one-hot encoded sequence (batch, time, vocab_size+2)  # input_seq
    firstmodel: first part model (BiLSTM -> [softmax dense, ReLU dense])
    secondmodel: second part model (Concat -> LSTM -> final dense)
    """
    size, seq_length, input_dim = X.shape
    vocab_size = 4   # A, C, G, T
    lstm_units = 16

    # Initial states for second LSTM
    initial_h = np.zeros((size, lstm_units))
    initial_c = np.zeros((size, lstm_units))

    # First part (BiLSTM + two dense layers)
    inter_softmax_out, inter_relu_out = firstmodel.predict(X)

    # Mutation input (start token)
    X_mut = np.zeros((size, 1, vocab_size * 2 + 1))
    X_mut[:, 0, -1] = 1  # mark start

    generated_sequence = np.zeros((size, seq_length, vocab_size))

    for t in tqdm(range(seq_length), desc="Generating sequence"):
        output, new_h, new_c = secondmodel.predict([
            X[:, t:t+1, :],                       # input_seq at step t
            X_mut,                                # mutation input
            inter_softmax_out[:, t:t+1, :],       # intermediate softmax output
            inter_relu_out[:, t:t+1, :],          # intermediate relu output
            initial_h, initial_c                  # LSTM states
        ],verbose=0)

        # Sample next nucleotide
        probs = output[:, 0, :]  # (batch, vocab_size)
        next_word_idx = [np.random.choice(vocab_size, p=p) for p in probs]
        next_word_one_hot = np.zeros((size, vocab_size))
        next_word_one_hot[np.arange(size), next_word_idx] = 1

        generated_sequence[:, t, :] = next_word_one_hot

        # Update states
        initial_h, initial_c = new_h, new_c

        # Update mutation input for next step
        if t < seq_length - 1:
            X_mut[:, 0, :vocab_size] = X[:, t, :vocab_size]        # original nucleotide
            X_mut[:, 0, vocab_size:2*vocab_size] = next_word_one_hot  # predicted nucleotide
            X_mut[:, 0, -1] = 0  # remove start flag after t=0

    return generated_sequence


def sequences_same_length(sequences):
    """
    Check if all sequences in a list have the same length.
    
    Parameters:
    - sequences: list of sequences (e.g., strings, lists, or arrays)
    
    Returns:
    - bool: True if all sequences have the same length, False otherwise.
    """
    # Get the length of the first sequence
    first_length = len(sequences[0])

    # Check if all sequences have the same length
    for seq in sequences:
        if len(seq) != first_length:
            return False

    return True


def pad_sequence(seq, maxlen, padding_value=0):
    pad_width = ((0, maxlen - seq.shape[0]), (0, 0))  # pad rows only
    return np.pad(seq, pad_width, mode='constant', constant_values=padding_value)


def generate_sequences(X_seq): #X _seq is a list of sequences ATCG sequences (faster if same length)
    """
    Generate list of VR from list of TR (one TR-> one VR).
    Parameters:
    - X_seq: list of TR sequences (e.g., strings, lists, or arrays)
    Returns:
    - list: list of VR sequence strings given TR sequences (corresponding to the initial TR list).
    """
    X=[np.concatenate((one_hot_encode((X_seq[k])[::-1]),[[i] for i in range(len(X_seq[k]))]),axis=1) for k in range(len(X_seq))]
    if sequences_same_length(X_seq): # check that all sequences have same length: allow for fast generation
        Y=generate_sequence_from_onehot(np.array(X),firstmodel,secondmodel)
        return [one_hot_decode(y)[::-1] for y in Y] # return sequence in good order
    else:
        max_len = max(len(seq)+2 for seq in X_seq)
        length=np.array([len(seq) for seq in X_seq])
        X_padded = np.array([pad_sequence(seq, max_len) for seq in X])
        Y=generate_sequence_from_onehot(X_padded,firstmodel,secondmodel)
        return [(one_hot_decode(Y[i])[:length[i]])[::-1] for i in range(len(Y))] # return sequence in good order

def generate_sequences_oneTR(TR,n=1000):
    """
    Generate list of VR from one TR (one TR-> n VR).
    Parameters:
    - TR: one TR sequence (e.g., strings, lists, or arrays)
    -n: integer corresponding to the number of VR to generate
    Returns:
    - list: list of n VR sequence strings given the one TR sequence.
    """
    TR_list=[TR]*n
    return generate_sequences(TR_list)
    
def to_tensor_inputs(*args):
    return [tf.convert_to_tensor(x) for x in args]
    
def compute_likelihood(TR, VR, firstmodel= firstmodel, secondmodel=secondmodel):
    """
    Compute log-likelihood of VR from TR.

    TR, VR: same-length strings
    Returns: log-likelihood
    """
    assert len(TR) == len(VR), "Mismatched lengths"

    vocab_size = 4
    lstm_units = 16
    seq_length = len(TR)

    # Encode and reverse sequences
    X = np.array([np.concatenate((one_hot_encode((TR)[::-1]),[[i] for i in range(len(TR))]),axis=1)])
    Y = np.array([one_hot_encode(VR[::-1])])

    # First model output
    inter_softmax_out, inter_relu_out = firstmodel.predict(X, verbose=0)

    # Build mutation input (start with all start tokens)
    X_mut = np.zeros((1, seq_length, 2 * vocab_size + 1))
    X_mut[:, 0, -1] = 1  # Start tokens

    # Fill mutation input from known inputs
    for t in range(1, seq_length):
        X_mut[:, t, :vocab_size] = X[:, t - 1, :vocab_size]
        X_mut[:, t, vocab_size:2*vocab_size] = Y[:, t - 1, :]

    # Initial LSTM states
    initial_h = np.zeros((1, lstm_units))
    initial_c = np.zeros((1, lstm_units))

    # Run full sequence in one call
    outputs, _, _ = secondmodel.predict([X, X_mut, inter_softmax_out, inter_relu_out, initial_h, initial_c], verbose=0)

    # Compute log-likelihoods
    log_likelihoods = []
    for b in range(1):
        log_prob = 0.
        for t in range(seq_length):
            prob = outputs[b, t, np.argmax(Y[b, t])]
            log_prob += np.log(prob + 1e-8)
        log_likelihoods.append(log_prob)

    return log_likelihoods[0]

def compute_likelihood_batch(TR_batch, VR_batch, firstmodel= firstmodel, secondmodel=secondmodel):
    """
    Compute log-likelihoods of generating each VR[i] from TR[i] (batched).

    TR_batch, VR_batch: list of same-length strings
    Returns: list of log-likelihoods
    """
    assert all(len(tr) == len(vr) for tr, vr in zip(TR_batch, VR_batch)), "Mismatched lengths"

    vocab_size = 4
    lstm_units = 16
    batch_size = len(TR_batch)
    seq_length = len(TR_batch[0])

    # Encode and reverse sequences
    X = np.array([np.concatenate((one_hot_encode((TR_batch[k])[::-1]),[[i] for i in range(len(TR_batch[k]))]),axis=1) for k in range(len(TR_batch))])
    Y = np.array([one_hot_encode(vr[::-1]) for vr in VR_batch])

    # First model output
    inter_softmax_out, inter_relu_out = firstmodel.predict(X, verbose=0)

    # Build mutation input (start with all start tokens)
    X_mut = np.zeros((batch_size, seq_length, 2 * vocab_size + 1))
    X_mut[:, 0, -1] = 1  # Start tokens

    # Fill mutation input from known inputs
    for t in range(1, seq_length):
        X_mut[:, t, :vocab_size] = X[:, t - 1, :vocab_size]
        X_mut[:, t, vocab_size:2*vocab_size] = Y[:, t - 1, :]

    # Initial LSTM states
    initial_h = np.zeros((batch_size, lstm_units))
    initial_c = np.zeros((batch_size, lstm_units))

    # Run full sequence in one call
    outputs, _, _ = secondmodel.predict([X, X_mut, inter_softmax_out, inter_relu_out, initial_h, initial_c], verbose=0)

    # Compute log-likelihoods
    log_likelihoods = []
    for b in range(batch_size):
        log_prob = 0.
        for t in range(seq_length):
            prob = outputs[b, t, np.argmax(Y[b, t])]
            log_prob += np.log(prob + 1e-8)
        log_likelihoods.append(log_prob)

    return log_likelihoods

def compute_likelihood_matrix(TR_list, VR_list, firstmodel= firstmodel, secondmodel=secondmodel, batch_size=64):
    """
    Compute log-likelihood matrix of generating each VR from each TR using batching.
    """
    result_matrix = []

    for i in range(0, len(TR_list)):
        row = []
        TR = TR_list[i]
        valid_VRs = [vr for vr in VR_list if len(vr) == len(TR)]

        if not valid_VRs:
            row = [-np.inf] * len(VR_list)
        else:
            batch_TRs = [TR] * len(valid_VRs)
            log_likelihoods = compute_likelihood_batch(batch_TRs, valid_VRs, firstmodel, secondmodel)
            idx = 0
            for vr in VR_list:
                if len(vr) != len(TR):
                    row.append(-np.inf)
                else:
                    row.append(log_likelihoods[idx])
                    idx += 1

        result_matrix.append(row)
    return result_matrix
