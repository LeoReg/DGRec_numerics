import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import tensorflow.keras.backend as K
import pandas as pd
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import metrics
import random
import sys
from tensorflow.keras.utils import to_categorical
from collections import defaultdict, Counter
import ast
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task-id", type=int, default=1)  # Récupère ${SLURM_ARRAY_TASK_ID}
args = parser.parse_args()


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

def one_hot_decode_proba(encoded_sequence):
    # Reverse the mapping used in encoding
    reverse_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    # Initialize an empty list to store the decoded sequence
    decoded_sequence = []

    # Loop through each one-hot encoded vector
    for one_hot_vector in encoded_sequence:
        # Get the probabilities by normalizing the one-hot vector
        probabilities = one_hot_vector / np.sum(one_hot_vector)

        # Sample an index according to the probabilities
        sampled_index = np.random.choice(len(probabilities), p=probabilities)

        # Convert the index back to the DNA base using the reverse mapping
        decoded_sequence.append(reverse_mapping[sampled_index])

    # Join the list into a string and return
    return ''.join(decoded_sequence)


EPS = 1e-8

def proportional_strings(data, total_size=1000):
    # Extract strings and counts
    strings, counts = zip(*data)

    # Calculate the total sum of counts
    total_counts = sum(counts)

    # Calculate proportional frequencies scaled to the target size (1000)
    scaled_occurrences = [count / total_counts * total_size for count in counts]

    # Generate the final list using probabilistic sampling
    result = []
    for string, scaled_occ in zip(strings, scaled_occurrences):
        # For each string, use a binomial distribution to decide how many times it should appear
        occ = np.random.binomial(total_size, scaled_occ / total_size)
        result.extend([string] * occ)

    # Ensure the final list is of size exactly 1000
    if len(result) > total_size:
        # If there are too many elements, truncate randomly
        result = random.sample(result, k=total_size)
    elif len(result) < total_size:
        # If there are too few elements, add random strings from the data set to fill up the list
        while len(result) < total_size:
            result.append(random.choice(result))

    # Shuffle the result to randomize the order of strings
    np.random.shuffle(result)

    return result

#proportional_strings([("ATCG",10),("AAAA",2),("GGGGG",5)],total_size=5)


df = pd.read_csv("Summary_all.csv",index_col=False)

print('Download ok')

# Convert stringified dicts to dicts (once)
df['Mutants'] = df['Mutants'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Precompute weighted value
df['weighted_mut'] = df['percentage_mutants'] * df['TR_count']

df = (
    df
    .groupby(['TR', 'Library'], as_index=False)
    .agg(
        TR_count=('TR_count', 'sum'),
        weighted_mut=('weighted_mut', 'sum'),
        Mutants=('Mutants', lambda x: dict(sum((Counter(d) for d in x), Counter())))
    )
)

# Final weighted average
df['percentage_mutants'] = df['weighted_mut'] / df['TR_count']

# Cleanup
df = df.drop(columns='weighted_mut')

# Shuffle if needed
df = df.sample(frac=1).reset_index(drop=True)

# Libraries of interest
libs = ['Lib_4A', 'Lib_4B']

# Keep only rows from Lib_4A and Lib_4B
df_4 = df[df['Library'].isin(libs)]

# Count how many DISTINCT libraries each TR appears in
tr_lib_counts = (
    df.groupby('TR')['Library']
      .nunique()
)

# Keep TRs that appear in ONLY ONE library overall
valid_trs = tr_lib_counts[tr_lib_counts == 1].index

# Final filtered dataframe
df_Lib4 = df_4[df_4['TR'].isin(valid_trs)]


X_seq=[]
y_seq=[]
sampled_weights=[]

Ngood=len(df_Lib4['TR'])
ITER=np.random.permutation(Ngood)[::]

print(Ngood)
index=0
for i, tr in df_Lib4.iloc[ITER].iterrows():
  print(str(index)+' is treated over '+str(Ngood))
  index+=1
  if tr.percentage_mutants>3:
    seqVR=list((tr.Mutants).keys())
    c_list=[(tr.Mutants)[s] for s in seqVR]
    if len(seqVR)>5:
        ind=np.array([len(v)==len(tr.TR) and not('N' in v) and v!=tr.TR for v in seqVR])
        seqVR=np.array(seqVR)[ind]
        c_list=np.array(c_list)[ind]
        geno_flt=list(zip(seqVR,c_list))
        if len(geno_flt)>10:
            mutseq_list=[(one_hot_encode(VR[::-1]),n) for VR, n in geno_flt]

            y_seq+=proportional_strings(mutseq_list,total_size=600)
            X_seq+=[np.concatenate((one_hot_encode(tr.TR[::-1]),[[i,len(tr.TR)] for i in range(len(tr.TR))]),axis=1)]*600
print('Finished')

seq_lengths=np.array([len(x) for x in X_seq])

def pad_sequence(seq, maxlen, padding_value=0):
    pad_width = ((0, maxlen - seq.shape[0]), (0, 0))  # pad rows only
    return np.pad(seq, pad_width, mode='constant', constant_values=padding_value)

# Step 1: Find max sequence length
max_len = max(seq.shape[0] for seq in X_seq + y_seq)

# Step 2: Pad all sequences
X_seq_padded = [pad_sequence(seq, max_len) for seq in X_seq]
y_seq_padded = [pad_sequence(seq, max_len) for seq in y_seq]

# Step 3: Convert to numpy arrays
X_seq = np.array(X_seq_padded)
y_seq = np.array(y_seq_padded)




#X_train, X_test, y_rate_train, y_rate_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_rate, y_seq, test_size=0.2)
X_train, X_test=X_seq[len(X_seq)*2//10:],X_seq[:len(X_seq)*2//10]
y_train, y_test=y_seq[len(X_seq)*2//10:],y_seq[:len(X_seq)*2//10]

X_test_copy,y_test_copy=np.copy(X_test),np.copy(y_test)

seq_lengths_train,seq_lengths_test=seq_lengths[len(X_seq)*2//10:],seq_lengths[:len(X_seq)*2//10]
indices = np.random.permutation(len(X_train))
X_train=X_train[indices]
seq_lengths_train=seq_lengths_train[indices]
y_train=y_train[indices]

indices = np.random.permutation(len(X_test))
X_test=X_test[indices]
seq_lengths_test=seq_lengths_test[indices]
y_test=y_test[indices]

vocab_size=4
X_mut_train=np.zeros((X_train.shape[0],X_train.shape[1],2*vocab_size+1))
X_mut_train[:,1:,:vocab_size]=X_train[:,:-1,:vocab_size]
X_mut_train[:,1:,vocab_size:2*vocab_size]=y_train[:,:-1,:]
X_mut_train[:,0,2*vocab_size]=1

X_mut_test = np.zeros((X_test.shape[0], X_test.shape[1], 2 * vocab_size + 1))

# Shift the sequence in X_test and populate the corresponding previous outputs from y_test
X_mut_test[:, 1:, :vocab_size] = X_test[:, :-1, :vocab_size]  # Shift X_test
X_mut_test[:, 1:, vocab_size:2 * vocab_size] = y_test[:, :-1, :]  # Previous output from y_test
X_mut_test[:, 0, 2 * vocab_size] = 1  # Set the first step to 1 for the special indicator

X_test=X_test[:,:,:5]
X_train=X_train[:,:,:5]

# Create a proper Keras Metric class instead of a function
class MaskedAccuracy(metrics.Metric):
    def __init__(self, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true, y_pred shape: (batch, time, vocab_size)
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        
        # Create mask
        mask = tf.cast(tf.reduce_any(tf.not_equal(y_true, 0.0), axis=-1), tf.float32)
        
        # Calculate matches
        matches = tf.cast(tf.equal(y_true_labels, y_pred_labels), tf.float32)
        
        # Apply mask
        masked_matches = matches * mask
        
        # Update totals
        self.total.assign_add(tf.reduce_sum(masked_matches))
        self.count.assign_add(tf.reduce_sum(mask))
    
    def result(self):
        return self.total / tf.maximum(self.count, 1e-8)
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

vocab_size = 4
max_len = X_train.shape[1]
lstm_units = 8

# ===== Inputs =====
input_seq = layers.Input(shape=(max_len, vocab_size+1), name='DNA_input')
input_mut = layers.Input(shape=(max_len, 2 * vocab_size + 1), name='mutation_input')

# ===== First BiLSTM encoder =====
masked_seq = layers.Masking(mask_value=0.0, name='mask_seq')(input_seq)

bilstm_1 = layers.Bidirectional(
    layers.LSTM(lstm_units, return_sequences=True),
    name='bilstm_1'
)(masked_seq)

intermediate_dense = layers.TimeDistributed(
    layers.Dense(vocab_size+2, activation='softmax'),
    name='intermediate_dense'
)(bilstm_1)

# Second intermediate dense (ReLU, size=4)
intermediate_dense_relu = layers.TimeDistributed(
    layers.Dense(2, activation='relu'),
    name='intermediate_dense_relu'
)(bilstm_1)

# ===== Concatenate inputs, mutation info, and intermediate outputs =====
concat_1 = layers.Concatenate(name='concat_inputs')([
    input_seq,
    input_mut,
    intermediate_dense,
    intermediate_dense_relu
])

masked_concat = layers.Masking(mask_value=0.0, name='mask_concat')(concat_1)

# ===== Second LSTM =====
lstm_2 = layers.LSTM(
    lstm_units, return_sequences=True, name='lstm_2'
)(masked_concat)

final_dense = layers.TimeDistributed(
    layers.Dense(vocab_size, activation='softmax'),
    name='final_dense'
)(lstm_2)

# ===== Model =====
model = models.Model(inputs=[input_seq, input_mut], outputs=final_dense, name='DNA_seq_model')

# Compile with the Metric class
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[MaskedAccuracy()]
)

# Train the model
batch_size = 256
epochs =  200 # Adjust as needed

model.summary()

# Create sample weights for masking the loss
def create_sample_weights(y):
    """Create sample weights: 1 for valid positions, 0 for padding"""
    mask = np.any(y != 0, axis=-1).astype('float32')
    return mask

# --- Training ---

X_train = X_train.astype('float32')
X_mut_train = X_mut_train.astype('float32')
y_train = y_train.astype('float32')

X_test = X_test.astype('float32')
X_mut_test = X_mut_test.astype('float32')
y_test = y_test.astype('float32')

# Create sample weights
sample_weight_train = create_sample_weights(y_train)
sample_weight_test = create_sample_weights(y_test)

print("Starting training...")
history = model.fit(
    [X_train, X_mut_train],
    y_train,
    sample_weight=sample_weight_train,
    validation_data=([X_test, X_mut_test], y_test, sample_weight_test),
    batch_size=batch_size,
    epochs=epochs
)

#model.save("LSTM_model_8_8.keras")
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history_8_8_"+str(args.task_id)+".csv", index=False)
