# RNA Reinforcement Learning with Structure and Motif Objectives + Penalty + Visualization + Oracle Hook

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os
import random

class Brain(keras.Model):
    def __init__(self, num_layers, action_network_num_outputs):
        super(Brain, self).__init__()
        self.layer1 = layers.Dense(num_layers, activation='tanh')
        self.output_layer = layers.Dense(action_network_num_outputs, activation=None)

    def call(self, input_vector):
        x = input_vector
        y = self.layer1(x)
        y2 = self.output_layer(y)
        return y2

class DifferentiableRNAEnvironment:
    def __init__(self, seq_length, state_dimension, num_memory_nodes, type_mem, vocab_size=4, batch_size=1):
        self.type_mem = type_mem
        self.state_dimension = state_dimension
        self.num_memory_nodes = num_memory_nodes
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.generated_sequences = None

        if self.type_mem == "Full_LSTM":
            self.memory_size = self.num_memory_nodes * 2
        elif self.type_mem in ["Minimal_GRU", "Full_GRU", "CARU"]:
            self.memory_size = self.num_memory_nodes
        else:
            self.memory_size = 0

        self.initial_state = self.reset()

    def reset(self):
        self.timestep = 0
        self.gc_target = np.random.uniform(0.3, 0.7)
        self.motif_target_index = random.randint(4, self.seq_length - 4)
        self.motif_target_base = random.randint(0, 3)
        self.generated_sequences = tf.TensorArray(dtype=tf.int32, size=self.seq_length)
        first_nt = tf.one_hot(tf.zeros((self.batch_size,), dtype=tf.int32), depth=self.vocab_size)
        memory_state = tf.zeros((self.batch_size, self.memory_size), dtype=tf.float32)
        state = tf.concat([first_nt, memory_state], axis=1)
        return state

    def step(self, state, action, t):
        if self.type_mem == "Full_LSTM":
            h = state[:, self.vocab_size:self.vocab_size + self.memory_size // 2]
            c = state[:, self.vocab_size + self.memory_size // 2:]
        old_memory_state = state[:, self.vocab_size:]

        next_base_logits = action[:, :self.vocab_size]
        sampled_soft_next_base = tf.nn.softmax(next_base_logits)
        chosen_index = tf.argmax(sampled_soft_next_base, axis=1, output_type=tf.int32)

        self.generated_sequences = self.generated_sequences.write(t, chosen_index)

        if self.num_memory_nodes > 0:
            action_memory = action[:, self.vocab_size:]
            if self.type_mem == "Minimal_GRU":
                input_gate, new_value = tf.split(action_memory, 2, axis=1)
                input_gate = tf.sigmoid(input_gate)
                new_memory = tf.tanh(new_value) * input_gate + (1 - input_gate) * old_memory_state
            elif self.type_mem == "Full_GRU":
                x_z, x_r, x_h = tf.split(action_memory, 3, axis=1)
                z = tf.sigmoid(x_z)
                r = tf.sigmoid(x_r)
                hh = tf.tanh(x_h * r)
                new_memory = (1 - z) * old_memory_state + z * hh
            elif self.type_mem == "Full_LSTM":
                input_gate, forget_gate, output_gate, new_value = tf.split(action_memory, 4, axis=1)
                i = tf.sigmoid(input_gate)
                f = tf.sigmoid(forget_gate)
                o = tf.sigmoid(output_gate)
                c = f * c + i * tf.tanh(new_value)
                h = o * tf.tanh(c)
                new_memory = tf.concat([h, c], axis=1)
            elif self.type_mem == "CARU":
                x, a, b = tf.split(action_memory, 3, axis=1)
                n = tf.tanh(a)
                l = tf.sigmoid(x) * tf.sigmoid(b)
                new_memory = (1 - l) * old_memory_state + l * n
            else:
                new_memory = tf.tanh(action_memory)
        else:
            new_memory = tf.zeros_like(old_memory_state)

        next_state = tf.concat([sampled_soft_next_base, new_memory], axis=1)
        reward = self.compute_combined_reward(sampled_soft_next_base, t, next_base_logits)
        return reward, next_state

    def compute_soft_gc_reward(self, soft_one_hot):
        gc_mask = tf.constant([0, 1, 1, 0], dtype=tf.float32)
        gc_probs = tf.reduce_sum(soft_one_hot * gc_mask, axis=1)
        return 1.0 - tf.abs(gc_probs - self.gc_target) * 2.0

    def compute_motif_reward(self, soft_one_hot, t):
        weights = tf.one_hot(self.motif_target_base, depth=4)
        mask = tf.cast(tf.equal(t, self.motif_target_index), tf.float32)
        return tf.reduce_sum(soft_one_hot * weights, axis=1) * mask

    def compute_structure_surrogate(self, soft_one_hot):
        entropy = -tf.reduce_sum(soft_one_hot * tf.math.log(soft_one_hot + 1e-8), axis=1)
        return tf.clip_by_value(entropy, 0.0, 1.0)

    def compute_repeat_penalty(self):
        gathered = tf.transpose(self.generated_sequences.stack(), perm=[1, 0])
        penalties = []
        for i in range(self.batch_size):
            seq = gathered[i].numpy()
            max_count = max([seq.tolist().count(nt) for nt in range(self.vocab_size)])
            penalties.append(-0.1 * max(0, max_count - (self.seq_length // self.vocab_size)))
        return tf.constant(penalties, dtype=tf.float32)

    def vienna_oracle_reward(self):
        idx_to_nt = ['A', 'C', 'G', 'U']
        gathered = tf.transpose(self.generated_sequences.stack(), perm=[1, 0])
        sequence_strs = ["".join([idx_to_nt[i] for i in ex.numpy()]) for ex in gathered]
        scores = []
        for seq in sequence_strs:
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp:
                temp.write(seq)
                temp_path = temp.name
            try:
                result = subprocess.run(["RNAfold", temp_path], capture_output=True, text=True)
                output = result.stdout
                energy_line = output.strip().split("\n")[-1]
                energy = float(energy_line.split('(')[-1].split(')')[0])
                scores.append(-energy)
            except:
                scores.append(0.0)
            finally:
                os.unlink(temp_path)
        return tf.constant(scores, dtype=tf.float32)

    def compute_combined_reward(self, soft_one_hot, t, logits):
        gc_reward = self.compute_soft_gc_reward(soft_one_hot)
        motif_reward = self.compute_motif_reward(soft_one_hot, t)
        structure_reward = self.compute_structure_surrogate(soft_one_hot)
        oracle_reward = self.vienna_oracle_reward() if t == self.seq_length - 2 else tf.zeros((self.batch_size,),
                                                                                              dtype=tf.float32)
        entropy = -tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=1)
        kl_prior = tf.nn.softmax(logits) * tf.math.log((tf.nn.softmax(logits) + 1e-8) / 0.25)
        kl_div = tf.reduce_sum(kl_prior, axis=1)
        repeat_penalty = self.compute_repeat_penalty() if t == self.seq_length - 2 else tf.zeros((self.batch_size,),
                                                                                                 dtype=tf.float32)
        return gc_reward + motif_reward + structure_reward + oracle_reward + 0.1 * entropy - 0.1 * kl_div + repeat_penalty


def decode_sequence(sequences):
    idx_to_nt = ['A', 'C', 'G', 'U']
    return ["".join(idx_to_nt[i] for i in seq) for seq in sequences]

def plot_nucleotide_frequencies(sequences, title):
    array = np.array(sequences)
    heatmap = np.zeros((4, array.shape[1]))
    for pos in range(array.shape[1]):
        for nt in range(4):
            heatmap[nt, pos] = np.mean(array[:, pos] == nt)
    plt.figure(figsize=(10, 3))
    plt.imshow(heatmap, cmap='Blues', aspect='auto')
    plt.colorbar(label='Frequency')
    plt.yticks([0, 1, 2, 3], ['A', 'C', 'G', 'U'])
    plt.xlabel('Position in Sequence')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_heatmap.png")
    plt.close()


def compute_gc_ratio(sequences_tensor):
    # sequences_tensor shape: [batch_size, seq_length]
    is_gc = tf.logical_or(tf.equal(sequences_tensor, 1), tf.equal(sequences_tensor, 2))
    gc_counts = tf.reduce_sum(tf.cast(is_gc, tf.float32), axis=1)
    gc_ratio = gc_counts / tf.cast(tf.shape(sequences_tensor)[1], tf.float32)
    return gc_ratio

def plot_gc_distribution(sequences, title):
    # Ensure it's a tensor
    if isinstance(sequences, tf.Tensor):
        sequences = sequences.numpy()
    elif isinstance(sequences, tf.TensorArray):
        sequences = sequences.stack().numpy()
        sequences = np.transpose(sequences, (1, 0))  # [batch, length]

    # Compute GC ratio for each sequence (G=2, C=1)
    gc_counts = np.sum((sequences == 1) | (sequences == 2), axis=1)
    gc_ratio = gc_counts / sequences.shape[1]

    # Plot histogram
    plt.hist(gc_ratio, bins=10, range=(0, 1), color='green', alpha=0.7)
    plt.title(f"GC Content Distribution - {title}")
    plt.xlabel("GC Ratio")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"results/{title.replace(' ', '_').lower()}_gc_histogram.png")
    plt.close()


def plot_oracle_structure_scores(sequences, title):
    idx_to_nt = ['A', 'C', 'G', 'U']
    sequence_strs = ["".join(idx_to_nt[i] for i in s) for s in sequences]
    energies = []
    for seq in sequence_strs:
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp:
            temp.write(seq)
            path = temp.name
        try:
            result = subprocess.run(["RNAfold", path], capture_output=True, text=True)
            energy_line = result.stdout.strip().split("\n")[-1]
            energy = float(energy_line.split('(')[-1].split(')')[0])
            energies.append(energy)
        except:
            energies.append(0.0)
        finally:
            os.unlink(path)
    plt.hist(energies, bins=15, color='red', alpha=0.7)
    plt.title(f"Structure Energies - {title}")
    plt.xlabel("Free Energy (kcal/mol)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"results/{title.replace(' ', '_').lower()}_structure_histogram.png")
    plt.close()


def run_pipeline(memory_type):
    print(f"Running for memory: {memory_type}")
    original_n_memories = 64
    num_memory_nodes = original_n_memories
    if memory_type == "Minimal_GRU":
        num_memory_nodes *= 2
    elif memory_type == "Full_GRU":
        num_memory_nodes *= 3
    elif memory_type == "Full_LSTM":
        num_memory_nodes *= 4
    elif memory_type == "CARU":
        num_memory_nodes *= 3
    else:
        num_memory_nodes = 0

    state_dimension = 4 + num_memory_nodes
    action_network_num_outputs = 4 + num_memory_nodes
    env = DifferentiableRNAEnvironment(
        seq_length=30,
        state_dimension=state_dimension,
        num_memory_nodes=original_n_memories,
        type_mem=memory_type,
        vocab_size=4,
        batch_size=10
    )
    brain = Brain(128, action_network_num_outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    losses, rewards = [], []
    for iteration in range(500):
        with tf.GradientTape() as tape:
            init_state = env.reset()
            state = init_state
            total_rewards = tf.zeros((10,), dtype=tf.float32)
            for t in range(29):
                action = brain(state)
                reward, state = env.step(state, action, t)
                total_rewards += reward
            loss = -tf.reduce_mean(total_rewards)
        grads = tape.gradient(loss, brain.trainable_variables)
        opt.apply_gradients(zip(grads, brain.trainable_variables))
        losses.append(loss.numpy())
        rewards.append(tf.reduce_mean(total_rewards).numpy())
        if iteration % 100 == 0:
            print(f"{memory_type} | Iteration {iteration} | Loss: {loss.numpy():.4f} | Reward: {rewards[-1]:.4f}")

    plt.plot(rewards, label=f"{memory_type} Reward")
    sampled = tf.transpose(env.generated_sequences.stack(), perm=[1, 0]).numpy()
    decoded = decode_sequence(sampled)
    with open(f"results/{memory_type}_samples.txt", "w") as f:
        for s in decoded:
            f.write(s + "\n")
    plot_nucleotide_frequencies(sampled, f"results/Nucleotide Frequencies - {memory_type}")
    plot_gc_distribution(sampled, memory_type)
    plot_oracle_structure_scores(sampled, memory_type)

memory_types = ["No_memory", "Minimal_GRU", "Full_GRU", "Full_LSTM", "CARU"]

for mt in memory_types:
    run_pipeline(mt)

plt.legend()
plt.title("Comparison of Memory Types")
plt.xlabel("Iteration")
plt.ylabel("Loss/Reward")
plt.tight_layout()
plt.savefig("results/memory_type_comparison.png")

