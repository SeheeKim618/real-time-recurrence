import jax
from jax import random, vmap, value_and_grad, jit, ops
import jax.random as jrandom
import jax.numpy as np
from jax.example_libraries import optimizers
from jax.nn import sigmoid, softmax
from jax._src.util import partial
import jax.numpy as jnp
import numpy as old_np
from scipy.linalg import svd
import time
#from sophia import SophiaG

from utils import CrossEntropyLoss, CrossEntropyLoss_RTRL, one_hot_encoding, calculateSnApPattern, SparseMatrix, jacrev

import os
import matplotlib.pyplot as plt

"""
LSTM model with BPTT and RTRL training algorithm.
"""
class GRU:

    def __init__(self, 
                 key,
                 output_size, 
                 hidden_size, 
                 batch_size, 
                 recurrent_density, 
                 in_out_density, 
                 snap_level, 
                 lossFunction, 
                 algo_type,
                 seq_length,
                 vocab_size, 
                 embedding_dim=32,
                 logEvery=1, 
                 learning_rate=5e-3, 
                 online=True):

        self.key = key
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size= batch_size
        self.recurrent_density = recurrent_density
        self.in_out_density = in_out_density
        self.logEvery = logEvery
        self.online = online
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.algo_type = algo_type

        self.activation = softmax
        self.lossFunction = lossFunction

        # Initialize the embedding matrix (random values)
        key = jrandom.PRNGKey(0)
        self.embedding_matrix = jrandom.normal(key, (vocab_size, embedding_dim)) * 0.01

        self.initialize_weights()
        self.jacobian_init_time = 0.0

        print('GRU with '+ self.algo_type)
        print('Dense GRU params: ', (3*self.hidden_size*(self.embedding_dim+self.hidden_size) + self.hidden_size*self.output_size))
        print('Sparse GRU params: ', len(self.paramsData.flatten()))
        print('Density: ', self.recurrent_density)
        #print("Estimated FLOPS per GRU forward step:", self.compute_forward_flops())  

        '''if self.algo_type == 'rtrl':
            #self.jacobian_init_time = self.initialize_jacob(2)
            self.jacobian_init_time = self.initialize_jacob_rtrl()

            print('Online Updates!')
            self.update = self.update_online
        elif self.algo_type == 'snap':
            self.jacobian_init_time = self.initialize_jacob(snap_level)

            if (self.online):
                print('Online Updates!')
                self.update = self.update_online

            else:
                print('Offline Updates!')
                self.update = self.update_offline
        elif self.algo_type in ['bptt', 'lora_rtrl']:
            self.update = self.update_BPTT'''

        if self.algo_type == 'snap':
            self.jacobian_init_time = self.initialize_jacob(snap_level)

            if (self.online):
                print('Online Updates!')
                self.update = self.update_online

            else:
                print('Offline Updates!')
                self.update = self.update_offline
        elif self.algo_type in ['bptt', 'lora_rtrl']:
            self.update = self.update_BPTT

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=self.learning_rate, b1=0.9, b2=0.999, eps=1e-8)
        self.opt_state = self.opt_init(self.paramsData)
        self.opt_update = jit(self.opt_update)

    def initialize_jacob(self, snap_level):
        start_time = time.time()
        print('Init Jacobian Matrix ', snap_level)

        # Combine the input weight matrix indices for GRU
        weightRows = np.concatenate((self.Wr.rows, self.Wu.rows, self.Wh.rows))
        weightCols = np.concatenate((self.Wr.cols, self.Wu.cols, self.Wh.cols))

        # Combine the recurrent weight matrix indices for GRU
        recurrentRows = np.concatenate((self.Rr.rows, self.Ru.rows, self.Rh.rows))
        recurrentCols = np.concatenate((self.Rr.cols, self.Ru.cols, self.Rh.cols))

        # Calculate the SnAP pattern based on GRU weight indices
        SnAP_rows, SnAP_cols = calculateSnApPattern(snap_level, weightRows, weightCols, recurrentRows, recurrentCols)
        
        self.J = SparseMatrix()
        #print("shape of SnAP_rows: ", SnAP_rows.shape) #(143360,)
        #print("shape of SnAP_cols: ", SnAP_cols.shape) #(143360,)
        self.J.jacobian(SnAP_rows, SnAP_cols, (self.hidden_size, self.Rh.end), 0)

        # End timing
        end_time = time.time()
        
        # Calculate elapsed time
        jacobian_time = end_time - start_time

        print('Jacobian Shape: ', self.J.shape)
        print('Jacobian params: ', self.J.len)
        print('Jacobian density: ', self.J.density)

        return jacobian_time
 
    def initialize_weights(self):
        k1, k2, k3, k4, k5, k6, k7 = random.split(self.key, 7)

        # Initialize GRU weights
        self.Wr = SparseMatrix(k1, self.embedding_dim, self.hidden_size, self.in_out_density, 0)
        Wr_data = self.Wr.init()
        self.Wu = SparseMatrix(k2, self.embedding_dim, self.hidden_size, self.in_out_density, self.Wr.end)
        Wu_data = self.Wu.init()
        self.Wh = SparseMatrix(k3, self.embedding_dim, self.hidden_size, self.in_out_density, self.Wu.end)
        Wh_data = self.Wh.init()

        self.Rr = SparseMatrix(k4, self.hidden_size, self.hidden_size, self.recurrent_density, self.Wh.end)
        Rr_data = self.Rr.init()
        self.Ru = SparseMatrix(k5, self.hidden_size, self.hidden_size, self.recurrent_density, self.Rr.end)
        Ru_data = self.Ru.init()
        self.Rh = SparseMatrix(k6, self.hidden_size, self.hidden_size, self.recurrent_density, self.Ru.end)
        Rh_data = self.Rh.init()

        self.V = SparseMatrix(k7, self.output_size, self.hidden_size, self.in_out_density, self.Rh.end)
        V_data = self.V.init()

        self.paramsData = np.concatenate((Wr_data, Wu_data, Wh_data, Rr_data, Ru_data, Rh_data, V_data))

    @partial(jit, static_argnums=(0,))
    def gru(self, params, x, h):
        # Convert materialized params to dense if necessary
        wr_dense = self.Wr.toDense(params[self.Wr.start:self.Wr.end,])
        wu_dense = self.Wu.toDense(params[self.Wu.start:self.Wu.end,])
        wh_dense = self.Wh.toDense(params[self.Wh.start:self.Wh.end,])
        rr_dense = self.Rr.toDense(params[self.Rr.start:self.Rr.end,])
        ru_dense = self.Ru.toDense(params[self.Ru.start:self.Ru.end,])
        rh_dense = self.Rh.toDense(params[self.Rh.start:self.Rh.end,])

        # Update gate
        z = sigmoid(np.dot(x, wu_dense) + np.dot(h, ru_dense))
        # Reset gate
        r = sigmoid(np.dot(x, wr_dense) + np.dot(h, rr_dense))
        # Candidate hidden state
        h_tilde = np.tanh(np.dot(x, wh_dense) + np.dot(r * h, rh_dense))

        # Final hidden state
        h = (1 - z) * h + z * h_tilde

        return h

    @partial(jit, static_argnums=(0,))
    def forward_step(self, params, x, h, Jh_data):

        x = self.embed_input(x)

        (grad_h_params, grad_h_h), h = jacrev(self.gru, argnums=(0, 2))(params, x, h)
    

        #print("type of grad_h_params: ", type(grad_h_params)) #DynamicJaxprTracer
        #print("shape of grad_h_params: ", grad_h_params.shape) #(32,4480)
        #print("grad_h_params: ", grad_h_params) #[32, 4480]

        #print("Jh_data: ", Jh_data.shape) #(143360,)
        #print("self.J.toDense(Jh_data): ", self.J.toDense(Jh_data).shape) #(32, 4480)
        #print("type of self.J: ", type(self.J)) #class 'utils.SparseMatrix'
        #print("shape of self.J: ", self.J.shape) #(32, 4480)
        #print("shape of self.J.coords: ", len(self.J.coords)) #2
        print("hiiiiii", self.J.toDense(Jh_data).shape)
        h_Jh = np.dot(grad_h_h, self.J.toDense(Jh_data))[tuple(self.J.coords)]
        Jh = grad_h_params[tuple(self.J.coords)] + h_Jh

        ############

        return h, Jh

    @partial(jit, static_argnums=(0,))
    def calculate_loss(self, params, h, y):
        #print("shape of params before activation: ", params.shape) #(32,)
        #print("shape of h before activation: ", h.shape) #(32,)
        
        logits = np.dot(self.V.toDense(params), h)  # Compute raw logits
        output = self.activation(logits)  # Apply softmax instead of sigmoid
        #print("Output (first few values):", output[:5])

        #print("y (first few samples):", y[:5])
        #print("train output: ", output.shape)
        loss = self.lossFunction(output, y)
        #print("train loss: ", output.shape)
        return loss
    
    @partial(jit, static_argnums=(0,))
    def combineGradients(self, grad_h, grad_out_params, Jh_data):
        grad_rec_params = np.dot(grad_h, self.J.toDense(Jh_data))
        #print("shape of grad_rec_params", grad_rec_params.shape)
        #print("shape of grad_out_params", grad_out_params.shape)
        
        return np.concatenate((grad_rec_params, grad_out_params))
    
    @partial(jit, static_argnums=(0,))
    def calculate_grads_step(self, params, x, y, h, Jh_data):

        #print("length of params In cal grads", len(params[:self.Rz.end,]))
        h, Jh_data = self.forward_step(params[:self.Rh.end,], x, h, Jh_data)
        #self.debug_forward_step(params[:self.Rz.end,], x, h, c, Jh_data, Jc_data)

        loss, (grad_out_params, grad_h) = value_and_grad(self.calculate_loss, argnums=(0,1))(params[self.Rh.end:,], h, y) # calculation of gradients 
        gradient = self.combineGradients(grad_h, grad_out_params, Jh_data)
            
        return loss, gradient, h, Jh_data

    batch_calculate_grads_step = vmap(calculate_grads_step, in_axes=(None, None, 0, 0, 0, 0))

    def update_online(self, params, x, y):
        h = np.zeros((self.batch_size, self.hidden_size))
        Jh_data = np.zeros((self.batch_size, self.J.len))

        losses = []

        for t in range(x.shape[1]): #sequence
            loss, grads, h, Jh_data = self.batch_calculate_grads_step(params, x[:, t], y[:, t], h, Jh_data)
            losses.append(np.mean(loss, axis=0))
            #print("Type of grads:", type(grads)) #ArrayImpl
            #print("Shape of grads:", grads.shape) #(16, 4512) -> 72192
            self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
            params = self.get_params(self.opt_state)

        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)

    def calculate_grads(self, params, x, y):
        
        h = np.zeros(self.hidden_size)

        Jh_data = np.zeros(self.J.len)

        losses = []
        gradients = np.zeros_like(self.paramsData)

        for t in range(x.shape[0]): #seq_length
            loss, gradient, h, Jh_data = self.calculate_grads_step(params, x[t], y[t], h, Jh_data)
            losses.append(loss)
            gradients = gradients + gradient/x.shape[0]
            
            
        return np.mean(np.array(losses)), gradients

    batch_calculate_grads = vmap(calculate_grads, in_axes=(None, None, 0, 0))

    def update_offline(self, params, x, y):
        losses, grads = self.batch_calculate_grads(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)

        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)

    def update_BPTT(self, params, x, y):
        #print("x_shape_training_1: ", x.shape) #x: (32, 100) (batch, sequence)
        loss, grads = self.batch_calculate_grads_BPTT(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)

        return self.get_params(self.opt_state), np.sum(loss, axis=0) / x.shape[0]
        
    def calculate_loss_BPTT(self, params, x, y):

        output = self.forward(params, x)
        #print("output: ", output.shape)
        loss = self.lossFunction(output, y)
        #print("loss: ", loss.shape)
        return loss

    def calculate_grads_BPTT(self, params, x, y):

        return value_and_grad(self.calculate_loss_BPTT)(params, x, y) #return grads from here

    batch_calculate_grads_BPTT = vmap(calculate_grads_BPTT, in_axes=(None, None, 0, 0)) #batch training, 배치 크기만큼의 손실 값 (예: shape (32,))이 반환

    def calculate_loss_val(self, params, x, y):

        output = self.forward(params, x)
        losses = vmap(self.lossFunction)(output, y)  # losses shape: (100,)
        mean_loss = jnp.mean(losses)
        return mean_loss

    def forward(self, params, x):

        x = self.embed_input(x)
        #print("after embedding: ", x.shape) #(100, 32)

        h = np.zeros(self.hidden_size)
        o = np.zeros((x.shape[0], self.output_size))

        for t in range(x.shape[0]): # x: (seq_legth, embedding_dim)
            h, o = self.forward_step_BPTT(params, x, t, h, o) #bptt는 self.forward_step_BPTT로 바꾸기, 리턴값에서 o만 씀
            
        return o

    @partial(jit, static_argnums=(0,))
    def forward_step_BPTT(self, paramsData, x, t, h, o):
        h = self.gru(paramsData[:self.Rh.end,], x[t], h)

        logits = np.dot(self.V.toDense(paramsData[self.Rh.end:,]), h)
        #print("Logits before softmax:", logits[:5])
        output = self.activation(logits)
        #print("Softmax output: {}", output)
        #print("Sum of softmax output (should be ~1): {}", np.sum(softmax(logits), axis=-1))
        o = o.at[t].set(output)

        return h, o

    def predict(self, x, y):
        #print("x_shape_validation: ", x.shape)  # Expected: (batch_size, seq_length)
        # Vectorize calculate_loss_BPTT over the batch dimension of x and y:
        if self.algo_type in ['bptt', 'lora_rtrl']:
            val_loss_fn = vmap(self.calculate_loss_BPTT, in_axes=(None, 0, 0))
        else:
            val_loss_fn = vmap(self.calculate_loss_val, in_axes=(None, 0, 0))
        # This applies calculate_loss_BPTT for each sample in the batch.
        return val_loss_fn(self.paramsData, x, y)

    def evaluate(self, x_val, y_val):
        # Compute the loss for each sample in the batch:
        losses = self.predict(x_val, y_val)
        # Average over the batch to get a scalar validation loss:
        return np.mean(losses)

    def embed_input(self, x):
        """Converts token IDs to embedding vectors."""
        return self.embedding_matrix[x]
    
    def sample_batch(self, token_ids, batch_size, seq_length):
        total_tokens = len(token_ids)
        max_start = total_tokens - seq_length - 1  # Prevents index overflow
        
        key = jax.random.PRNGKey(0)  # Ensure reproducibility
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=max_start)  # Random start points
        
        batch = np.array([token_ids[idx: idx + seq_length + 1] for idx in indices], dtype=np.int32)  # Ensure integer type

        return jnp.array(batch)

    def run(self, epochs, data, validation_data):

        losses = []
        validation_losses = []
        epochs_list = []  # To store epoch numbers

        # Start timing
        start_time = time.time()

        for i, k in zip(range(epochs), random.split(self.key, epochs)):
            
            #print("before sampling x: ", data.shape)
            sampled_data = self.sample_batch(data, self.batch_size, self.seq_length)
            #print("after sampling x: ", sampled_data.shape)
            x, y = sampled_data[:, :-1], sampled_data[:, 1:]
            #print("train x: ", x.shape) #(32, 100)
            self.paramsData, loss = self.update(self.paramsData, x, y)
            losses.append(loss)

            # 검증 데이터에 대한 손실을 계산합니다.
            validation_sampled_data = self.sample_batch(validation_data, self.batch_size, self.seq_length)
            x_val, y_val = validation_sampled_data[:, :-1], validation_sampled_data[:, 1:]
            #print("valid x: ", x_val.shape)
            val_loss = self.evaluate(x_val, y_val)
            validation_losses.append(val_loss)

            # Track epoch number
            epochs_list.append(i)

            if i % 5 == 0:
                print('Epoch', "{:04d}".format(i))
                print('Train Loss ', loss)
                print('Validation Loss:', val_loss)

        # End timer
        end_time = time.time()

        # Calculate total training time
        total_training_time = end_time - start_time
        print(f"Total training time: {total_training_time} seconds")

        return self.paramsData, losses, validation_losses, epochs_list, total_training_time

    '''def compute_forward_flops(self):
        """
        Compute an approximate theoretical FLOPS count for one forward step 
        in the GRU (dense version). This is an estimate based on the dominant
        operations (matrix multiplications and elementwise operations).

        Note: This is an approximation.
        """
        flops = 0
        # For the GRU cell, assume the following dot products:
        # Update gate: np.dot(x, wu_dense)  --> 2 * embedding_dim * hidden_size
        flops += 2 * self.embedding_dim * self.hidden_size
        # Update gate: np.dot(h, ru_dense)  --> 2 * hidden_size * hidden_size
        flops += 2 * self.hidden_size * self.hidden_size
        
        # Reset gate: np.dot(x, wr_dense)
        flops += 2 * self.embedding_dim * self.hidden_size
        # Reset gate: np.dot(h, rr_dense)
        flops += 2 * self.hidden_size * self.hidden_size
        
        # Candidate hidden state: np.dot(x, wh_dense)
        flops += 2 * self.embedding_dim * self.hidden_size
        # Candidate hidden state: np.dot(r * h, rh_dense)
        flops += 2 * self.hidden_size * self.hidden_size

        # Elementwise operations: 
        #  - Multiplication for r * h
        flops += self.hidden_size
        #  - Combining hidden states: (1-z)*h + z*h_tilde
        flops += 2 * self.hidden_size

        return flops'''

    