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
class GRU_TBPTT:

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
                 trunc_length, 
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
        self.trunc_length = trunc_length

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

    def update_TBPTT(self, params, x, y, trunc_length):
        """
        TBPTT 방식의 온라인 업데이트.
        x: (batch_size, seq_length, input_dim)
        y: (batch_size, seq_length, output_dim)
        trunc_length: 한 번에 역전파할 시퀀스 길이
        """
        # 초기 은닉 상태
        h = np.zeros((self.batch_size, self.hidden_size))
        losses = []
        seq_length = x.shape[1]

        # 전체 시퀀스를 trunc_length 단위로 나눕니다.
        for t in range(0, seq_length, trunc_length):
            # 현재 chunk 추출 (batch_size, trunc_length, input_dim)
            x_chunk = x[:, t:t+trunc_length]
            # 현재 chunk의 target 추출 (batch_size, trunc_length, output_dim)
            y_chunk = y[:, t:t+trunc_length]

            # TBPTT를 위한 loss 및 gradient 계산 함수:
            # 이 함수는 주어진 chunk에 대해 순전파와 역전파를 진행하고,
            # 누적된 loss와, gradient, 그리고 마지막 은닉 상태를 반환해야 합니다.
            loss_chunk, grads, h = self.batch_calculate_grads_BPTT(self.paramsData, x_chunk, y_chunk, h)
            losses.append(loss_chunk)

            # 파라미터 업데이트: 여기서는 예시로 grads를 batch 전체에 대해 sum한 후 업데이트합니다.
            self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
            self.paramsData = self.get_params(self.opt_state)
            
            # 중요한 부분: 은닉 상태를 detach(또는 stop_gradient)해서 역전파 그래프를 분리합니다.
            h = jax.lax.stop_gradient(h)

        # 모든 chunk의 loss를 평균내어 최종 loss 반환
        return np.mean(np.array(losses))
        
    def calculate_loss_BPTT(self, params, x, y, h):

        output, h = self.forward(params, x, h)
        #print("output: ", output.shape)
        loss = self.lossFunction(output, y)
        #print("loss: ", loss.shape)
        return loss, h

    def calculate_grads_BPTT(self, params, x, y, h):
        (loss, h), grads = value_and_grad(self.calculate_loss_BPTT, has_aux=True)(params, x, y, h)
        return loss, grads, h

    batch_calculate_grads_BPTT = vmap(calculate_grads_BPTT, in_axes=(None, None, 0, 0, 0)) #batch training, 배치 크기만큼의 손실 값 (예: shape (32,))이 반환

    def forward(self, params, x, h):

        x = self.embed_input(x)
        #print("after embedding: ", x.shape) #(100, 32)

        o = np.zeros((x.shape[0], self.output_size))

        for t in range(x.shape[0]): # x: (seq_legth, embedding_dim)
            h, o = self.forward_step_BPTT(params, x, t, h, o) #bptt는 self.forward_step_BPTT로 바꾸기, 리턴값에서 o만 씀
            
        return o, h

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

        # 각 validation 샘플마다 초기 hidden state를 0으로 설정
        h_init = np.zeros((self.batch_size, self.hidden_size))
        val_loss_fn = vmap(self.calculate_loss_BPTT, in_axes=(None, 0, 0, 0))
        losses, _ = val_loss_fn(self.paramsData, x, y, h_init)

        # This applies calculate_loss_BPTT for each sample in the batch.
        return losses

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
            loss = self.update_TBPTT(self.paramsData, x, y, self.trunc_length)
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


    