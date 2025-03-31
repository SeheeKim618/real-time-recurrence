from jax import random, vmap, value_and_grad, jit, ops
import jax.numpy as np
from jax.example_libraries import optimizers
from jax.nn import sigmoid
from jax._src.util import partial
import jax.numpy as jnp
import numpy as old_np
from scipy.linalg import svd
import time
#from sophia import SophiaG

from utils import BinaryCrossEntropyLoss, calculateSnApPattern, SparseMatrix, jacrev

import os
import matplotlib.pyplot as plt

"""
LSTM model with BPTT and RTRL training algorithm.
"""
class GRU:

    def __init__(self, 
                 key,
                 input_size, 
                 output_size, 
                 hidden_size, 
                 batch_size, 
                 recurrent_density, 
                 in_out_density, 
                 snap_level, 
                 lossFunction, 
                 algo_type, 
                 logEvery=1, 
                 learning_rate=1e-3, 
                 online=True):

        self.key = key
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size= batch_size
        self.recurrent_density = recurrent_density
        self.in_out_density = in_out_density
        self.logEvery = logEvery
        self.online = online
        self.activation = sigmoid

        self.initialize_weights()
        self.jacobian_init_time = 0.0

        print('GRU with '+ algo_type)
        print('Dense GRU params: ', (3*hidden_size*(input_size+hidden_size) + hidden_size*output_size))
        print('Sparse GRU params: ', len(self.paramsData.flatten()))
        print('Density: ', recurrent_density)

        if algo_type == 'rtrl':
            self.jacobian_init_time = self.initialize_jacob(2)

            print('Online Updates!')
            self.update = self.update_online
        elif algo_type == 'snap':
            self.jacobian_init_time = self.initialize_jacob(snap_level)

            if (self.online):
                print('Online Updates!')
                self.update = self.update_online

            else:
                print('Offline Updates!')
                self.update = self.update_offline
        elif algo_type == 'bptt':
            self.update = self.update_bptt

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state = self.opt_init(self.paramsData)
        self.opt_update = jit(self.opt_update)
        self.lossFunction = lossFunction

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
        self.Wr = SparseMatrix(k1, self.input_size, self.hidden_size, self.in_out_density, 0)
        Wr_data = self.Wr.init()
        self.Wu = SparseMatrix(k2, self.input_size, self.hidden_size, self.in_out_density, self.Wr.end)
        Wu_data = self.Wu.init()
        self.Wh = SparseMatrix(k3, self.input_size, self.hidden_size, self.in_out_density, self.Wu.end)
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

        '''start_time = time.time()
        grad_h_params = jacrev(self.calc_grad_h_params, argnums=0)(params, x, h, c)
        grad_h_params_time = time.time() - start_time
        print(f"Time taken for grad_h_params: {grad_h_params_time:.6f} seconds")'''

        #print("length of params In forward", len(params)) #4480
        #print("params: ", params) #4480
        
        (grad_h_params, grad_h_h), h = jacrev(self.gru, argnums=(0, 2))(params, x, h)
    

        #print("type of grad_h_params: ", type(grad_h_params)) #DynamicJaxprTracer
        #print("shape of grad_h_params: ", grad_h_params.shape) #(32,4480)
        #print("grad_h_params: ", grad_h_params) #[32, 4480]

        #print("Jh_data: ", Jh_data.shape) #(143360,)
        #print("self.J.toDense(Jh_data): ", self.J.toDense(Jh_data).shape) #(32, 4480)
        #print("type of self.J: ", type(self.J)) #class 'utils.SparseMatrix'
        #print("shape of self.J: ", self.J.shape) #(32, 4480)
        #print("shape of self.J.coords: ", len(self.J.coords)) #2

        h_Jh = np.dot(grad_h_h, self.J.toDense(Jh_data))[tuple(self.J.coords)]
        Jh = grad_h_params[tuple(self.J.coords)] + h_Jh

        ############

        return h, Jh

    @partial(jit, static_argnums=(0,))
    def calculate_loss(self, params, h, y):
        #print("shape of params before activation: ", params.shape) #(32,)
        #print("shape of h before activation: ", h.shape) #(32,)
        output = self.activation(np.dot(self.V.toDense(params), h)) #inference에서는 얘를 forward_step 안으로 보내야하나?
        #print(f"Shape of output: {output.shape}") #(1,)
        loss = self.lossFunction(output, y)
        return loss
    
    @partial(jit, static_argnums=(0,))
    def combineGradients(self, grad_h, grad_out_params, Jh_data):
        grad_rec_params = np.dot(grad_h, self.J.toDense(Jh_data))
        #print("shape of grad_rec_params", grad_rec_params.shape)
        #print("shape of grad_out_params", grad_out_params.shape)
        
        return np.concatenate((grad_rec_params, grad_out_params))
    
    @partial(jit, static_argnums=(0,))
    def calculate_grads_step(self, params, x, y, h, Jh_data):

        #print("x_shape: ", x.shape) #(3,) - offline, online

        #print("length of params In cal grads", len(params[:self.Rz.end,]))
        h, Jh_data = self.forward_step(params[:self.Rh.end,], x, h, Jh_data)
        #self.debug_forward_step(params[:self.Rz.end,], x, h, c, Jh_data, Jc_data)

        loss, (grad_out_params, grad_h) = value_and_grad(self.calculate_loss, argnums=(0,1))(params[self.Rh.end:,], h, y) # calculation of gradients 
        gradient = self.combineGradients(grad_h, grad_out_params, Jh_data)
            
        return loss, gradient, h, Jh_data

    batch_calculate_grads_step = vmap(calculate_grads_step, in_axes=(None, None, 0, 0, 0, 0)) #batch training

    def update_online(self, params, x, y):
        h = np.zeros((self.batch_size, self.hidden_size))
        Jh_data = np.zeros((self.batch_size, self.J.len)) #(batch_size, Jacob_params) (16, 107520)
        #print("hiiiiiiiiiii", Jh_data.shape)

        losses = []

        #print("x_shape: ", x.shape) # (16, 22, 3)

        for t in range(x.shape[1]): #(batch_size, seq_length, input_size)
            #print("x[:, t, :]_shape: ", x[:, t, :].shape) #(16,3)
            loss, grads, h, Jh_data = self.batch_calculate_grads_step(params, x[:, t, :], y[:, t, :], h, Jh_data)
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

        for t in range(x.shape[0]): #(seq_length, input_size)
            loss, gradient, h, Jh_data = self.calculate_grads_step(params, x[t], y[t], h, Jh_data)
            losses.append(loss)
            gradients = gradients + gradient/x.shape[0]
            
        return np.mean(np.array(losses)), gradients

    batch_calculate_grads = vmap(calculate_grads, in_axes=(None, None, 0, 0))

    def update_offline(self, params, x, y):
        #print("x_shape_offline: ", x.shape) #(16, 22, 3)
        losses, grads = self.batch_calculate_grads(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)

        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)

    def update_bptt(self, params, x, y):
        print("x_shape_training_0: ", x.shape) #(16, 22, 3)
        loss, grads = self.batch_calculate_grads_BPTT(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)

        return self.get_params(self.opt_state), np.sum(loss, axis=0) / x.shape[0]

    @partial(jit, static_argnums=(0,))
    def forward_step_BPTT(self, paramsData, x, t, h, o):
        h = self.gru(paramsData[:self.Rh.end,], x[t], h)
        #print("V.toDense(paramsData[self.Rz.end:,]).shape: ", self.V.toDense(paramsData[self.Rz.end:,]).shape)
        #print("h: ", h.shape)
        '''if h.ndim == 2 and h.shape[1] != 1:  # h가 2차원이고 마지막 차원이 1이 아닌 경우
            h = h.reshape(-1)
        print('after:', h.shape)'''

        output = self.activation(np.dot(self.V.toDense(paramsData[self.Rh.end:,]), h))
        o = o.at[t].set(output)

        '''V_dense = self.V.toDense(paramsData[self.Rz.end:,])  # Assume V is now (output_size, hidden_size)

        # Dot product over the last dimension of h and the second dimension of V
        output = self.activation(jnp.dot(h, V_dense.T))  # jnp.dot should be used for JAX arrays

        # Set output for all batches
        o = o.at[t].set(output)'''
        return h, o
        
    def calculate_loss_BPTT(self, params, x, y):

        #print("x_shape_training_2: ", x.shape)
        output = self.forward(params, x)
        #print("output: ", output.shape)
        loss = self.lossFunction(output, y)
        #print("loss: ", loss.shape)
        return loss

    def calculate_grads_BPTT(self, params, x, y):
        #print("x2: ", x.shape)
        #print("x_shape_training_1: ", x.shape)
        return value_and_grad(self.calculate_loss_BPTT)(params, x, y)

    batch_calculate_grads_BPTT = vmap(calculate_grads_BPTT, in_axes=(None, None, 0, 0))

    def forward(self, params, x):

        '''if x.shape[0] == 1:
            x = np.squeeze(x, axis=0)'''

        h = np.zeros(self.hidden_size)
        o = np.zeros((x.shape[0], self.output_size))
        #print("h: ", h.shape)
        #print("x: ", x.shape)
        #print("x[0]: ", x.shape[0])

        for t in range(x.shape[0]): #(seq_length, input_size)
            h, o = self.forward_step_BPTT(params, x, t, h, o) #bptt는 self.forward_step_BPTT로 바꾸기, 리턴값에서 o만 씀
            
        return o

    '''def predict(self, x):
        print("x_shape_validation: ", x.shape)
        batch_forward = vmap(self.forward, in_axes=(None, 0))
    
        # vmap을 사용하여 전체 배치에 대해 forward 함수를 적용하고 결과를 반환
        return batch_forward(self.paramsData, x)

    def evaluate(self, x_val, y_val):
        # 검증 데이터에 대해 손실을 계산하는 메서드입니다.
        #x_val_reshaped = self.reshape_input(x_val)
        #batched_loss_function = vmap(self.lossFunction, in_axes=(0, 0))
        preds = self.predict(x_val)

        #preds = preds.reshape((-1,))
        val_loss = self.lossFunction(preds, y_val)  # loss_function은 모델에 정의되어야 하는 메서드입니다.
        #print(val_loss.shape)
        #print(val_loss)

        return val_loss #np.sum(val_loss) / x_val.shape[0] '''

    def predict(self, x, y):
        #print("x_shape_validation: ", x.shape)  # Expected: (batch_size, seq_length)
        # Vectorize calculate_loss_BPTT over the batch dimension of x and y:
        val_loss_fn = vmap(self.calculate_loss_BPTT, in_axes=(None, 0, 0))
        # This applies calculate_loss_BPTT for each sample in the batch.
        return val_loss_fn(self.paramsData, x, y)

    def evaluate(self, x_val, y_val):
        # Compute the loss for each sample in the batch:
        losses = self.predict(x_val, y_val)
        # Average over the batch to get a scalar validation loss:
        return np.mean(losses)

    def run(self, epochs, data, validation_data):

        losses = []
        validation_losses = []
        achieved_sequence_lengths = []
        epochs_list = []  # To store epoch numbers

        '''L = 1 #10
        data.maxSeqLength = L
        validation_data.maxSeqLength = L'''

        # Start timing
        start_time = time.time()

        for i, k in zip(range(epochs), random.split(self.key, epochs)):

            x, y = data.getSample(k)
            #print("batch: ", data.batch_size)
            #print("train x: ", x.shape)
            #print("train y: ", y.shape)
            #print("self.paramsData:", self.paramsData.shape) #(4512,)
            self.paramsData, loss = self.update(self.paramsData, x, y)
            #total_loss = np.sum(loss)
            losses.append(loss)

            # Calculate average bits per character here
            #num_characters = y.shape[0] * y.shape[1]  # Total number of characters in the batch
            '''avg_loss_per_char = loss / y.shape[1]
            avg_bits_per_char = avg_loss_per_char / np.log(2)
            print("avg_loss_per_char:", avg_loss_per_char)
            print("avg_bits_per_char:", avg_bits_per_char)'''

            '''avg_bits_per_char = loss / np.log(2)

            print("avg_bits_per_char:", avg_bits_per_char)'''

            '''# Check if the condition to increase L is met
            if loss < 0.15:
                data.maxSeqLength += 1
                validation_data.maxSeqLength += 1
                # Make sure to update the data generation to reflect new maxSeqLength if necessary'''

            if i % 5 == 0:
                print('Epoch', "{:04d}".format(i))
                print('Train Loss ', loss)

                x_val, y_val = validation_data.getSample(k)
                #print("valid x: ", x_val.shape)
                #print("valid x: ", x_val.shape)
                #print("valid y: ", y_val.shape)
                val_loss = self.evaluate(x_val, y_val) 
                validation_losses.append(val_loss)
                print('Validation Loss:', val_loss)

                # Track epoch number
                epochs_list.append(i)

                achieved_length = data.maxSeqLength
                #achieved_sequence_lengths.append(achieved_length)
                print('Achieved sequence length:', achieved_length)

        # End timer
        end_time = time.time()

        # Calculate total training time
        total_training_time = end_time - start_time
        print(f"Total training time: {total_training_time} seconds")

        return self.paramsData, losses, validation_losses, epochs_list, achieved_sequence_lengths, total_training_time
    
