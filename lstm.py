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
class LSTM:

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

        print('LSTM with '+ algo_type)
        print('Dense LSTM params: ', (4*hidden_size*(input_size+hidden_size) + hidden_size*output_size))
        print('Sparse LSTM params: ', len(self.paramsData.flatten()))
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

        weightRows = np.concatenate((self.Wi.rows, self.Wo.rows, self.Wf.rows, self.Wz.rows))
        weightCols = np.concatenate((self.Wi.cols, self.Wo.cols, self.Wf.cols, self.Wz.cols))
        
        recurrentRows= np.concatenate((self.Ri.rows, self.Ro.rows, self.Rf.rows, self.Rz.rows))
        recurrentCols = np.concatenate((self.Ri.cols, self.Ro.cols, self.Rf.cols, self.Rz.cols))

        SnAP_rows, SnAP_cols  = calculateSnApPattern(snap_level, weightRows, weightCols, recurrentRows, recurrentCols)
        
        self.J = SparseMatrix()
        #print("shape of SnAP_rows: ", SnAP_rows.shape) #(143360,)
        #print("shape of SnAP_cols: ", SnAP_cols.shape) #(143360,)
        self.J.jacobian(SnAP_rows, SnAP_cols, (self.hidden_size, self.Rz.end), 0)

        # End timing
        end_time = time.time()
        
        # Calculate elapsed time
        jacobian_time = end_time - start_time

        print('Jacobian Shape: ', self.J.shape) #(32, 4480)
        print('Jacobian params: ', self.J.len)
        print('Jacobian density: ', self.J.density)

        return jacobian_time
 
    def initialize_weights(self):
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = random.split(self.key, 9)

        self.Wi = SparseMatrix(k1, self.input_size, self.hidden_size, self.in_out_density, 0)
        Wi_data = self.Wi.init()
        self.Wo = SparseMatrix(k2, self.input_size, self.hidden_size, self.in_out_density, self.Wi.end)
        Wo_data = self.Wo.init()
        self.Wf = SparseMatrix(k3, self.input_size, self.hidden_size, self.in_out_density, self.Wo.end)
        Wf_data = self.Wf.init()
        self.Wz = SparseMatrix(k4, self.input_size, self.hidden_size, self.in_out_density, self.Wf.end)
        Wz_data = self.Wz.init()

        self.Ri = SparseMatrix(k5, self.hidden_size, self.hidden_size, self.recurrent_density, self.Wz.end)
        Ri_data = self.Ri.init()
        self.Ro = SparseMatrix(k6, self.hidden_size, self.hidden_size, self.recurrent_density, self.Ri.end)
        Ro_data = self.Ro.init()
        self.Rf = SparseMatrix(k7, self.hidden_size, self.hidden_size, self.recurrent_density, self.Ro.end)
        Rf_data = self.Rf.init()
        self.Rz = SparseMatrix(k8, self.hidden_size, self.hidden_size, self.recurrent_density, self.Rf.end)
        Rz_data = self.Rz.init()

        self.V = SparseMatrix(k9, self.output_size, self.hidden_size, self.in_out_density, self.Rz.end)
        V_data = self.V.init()
        
        self.paramsData = np.concatenate((Wi_data, Wo_data, Wf_data, Wz_data, Ri_data, Ro_data, Rf_data, Rz_data, V_data))


    @partial(jit, static_argnums=(0,))
    def lstm(self, params, x, h, c):

        #print("Wi: ", params[self.Wi.start:self.Wi.end,].shape)
        #print("Ri: ", params[self.Ri.start:self.Ri.end,].shape)

        #print(f"Ri start: {self.Ri.start}, Ri end: {self.Ri.end}")
        #print("length of params In lstm", len(params))

        # Convert materialized params to dense if necessary
        wi_dense = self.Wi.toDense(params[self.Wi.start:self.Wi.end,])
        wo_dense = self.Wo.toDense(params[self.Wo.start:self.Wo.end,])
        wf_dense = self.Wf.toDense(params[self.Wf.start:self.Wf.end,])
        wz_dense = self.Wz.toDense(params[self.Wz.start:self.Wz.end,])
        ri_dense = self.Ri.toDense(params[self.Ri.start:self.Ri.end,])
        ro_dense = self.Ro.toDense(params[self.Ro.start:self.Ro.end,])
        rf_dense = self.Rf.toDense(params[self.Rf.start:self.Rf.end,])
        rz_dense = self.Rz.toDense(params[self.Rz.start:self.Rz.end,])
        #print("Shape of wi_dense:", wi_dense.shape)
        #print("Shape of ri_dense:", ri_dense.shape)

        # Use the dense matrices in the LSTM computations
        inputGate = sigmoid(np.dot(x, wi_dense) + np.dot(h, ri_dense))
        outputGate = sigmoid(np.dot(x, wo_dense) + np.dot(h, ro_dense))
        forgetGate = sigmoid(np.dot(x, wf_dense) + np.dot(h, rf_dense))

        # Cell Input
        #print("Wz: ", params[self.Wz.start:self.Wz.end,].shape)
        z = np.tanh(np.dot(x, self.Wz.toDense(params[self.Wz.start:self.Wz.end,])) + np.dot(h, self.Rz.toDense(params[self.Rz.start:self.Rz.end,])))

        # Cell State
        c = forgetGate * c + inputGate * z

        # Cell Output
        h = outputGate * np.tanh(c)

        return h, c
    
    @partial(jit, static_argnums=(0,))
    def forward_step(self, params, x, h, c, Jh_data, Jc_data):

        #print("length of params In forward", len(params)) #4480
        #print("params: ", params) #4480

        ###########
        
        ((grad_h_params, grad_h_h, grad_h_c), (grad_c_params, grad_c_h, grad_c_c)), (h, c) = jacrev(self.lstm, argnums=(0,2,3))(params, x, h, c)

        #print("type of grad_h_params: ", type(grad_h_params)) #DynamicJaxprTracer
        #print("shape of grad_h_params: ", grad_h_params.shape) #(32,4480)
        #print("grad_h_params: ", grad_h_params) #[32, 4480]

        #print("Jh_data: ", Jh_data.shape) #(143360,)
        #print("self.J.toDense(Jh_data): ", self.J.toDense(Jh_data).shape) #(32, 4480)
        #print("type of self.J: ", type(self.J)) #class 'utils.SparseMatrix'
        #print("shape of self.J: ", self.J.shape) #(32, 4480)
        #print("shape of self.J.coords: ", len(self.J.coords)) #2

        h_Jh = np.dot(grad_h_h, self.J.toDense(Jh_data))[tuple(self.J.coords)]
        h_Jc = np.dot(grad_h_c, self.J.toDense(Jc_data))[tuple(self.J.coords)]
        Jh = grad_h_params[tuple(self.J.coords)] + h_Jh + h_Jc

        c_Jh = np.dot(grad_c_h, self.J.toDense(Jh_data))[tuple(self.J.coords)]
        c_Jc = np.dot(grad_c_c, self.J.toDense(Jc_data))[tuple(self.J.coords)]
        Jc = grad_c_params[tuple(self.J.coords)] + c_Jh + c_Jc
        ############

        return h, c, Jh, Jc

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

        #print("shape of grad_h", grad_h.shape) #(32,)
        grad_rec_params = np.dot(grad_h, self.J.toDense(Jh_data))
        #print("shape of grad_rec_params", grad_rec_params.shape)
        #print("shape of grad_out_params", grad_out_params.shape)
        
        return np.concatenate((grad_rec_params, grad_out_params))


    @partial(jit, static_argnums=(0,))
    def calculate_grads_step(self, params, x, y, h, c, Jh_data, Jc_data):

        #print("length of params In cal grads", len(params[:self.Rz.end,]))
        h, c, Jh_data, Jc_data = self.forward_step(params[:self.Rz.end,], x, h, c, Jh_data, Jc_data)
        #self.debug_forward_step(params[:self.Rz.end,], x, h, c, Jh_data, Jc_data)

        loss, (grad_out_params, grad_h) = value_and_grad(self.calculate_loss, argnums=(0,1))(params[self.Rz.end:,], h, y) # calculation of gradients 
        gradient = self.combineGradients(grad_h, grad_out_params, Jh_data)
            
        return loss, gradient, h, c, Jh_data, Jc_data

    batch_calculate_grads_step = vmap(calculate_grads_step, in_axes=(None, None, 0, 0, 0, 0, 0, 0))

    def calculate_grads(self, params, x, y):
        
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        Jh_data = np.zeros(self.J.len)
        Jc_data = np.zeros(self.J.len)

        losses = []
        gradients = np.zeros_like(self.paramsData)

        for t in range(x.shape[0]):
            loss, gradient, h, c, Jh_data, Jc_data = self.calculate_grads_step(params, x[t], y[t], h, c, Jh_data, Jc_data)
            losses.append(loss)
            gradients = gradients + gradient/x.shape[0]
            
        return np.mean(np.array(losses)), gradients

    batch_calculate_grads = vmap(calculate_grads, in_axes=(None, None, 0, 0))

    def update_online(self, params, x, y):
        h = np.zeros((self.batch_size, self.hidden_size))
        c = np.zeros((self.batch_size, self.hidden_size))

        Jh_data = np.zeros((self.batch_size, self.J.len))
        Jc_data = np.zeros((self.batch_size, self.J.len))

        losses = []

        for t in range(x.shape[1]):
            loss, grads, h, c, Jh_data, Jc_data = self.batch_calculate_grads_step(params, x[:,t,:], y[:,t,:], h, c, Jh_data, Jc_data)
            losses.append(np.mean(loss, axis=0))
            #print("Type of grads:", type(grads)) #ArrayImpl
            #print("Shape of grads:", grads.shape) #(16, 4512) -> 72192
            self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
            params = self.get_params(self.opt_state)

        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)

    def update_bptt(self, params, x, y):
        #print("x1: ", x.shape)
        loss, grads = self.batch_calculate_grads_BPTT(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
        #print("train loss: ", loss)
        return self.get_params(self.opt_state), np.sum(loss, axis=0) / x.shape[0]

    def update_offline(self, params, x, y):
        losses, grads = self.batch_calculate_grads(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)

        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)

    @partial(jit, static_argnums=(0,))
    def forward_step_BPTT(self, paramsData, x, t, h, c, o):
        h, c = self.lstm(paramsData[:self.Rz.end,], x[t], h, c)

        output = self.activation(np.dot(self.V.toDense(paramsData[self.Rz.end:,]), h))
        o = o.at[t].set(output)
        return h, c, o

    def calculate_loss_BPTT(self, params, x, y):
        #print("x3: ", x.shape)
        output = self.forward(params, x)
        loss = self.lossFunction(output, y)
        return loss

    def calculate_grads_BPTT(self, params, x, y):
        #print("x2: ", x.shape)
        return value_and_grad(self.calculate_loss_BPTT)(params, x, y)

    batch_calculate_grads_BPTT = vmap(calculate_grads_BPTT, in_axes=(None, None, 0, 0))

    def forward(self, params, x):

        '''if x.shape[0] == 1:
            x = np.squeeze(x, axis=0)'''

        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        o = np.zeros((x.shape[0], self.output_size))
        #print("h: ", h.shape)
        #print("x: ", x.shape)
        #print("x[0]: ", x.shape[0])

        for t in range(x.shape[0]):
            h, c, o = self.forward_step_BPTT(params, x, t, h, c, o) #bptt는 self.forward_step_BPTT로 바꾸기, 리턴값에서 o만 씀
            
        return o

    def predict(self, x):
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

        return val_loss #np.sum(val_loss) / x_val.shape[0] 

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
                val_loss = self.evaluate(x_val, y_val) 
                validation_losses.append(val_loss)
                print('Validation Loss:', val_loss)

                # Track epoch number
                epochs_list.append(i)

                '''achieved_length = data.maxSeqLength
                achieved_sequence_lengths.append(achieved_length)
                print('Achieved sequence length:', achieved_length)'''

        # End timer
        end_time = time.time()

        # Calculate total training time
        total_training_time = end_time - start_time
        print(f"Total training time: {total_training_time} seconds")

        return self.paramsData, losses, validation_losses, epochs_list, achieved_sequence_lengths, total_training_time
    
