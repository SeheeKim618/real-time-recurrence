from jax import random, vmap, value_and_grad, jit, ops
import jax.numpy as np
from jax.example_libraries import optimizers
from jax.nn import sigmoid
from jax._src.util import partial
import jax.numpy as jnp
import numpy as old_np
from scipy.linalg import svd
import time
import os
import matplotlib.pyplot as plt

# Import custom utilities
from utils import BinaryCrossEntropyLoss, calculateSnApPattern, SparseMatrix, SparseMatrix_RTRL, jacrev

class LSTM:
    def __init__(self, key, input_size, output_size, hidden_size, batch_size,
                 recurrent_density, in_out_density, snap_level, lossFunction,
                 algo_type, logEvery=1, learning_rate=1e-3, online=True):
        # Initialize LSTM parameters
        self.key = key
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.recurrent_density = recurrent_density
        self.in_out_density = in_out_density
        self.logEvery = logEvery
        self.online = online
        self.activation = sigmoid
        self.learning_rate = learning_rate

        # Initialize weights and Jacobian
        self.initialize_weights()
        self.jacobian_init_time = 0.0

        print(f'LSTM with {algo_type}')
        print(f'Dense LSTM params: {4 * hidden_size * (input_size + hidden_size) + hidden_size * output_size}')
        print(f'Sparse LSTM params: {len(self.paramsData.flatten())}')
        print(f'Density: {recurrent_density}')

        # Configure algorithm-specific updates
        if algo_type == 'rtrl':
            self.jacobian_init_time = self.initialize_jacob(2)
            print('Online Updates!')
            self.update = self.update_online
        elif algo_type == 'snap':
            self.jacobian_init_time = self.initialize_jacob(snap_level)
            if self.online:
                print('Online Updates!')
                self.update = self.update_online
            else:
                print('Offline Updates!')
                self.update = self.update_offline
        elif algo_type == 'bptt':
            self.update = self.update_bptt

        # Set up optimizer
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state = self.opt_init(self.paramsData)
        self.opt_update = jit(self.opt_update)
        self.lossFunction = lossFunction

    def initialize_jacob(self, snap_level):
        # Initialize the Jacobian matrix for RTRL or SnAp
        start_time = time.time()
        print(f'Initializing Jacobian Matrix at SnAp level {snap_level}')

        # Combine weight and recurrent matrix row and column indices
        weightRows = np.concatenate((self.Wi.rows, self.Wo.rows, self.Wf.rows, self.Wz.rows))
        weightCols = np.concatenate((self.Wi.cols, self.Wo.cols, self.Wf.cols, self.Wz.cols))
        recurrentRows = np.concatenate((self.Ri.rows, self.Ro.rows, self.Rf.rows, self.Rz.rows))
        recurrentCols = np.concatenate((self.Ri.cols, self.Ro.cols, self.Rf.cols, self.Rz.cols))

        # Calculate SnAp sparsity pattern
        SnAP_rows, SnAP_cols = calculateSnApPattern(snap_level, weightRows, weightCols, recurrentRows, recurrentCols)

        # Initialize sparse Jacobian matrix
        self.J = SparseMatrix()
        self.J.jacobian(SnAP_rows, SnAP_cols, (self.hidden_size, self.Rz.end), 0)

        # Print Jacobian stats
        print(f'Jacobian Shape: {self.J.shape}')
        print(f'Jacobian params: {self.J.len}')
        print(f'Jacobian density: {self.J.density}')

        return time.time() - start_time

    def initialize_weights(self):
        # Initialize sparse weight matrices for LSTM gates
        keys = random.split(self.key, 9)
        self.Wi = SparseMatrix(keys[0], self.input_size, self.hidden_size, self.in_out_density, 0)
        self.Wo = SparseMatrix(keys[1], self.input_size, self.hidden_size, self.in_out_density, self.Wi.end)
        self.Wf = SparseMatrix(keys[2], self.input_size, self.hidden_size, self.in_out_density, self.Wo.end)
        self.Wz = SparseMatrix(keys[3], self.input_size, self.hidden_size, self.in_out_density, self.Wf.end)
        self.Ri = SparseMatrix(keys[4], self.hidden_size, self.hidden_size, self.recurrent_density, self.Wz.end)
        self.Ro = SparseMatrix(keys[5], self.hidden_size, self.hidden_size, self.recurrent_density, self.Ri.end)
        self.Rf = SparseMatrix(keys[6], self.hidden_size, self.hidden_size, self.recurrent_density, self.Ro.end)
        self.Rz = SparseMatrix(keys[7], self.hidden_size, self.hidden_size, self.recurrent_density, self.Rf.end)
        self.V = SparseMatrix(keys[8], self.output_size, self.hidden_size, self.in_out_density, self.Rz.end)

        # Flatten weight matrices into a single parameter array
        self.paramsData = np.concatenate([
            self.Wi.init(), self.Wo.init(), self.Wf.init(), self.Wz.init(),
            self.Ri.init(), self.Ro.init(), self.Rf.init(), self.Rz.init(), self.V.init()
        ])

    @partial(jit, static_argnums=(0,))
    def lstm(self, params, x, h, c):
        # Perform a single forward step in the LSTM
        wi_dense = self.Wi.toDense(params[self.Wi.start:self.Wi.end])
        wo_dense = self.Wo.toDense(params[self.Wo.start:self.Wo.end])
        wf_dense = self.Wf.toDense(params[self.Wf.start:self.Wf.end])
        wz_dense = self.Wz.toDense(params[self.Wz.start:self.Wz.end])
        ri_dense = self.Ri.toDense(params[self.Ri.start:self.Ri.end])
        ro_dense = self.Ro.toDense(params[self.Ro.start:self.Ro.end])
        rf_dense = self.Rf.toDense(params[self.Rf.start:self.Rf.end])
        rz_dense = self.Rz.toDense(params[self.Rz.start:self.Rz.end])

        inputGate = sigmoid(np.dot(x, wi_dense) + np.dot(h, ri_dense))
        outputGate = sigmoid(np.dot(x, wo_dense) + np.dot(h, ro_dense))
        forgetGate = sigmoid(np.dot(x, wf_dense) + np.dot(h, rf_dense))
        z = np.tanh(np.dot(x, wz_dense) + np.dot(h, rz_dense))

        c = forgetGate * c + inputGate * z
        h = outputGate * np.tanh(c)
        return h, c
    
    # Forward step for computing the next hidden and cell states and their gradients.
    @partial(jit, static_argnums=(0,))
    def forward_step(self, params, x, h, c, Jh_data, Jc_data):
        # Compute Jacobians of hidden and cell states using `jacrev`
        ((grad_h_params, grad_h_h, grad_h_c), 
         (grad_c_params, grad_c_h, grad_c_c)), (h, c) = jacrev(self.lstm, argnums=(0, 2, 3))(params, x, h, c)
    
        # Compute updated Jacobian entries for hidden state
        h_Jh = np.dot(grad_h_h, self.J.toDense(Jh_data))[tuple(self.J.coords)]
        h_Jc = np.dot(grad_h_c, self.J.toDense(Jc_data))[tuple(self.J.coords)]
        Jh = grad_h_params[tuple(self.J.coords)] + h_Jh + h_Jc
    
        # Compute updated Jacobian entries for cell state
        c_Jh = np.dot(grad_c_h, self.J.toDense(Jh_data))[tuple(self.J.coords)]
        c_Jc = np.dot(grad_c_c, self.J.toDense(Jc_data))[tuple(self.J.coords)]
        Jc = grad_c_params[tuple(self.J.coords)] + c_Jh + c_Jc
    
        return h, c, Jh, Jc
    
    # Calculate loss for a single forward pass
    @partial(jit, static_argnums=(0,))
    def calculate_loss(self, params, h, y):
        # Compute model output
        output = self.activation(np.dot(self.V.toDense(params), h))
        # Compute loss using the defined loss function
        loss = self.lossFunction(output, y)
        return loss
    
    # Combine gradients from recurrent parameters and output layer parameters
    @partial(jit, static_argnums=(0,))
    def combineGradients(self, grad_h, grad_out_params, Jh_data):
        grad_rec_params = np.dot(grad_h, self.J.toDense(Jh_data))
        return np.concatenate((grad_rec_params, grad_out_params))
    
    # Calculate gradients for a single time step
    @partial(jit, static_argnums=(0,))
    def calculate_grads_step(self, params, x, y, h, c, Jh_data, Jc_data):
        # Forward step to compute hidden and cell states and their Jacobians
        h, c, Jh_data, Jc_data = self.forward_step(params[:self.Rz.end,], x, h, c, Jh_data, Jc_data)
    
        # Compute loss and gradients for the output layer
        loss, (grad_out_params, grad_h) = value_and_grad(self.calculate_loss, argnums=(0, 1))(
            params[self.Rz.end:,], h, y
        )
    
        # Combine gradients from the recurrent and output layers
        gradient = self.combineGradients(grad_h, grad_out_params, Jh_data)
        return loss, gradient, h, c, Jh_data, Jc_data
    
    # Batch processing for gradient calculation
    batch_calculate_grads_step = vmap(calculate_grads_step, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
    
    # Compute gradients for the entire sequence
    def calculate_grads(self, params, x, y):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        Jh_data = np.zeros(self.J.len)
        Jc_data = np.zeros(self.J.len)
        losses = []
        gradients = np.zeros_like(self.paramsData)
    
        # Iterate through each time step
        for t in range(x.shape[0]):
            loss, gradient, h, c, Jh_data, Jc_data = self.calculate_grads_step(params, x[t], y[t], h, c, Jh_data, Jc_data)
            losses.append(loss)
            gradients = gradients + gradient / x.shape[0]  # Average the gradients
    
        return np.mean(np.array(losses)), gradients
    
    # Batch gradient calculation across sequences
    batch_calculate_grads = vmap(calculate_grads, in_axes=(None, None, 0, 0))
    
    # Online update for training
    def update_online(self, params, x, y):
        h = np.zeros((self.batch_size, self.hidden_size))
        c = np.zeros((self.batch_size, self.hidden_size))
        Jh_data = np.zeros((self.batch_size, self.J.len))
        Jc_data = np.zeros((self.batch_size, self.J.len))
        losses = []
    
        # Process each time step in the sequence
        for t in range(x.shape[1]):
            loss, grads, h, c, Jh_data, Jc_data = self.batch_calculate_grads_step(params, x[:, t, :], y[:, t, :], h, c, Jh_data, Jc_data)
            losses.append(np.mean(loss, axis=0))
            # Update parameters using optimizer
            self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
            params = self.get_params(self.opt_state)
    
        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)
    
    # Offline update for training (process entire sequence at once)
    def update_offline(self, params, x, y):
        losses, grads = self.batch_calculate_grads(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)
    
    # Run the training loop
    def run(self, epochs, data, validation_data):
        losses = []
        validation_losses = []
        epochs_list = []
        start_time = time.time()
    
        # Iterate through epochs
        for i, k in zip(range(epochs), random.split(self.key, epochs)):
            x, y = data.getSample(k)
            self.paramsData, loss = self.update(self.paramsData, x, y)
            losses.append(loss)
    
            # Evaluate on validation data periodically
            if i % 5 == 0:
                print(f'Epoch {i:04d}, Train Loss: {loss}')
                x_val, y_val = validation_data.getSample(k)
                val_loss = self.evaluate(x_val, y_val)
                validation_losses.append(val_loss)
                print(f'Validation Loss: {val_loss}')
                epochs_list.append(i)
    
        total_training_time = time.time() - start_time
        print(f"Total training time: {total_training_time} seconds")
        return self.paramsData, losses, validation_losses, epochs_list, total_training_time
