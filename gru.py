from jax import random, vmap, value_and_grad, jit
import jax.numpy as np
from jax.example_libraries import optimizers
from jax.nn import sigmoid
from jax._src.util import partial
from utils import BinaryCrossEntropyLoss, calculateSnApPattern, SparseMatrix, jacrev
import time


class GRU:
    """
    GRU model with BPTT and RTRL training algorithms, including sparse weight representation.
    """

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

        # Initialize model hyperparameters and state variables
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
        self.lossFunction = lossFunction

        self.initialize_weights()  # Initialize sparse weight matrices
        self.jacobian_init_time = 0.0

        # Display configuration details
        print(f"GRU with {algo_type}")
        print(f"Dense GRU params: {3 * hidden_size * (input_size + hidden_size) + hidden_size * output_size}")
        print(f"Sparse GRU params: {len(self.paramsData.flatten())}")
        print(f"Density: {recurrent_density}")

        # Choose algorithm type and initialize Jacobians if needed
        if algo_type == 'rtrl':
            self.jacobian_init_time = self.initialize_jacob(2)
            print("Online Updates!")
            self.update = self.update_online
        elif algo_type == 'snap':
            self.jacobian_init_time = self.initialize_jacob(snap_level)
            self.update = self.update_online if online else self.update_offline
            print("Online Updates!" if online else "Offline Updates!")
        elif algo_type == 'bptt':
            self.update = self.update_bptt

        # Initialize optimizer
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state = self.opt_init(self.paramsData)
        self.opt_update = jit(self.opt_update)

    def initialize_jacob(self, snap_level):
        """
        Initialize sparse Jacobian matrix for gradient computation.
        """
        start_time = time.time()
        print(f"Initializing Jacobian Matrix with SnAP level: {snap_level}")

        # Combine weight and recurrent weight indices for GRU
        weightRows = np.concatenate((self.Wr.rows, self.Wu.rows, self.Wh.rows))
        weightCols = np.concatenate((self.Wr.cols, self.Wu.cols, self.Wh.cols))
        recurrentRows = np.concatenate((self.Rr.rows, self.Ru.rows, self.Rh.rows))
        recurrentCols = np.concatenate((self.Rr.cols, self.Ru.cols, self.Rh.cols))

        # Compute SnAP pattern based on indices
        SnAP_rows, SnAP_cols = calculateSnApPattern(snap_level, weightRows, weightCols, recurrentRows, recurrentCols)
        
        self.J = SparseMatrix()
        self.J.jacobian(SnAP_rows, SnAP_cols, (self.hidden_size, self.Rh.end), 0)

        jacobian_time = time.time() - start_time
        print(f"Jacobian Shape: {self.J.shape}")
        print(f"Jacobian params: {self.J.len}")
        print(f"Jacobian density: {self.J.density}")
        return jacobian_time
 
    def initialize_weights(self):
        """
        Initialize sparse weight matrices for GRU.
        """
        k1, k2, k3, k4, k5, k6, k7 = random.split(self.key, 7)

        # Input weights
        self.Wr = SparseMatrix(k1, self.input_size, self.hidden_size, self.in_out_density, 0)
        Wr_data = self.Wr.init()
        self.Wu = SparseMatrix(k2, self.input_size, self.hidden_size, self.in_out_density, self.Wr.end)
        Wu_data = self.Wu.init()
        self.Wh = SparseMatrix(k3, self.input_size, self.hidden_size, self.in_out_density, self.Wu.end)
        Wh_data = self.Wh.init()

        # Recurrent weights
        self.Rr = SparseMatrix(k4, self.hidden_size, self.hidden_size, self.recurrent_density, self.Wh.end)
        Rr_data = self.Rr.init()
        self.Ru = SparseMatrix(k5, self.hidden_size, self.hidden_size, self.recurrent_density, self.Rr.end)
        Ru_data = self.Ru.init()
        self.Rh = SparseMatrix(k6, self.hidden_size, self.hidden_size, self.recurrent_density, self.Ru.end)
        Rh_data = self.Rh.init()

        # Output weights
        self.V = SparseMatrix(k7, self.output_size, self.hidden_size, self.in_out_density, self.Rh.end)
        V_data = self.V.init()

        # Concatenate all weight data into a single parameter array
        self.paramsData = np.concatenate((Wr_data, Wu_data, Wh_data, Rr_data, Ru_data, Rh_data, V_data))

    @partial(jit, static_argnums=(0,))
    def gru(self, params, x, h):
        """
        Perform a GRU step using sparse weights.
        """
        # Convert sparse matrices to dense format
        wr_dense = self.Wr.toDense(params[self.Wr.start:self.Wr.end])
        wu_dense = self.Wu.toDense(params[self.Wu.start:self.Wu.end])
        wh_dense = self.Wh.toDense(params[self.Wh.start:self.Wh.end])
        rr_dense = self.Rr.toDense(params[self.Rr.start:self.Rr.end])
        ru_dense = self.Ru.toDense(params[self.Ru.start:self.Ru.end])
        rh_dense = self.Rh.toDense(params[self.Rh.start:self.Rh.end])

        # Compute GRU gates and hidden state
        z = sigmoid(np.dot(x, wu_dense) + np.dot(h, ru_dense))  # Update gate
        r = sigmoid(np.dot(x, wr_dense) + np.dot(h, rr_dense))  # Reset gate
        h_tilde = np.tanh(np.dot(x, wh_dense) + np.dot(r * h, rh_dense))  # Candidate hidden state
        h = (1 - z) * h + z * h_tilde  # Final hidden state
        return h

    @partial(jit, static_argnums=(0,))
    def forward_step(self, params, x, h, Jh_data):
        """
        Perform a single forward step and update Jacobian for RTRL.
        """
        # Compute gradients of GRU outputs w.r.t parameters and hidden states
        (grad_h_params, grad_h_h), h = jacrev(self.gru, argnums=(0, 2))(params, x, h)

        # Update Jacobian using sparse representation
        h_Jh = np.dot(grad_h_h, self.J.toDense(Jh_data))[tuple(self.J.coords)]
        Jh = grad_h_params[tuple(self.J.coords)] + h_Jh
        return h, Jh
        
    @partial(jit, static_argnums=(0,))
    def calculate_loss(self, params, h, y):
        """
        Calculate the loss for a single timestep.
        """
        output = self.activation(np.dot(self.V.toDense(params), h))
        loss = self.lossFunction(output, y)
        return loss
    
    @partial(jit, static_argnums=(0,))
    def combineGradients(self, grad_h, grad_out_params, Jh_data):
        """
        Combine gradients for recurrent and output parameters.
        """
        grad_rec_params = np.dot(grad_h, self.J.toDense(Jh_data))
        return np.concatenate((grad_rec_params, grad_out_params))
    
    @partial(jit, static_argnums=(0,))
    def calculate_grads_step(self, params, x, y, h, Jh_data):
        """
        Calculate gradients for a single timestep.
        """
        h, Jh_data = self.forward_step(params[:self.Rh.end], x, h, Jh_data)
        loss, (grad_out_params, grad_h) = value_and_grad(self.calculate_loss, argnums=(0, 1))(params[self.Rh.end:], h, y)
        gradient = self.combineGradients(grad_h, grad_out_params, Jh_data)
        return loss, gradient, h, Jh_data

    batch_calculate_grads_step = vmap(calculate_grads_step, in_axes=(None, None, 0, 0, 0))

    def calculate_grads(self, params, x, y):
        """
        Calculate gradients over the entire sequence.
        """
        h = np.zeros(self.hidden_size)
        Jh_data = np.zeros(self.J.len)
        losses = []
        gradients = np.zeros_like(self.paramsData)

        for t in range(x.shape[0]):
            loss, gradient, h, Jh_data = self.calculate_grads_step(params, x[t], y[t], h, Jh_data)
            losses.append(loss)
            gradients += gradient / x.shape[0]
            
        return np.mean(np.array(losses)), gradients

    batch_calculate_grads = vmap(calculate_grads, in_axes=(None, None, 0, 0))

    def update_online(self, params, x, y):
        """
        Update parameters using online learning.
        """
        h = np.zeros((self.batch_size, self.hidden_size))
        Jh_data = np.zeros((self.batch_size, self.J.len))
        losses = []

        for t in range(x.shape[1]):
            loss, grads, h, Jh_data = self.batch_calculate_grads_step(params, x[:, t, :], y[:, t, :], h, Jh_data)
            losses.append(np.mean(loss, axis=0))
            self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
            params = self.get_params(self.opt_state)

        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)

    def update_bptt(self, params, x, y):
        """
        Update parameters using Backpropagation Through Time (BPTT).
        """
        loss, grads = self.batch_calculate_grads_BPTT(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
        return self.get_params(self.opt_state), np.sum(loss, axis=0) / x.shape[0]

    def update_offline(self, params, x, y):
        """
        Update parameters using offline batch learning.
        """
        losses, grads = self.batch_calculate_grads(params, x, y)
        self.opt_state = self.opt_update(0, np.sum(grads, axis=0), self.opt_state)
        return self.get_params(self.opt_state), np.mean(np.array(losses), axis=0)

    @partial(jit, static_argnums=(0,))
    def forward_step_BPTT(self, paramsData, x, t, h, o):
        """
        Perform a single forward step for BPTT.
        """
        h = self.gru(paramsData[:self.Rh.end], x[t], h)
        output = self.activation(np.dot(self.V.toDense(paramsData[self.Rh.end:]), h))
        o = o.at[t].set(output)
        return h, o
        
    def calculate_loss_BPTT(self, params, x, y):
        """
        Calculate loss for BPTT over the entire sequence.
        """
        output = self.forward(params, x)
        loss = self.lossFunction(output, y)
        return loss

    def calculate_grads_BPTT(self, params, x, y):
        """
        Calculate gradients for BPTT over the entire sequence.
        """
        return value_and_grad(self.calculate_loss_BPTT)(params, x, y)

    batch_calculate_grads_BPTT = vmap(calculate_grads_BPTT, in_axes=(None, None, 0, 0))

    def forward(self, params, x):
        """
        Perform a forward pass over the entire sequence.
        """
        h = np.zeros(self.hidden_size)
        o = np.zeros((x.shape[0], self.output_size))

        for t in range(x.shape[0]):
            h, o = self.forward_step_BPTT(params, x, t, h, o)
            
        return o

    def predict(self, x):
        """
        Predict outputs for a batch of inputs.
        """
        batch_forward = vmap(self.forward, in_axes=(None, 0))
        return batch_forward(self.paramsData, x)

    def evaluate(self, x_val, y_val):
        """
        Evaluate the model on validation data.
        """
        preds = self.predict(x_val)
        val_loss = self.lossFunction(preds, y_val)
        return val_loss

    def run(self, epochs, data, validation_data):
        """
        Run the training loop.
        """
        losses = []
        validation_losses = []
        epochs_list = []

        # Start timing
        start_time = time.time()

        for i, k in zip(range(epochs), random.split(self.key, epochs)):
            x, y = data.getSample(k)
            self.paramsData, loss = self.update(self.paramsData, x, y)
            losses.append(loss)

            if i % 5 == 0:
                print(f'Epoch {i:04d}')
                print(f'Train Loss: {loss}')

                x_val, y_val = validation_data.getSample(k)
                val_loss = self.evaluate(x_val, y_val)
                validation_losses.append(val_loss)
                print(f'Validation Loss: {val_loss}')

                epochs_list.append(i)

        total_training_time = time.time() - start_time
        print(f"Total training time: {total_training_time} seconds")

        return self.paramsData, losses, validation_losses, epochs_list, total_training_time
    
