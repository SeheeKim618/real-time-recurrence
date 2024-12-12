from jax import random, vmap, value_and_grad, jit
import jax.numpy as np
from jax.example_libraries import optimizers
from jax.nn import sigmoid
from functools import partial
import time
from utils import SparseMatrix, jacrev, BinaryCrossEntropyLoss
from lorax2.transform2 import LoraWeight, lora
from lorax2.helpers2 import init_lora

class LSTM_LORA:
    """
    LSTM model with LoRA (Low-Rank Adaptation) and support for BPTT and RTRL algorithms.
    """

    def __init__(self, key, input_size, output_size, hidden_size, batch_size, 
                 recurrent_density, in_out_density, snap_level, lossFunction, 
                 algo_type, rank_constraint_w, rank_constraint_r, frozen_params, 
                 logEvery=1, learning_rate=1e-3):
        """
        Initializes the LSTM_LORA model.
        """
        self.key = key
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.recurrent_density = recurrent_density
        self.in_out_density = in_out_density
        self.logEvery = logEvery
        self.activation = sigmoid
        self.rank_constraint_w = rank_constraint_w  # Rank constraint for input weights
        self.rank_constraint_r = rank_constraint_r  # Rank constraint for recurrent weights
        self.frozen_params = frozen_params
        self.jacobian_init_time = 0.0

        print(f'LSTM with {algo_type}')
        print(f'Dense LSTM params: {4 * hidden_size * (input_size + hidden_size) + hidden_size * output_size}')
        print(f'Density: {recurrent_density}, Rank_w: {rank_constraint_w}, Rank_r: {rank_constraint_r}')
        print(f'Shape of Frozen params: {frozen_params.shape}')

        # Initialize LoRA parameters
        param_tree = self.reshape_params(frozen_params)
        lora_spec = {
            'Wi': rank_constraint_w,
            'Wo': rank_constraint_w,
            'Wf': rank_constraint_w,
            'Wz': rank_constraint_w,
            'Ri': rank_constraint_r,
            'Ro': rank_constraint_r,
            'Rf': rank_constraint_r,
            'Rz': rank_constraint_r,
            'V': 1  # No rank constraint for V
        }
        self.lora_params = init_lora(param_tree=param_tree, spec=lora_spec, rng=random.PRNGKey(0))

        # Initialize Jacobians for RTRL
        self.jacobian_init_time = self.initialize_jacob()
        print('Online Updates!')

        # Optimizer setup
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_update = jit(self.opt_update)

        # Flatten 'a' and 'b' parameters for optimization
        self.param_order = ['Wi', 'Wo', 'Wf', 'Wz', 'Ri', 'Ro', 'Rf', 'Rz', 'V']
        a_params = self.extract_params(self.lora_params, 'a')
        b_params = self.extract_params(self.lora_params, 'b')
        flattened_a_params = self.flatten_params_in_order(a_params, self.param_order)
        flattened_b_params = self.flatten_params_in_order(b_params, self.param_order)

        self.opt_state_a = self.opt_init(flattened_a_params)
        self.opt_state_b = self.opt_init(flattened_b_params)
        self.lossFunction = lossFunction

    def reshape_params(self, params):
        """
        Reshapes flattened parameters into structured weight matrices.
        """
        shapes = {
            'Wi': (self.input_size, self.hidden_size),
            'Wo': (self.input_size, self.hidden_size),
            'Wf': (self.input_size, self.hidden_size),
            'Wz': (self.input_size, self.hidden_size),
            'Ri': (self.hidden_size, self.hidden_size),
            'Ro': (self.hidden_size, self.hidden_size),
            'Rf': (self.hidden_size, self.hidden_size),
            'Rz': (self.hidden_size, self.hidden_size),
            'V': (self.hidden_size, self.output_size)
        }

        offset = 0
        param_tree = {}
        for key, shape in shapes.items():
            size = np.prod(np.array(shape))
            param_tree[key] = params[offset:offset + size].reshape(shape)
            offset += size
        return param_tree

    def initialize_jacob(self):
        """
        Initializes Jacobian matrices for RTRL.
        """
        start_time = time.time()
        total_a_params, total_b_params = 0, 0
        for key in ['Wi', 'Wo', 'Wf', 'Wz', 'Ri', 'Ro', 'Rf', 'Rz']:
            lora_weight = self.lora_params[key]
            total_a_params += np.prod(lora_weight.a.shape)
            total_b_params += np.prod(lora_weight.b.shape)

        self.J_a = SparseMatrix(m=self.hidden_size, n=total_a_params)
        self.J_b = SparseMatrix(m=self.hidden_size, n=total_b_params)
        self.J_a_data = self.J_a.init()
        self.J_b_data = self.J_b.init()

        jacobian_time = time.time() - start_time
        print(f'Jacobian Shapes: J_a: {self.J_a.shape}, J_b: {self.J_b.shape}')
        print(f'Total Jacobian Parameters: {self.J_a.len + self.J_b.len}')
        return jacobian_time

    @partial(jit, static_argnums=(0,))
    def lstm(self, params, x, h, c):
        """
        Performs the forward pass of the LSTM cell.
        """
        inputGate = sigmoid(np.dot(x, params['Wi']) + np.dot(h, params['Ri']))
        outputGate = sigmoid(np.dot(x, params['Wo']) + np.dot(h, params['Ro']))
        forgetGate = sigmoid(np.dot(x, params['Wf']) + np.dot(h, params['Rf']))
        z = np.tanh(np.dot(x, params['Wz']) + np.dot(h, params['Rz']))

        c = forgetGate * c + inputGate * z
        h = outputGate * np.tanh(c)
        return h, c

    def forward_step(self, params, x, h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b):
        """
        Computes forward pass and updates Jacobians for LoRA weights.
        """
        ((grad_h_params, grad_h_h, grad_h_c),
         (grad_c_params, grad_c_h, grad_c_c)), (h, c) = jacrev(self.lora_model, argnums=(0, 2, 3))(params, x, h, c)

        # Compute Jacobian updates
        h_Jh_a = np.dot(grad_h_h, self.J_a.toDense(Jh_data_a))
        h_Jc_a = np.dot(grad_h_c, self.J_a.toDense(Jc_data_a))
        Jh_a = grad_h_params['a'] + h_Jh_a + h_Jc_a

        h_Jh_b = np.dot(grad_h_h, self.J_b.toDense(Jh_data_b))
        h_Jc_b = np.dot(grad_h_c, self.J_b.toDense(Jc_data_b))
        Jh_b = grad_h_params['b'] + h_Jh_b + h_Jc_b

        c_Jh_a = np.dot(grad_c_h, self.J_a.toDense(Jh_data_a))
        c_Jc_a = np.dot(grad_c_c, self.J_a.toDense(Jc_data_a))
        Jc_a = grad_c_params['a'] + c_Jh_a + c_Jc_a

        c_Jh_b = np.dot(grad_c_h, self.J_b.toDense(Jh_data_b))
        c_Jc_b = np.dot(grad_c_c, self.J_b.toDense(Jc_data_b))
        Jc_b = grad_c_params['b'] + c_Jh_b + c_Jc_b

        return h, c, Jh_a, Jh_b, Jc_a, Jc_b

@partial(jit, static_argnums=(0,))
def calculate_loss(self, params, h, y):
    # Materialize the parameters to dense format
    materialized_V = params.materialize()
    
    # Compute the output using the activation function
    output = self.activation(np.dot(materialized_V.T, h))
    
    # Calculate loss using the provided loss function
    loss = self.lossFunction(output, y)
    return loss

@partial(jit, static_argnums=(0,))
def combineGradients_a(self, grad_h, grad_out_params_a, Jh_data_a):
    # Combine gradients for 'a' parameters
    grad_rec_params = np.dot(grad_h, self.J_a.toDense(Jh_data_a))
    grad_out_params_flat = grad_out_params_a.flatten()
    return np.concatenate((grad_rec_params, grad_out_params_flat))

@partial(jit, static_argnums=(0,))
def combineGradients_b(self, grad_h, grad_out_params_b, Jh_data_b):
    # Combine gradients for 'b' parameters
    grad_rec_params = np.dot(grad_h, self.J_b.toDense(Jh_data_b))
    grad_out_params_flat = grad_out_params_b.flatten()
    return np.concatenate((grad_rec_params, grad_out_params_flat))

@partial(jit, static_argnums=(0,))
def calculate_grads_step(self, params, x, y, h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b):
    # Filter parameters to exclude 'V' for certain computations
    filtered_params = {k: v for k, v in params.items() if k != 'V'}

    # Perform forward step
    h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b = self.forward_step(
        filtered_params, x, h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b
    )

    # Compute loss and gradients for 'V' parameters
    loss, (grad_out_params, grad_h) = value_and_grad(self.calculate_loss, argnums=(0, 1))(params['V'], h, y)

    # Separate gradients for 'a' and 'b' components
    grad_out_params_a = grad_out_params.a
    grad_out_params_b = grad_out_params.b

    # Combine gradients for 'a' and 'b'
    gradient_a = self.combineGradients_a(grad_h, grad_out_params_a, Jh_data_a)
    gradient_b = self.combineGradients_b(grad_h, grad_out_params_b, Jh_data_b)

    return loss, gradient_a, gradient_b, h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b

batch_calculate_grads_step = vmap(calculate_grads_step, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0))

@partial(jit, static_argnums=(0,))
def forward_step_BPTT(self, params, x, t, h, c, o):
    # Perform forward step using LoRA model
    filtered_params = {k: v for k, v in params.items() if k != 'V'}
    h, c = self.lora_model(filtered_params, x[t], h, c)

    # Compute output using activation and 'V' parameters
    V_param = params['V']
    materialized_V = V_param.materialize()
    output = self.activation(np.dot(materialized_V.T, h))

    # Update output sequence
    o = o.at[t].set(output)
    return h, c, o

def forward(self, params, x):
    # Initialize hidden, cell states, and output sequence
    h = np.zeros(self.hidden_size)
    c = np.zeros(self.hidden_size)
    o = np.zeros((x.shape[0], self.output_size))

    # Process input sequence step by step
    for t in range(x.shape[0]):
        h, c, o = self.forward_step_BPTT(params, x, t, h, c, o)

    return o

def predict(self, x):
    # Batch-wise prediction using vmap
    batch_forward = vmap(self.forward, in_axes=(None, 0))
    return batch_forward(self.lora_params, x)

def evaluate(self, x_val, y_val):
    # Predict outputs for validation data and calculate loss
    preds = self.predict(x_val)
    val_loss = self.lossFunction(preds, y_val)
    return val_loss

def unflatten_params_from_order(self, flattened_params, param_order, lora_params, ab_key):
    # Convert flattened array back into dictionary of parameters
    new_params = {}
    idx = 0
    for key in param_order:
        shape = lora_params[key].a.shape if ab_key == 'a' else lora_params[key].b.shape
        size = shape[0] * shape[1]
        new_params[key] = flattened_params[idx:idx + size].reshape(shape)
        idx += size
    return new_params

def update_lora_params(self, lora_params, new_flattened_a, new_flattened_b, param_order):
    # Update LoRA parameters with new values for 'a' and 'b'
    new_a_params = self.unflatten_params_from_order(new_flattened_a, param_order, lora_params, 'a')
    new_b_params = self.unflatten_params_from_order(new_flattened_b, param_order, lora_params, 'b')

    for key in param_order:
        lora_params[key] = lora_params[key].update(a=new_a_params[key], b=new_b_params[key])

    return lora_params

def update_fn(self, lora_params, x, y):
    # Initialize states and Jacobians
    h = np.zeros((self.batch_size, self.hidden_size))
    c = np.zeros((self.batch_size, self.hidden_size))
    Jh_data_a = np.zeros((self.batch_size, self.J_a.len))
    Jh_data_b = np.zeros((self.batch_size, self.J_b.len))
    Jc_data_a = np.zeros((self.batch_size, self.J_a.len))
    Jc_data_b = np.zeros((self.batch_size, self.J_b.len))

    losses = []

    for t in range(x.shape[1]):
        # Calculate gradients and update LoRA parameters
        loss, grads_a, grads_b, h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b = self.batch_calculate_grads_step(
            lora_params, x[:, t, :], y[:, t, :], h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b
        )
        losses.append(np.mean(loss, axis=0))

        # Update optimizer states for 'a' and 'b'
        self.opt_state_a = self.opt_update(0, np.sum(grads_a, axis=0), self.opt_state_a)
        self.opt_state_b = self.opt_update(0, np.sum(grads_b, axis=0), self.opt_state_b)

        # Update LoRA parameters
        new_flattened_a = self.get_params(self.opt_state_a)
        new_flattened_b = self.get_params(self.opt_state_b)
        lora_params = self.update_lora_params(lora_params, new_flattened_a, new_flattened_b, self.param_order)

    return np.mean(np.array(losses)), lora_params

def run(self, epochs, data, validation_data):
    # Main training loop with validation and timing
    losses = []
    validation_losses = []
    epochs_list = []

    start_time = time.time()

    for i, k in zip(range(epochs), random.split(self.key, epochs)):
        x, y = data.getSample(k)
        loss, self.lora_params = self.update_fn(self.lora_params, x, y)
        losses.append(loss)

        if i % 5 == 0:
            print('Epoch', "{:04d}".format(i))
            print('Train Loss ', loss)

            x_val, y_val = validation_data.getSample(k)
            val_loss = self.evaluate(x_val, y_val)
            validation_losses.append(val_loss)
            print('Validation Loss:', val_loss)

        epochs_list.append(i)

    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time} seconds")

    return self.lora_params, losses, validation_losses, epochs_list, total_training_time
