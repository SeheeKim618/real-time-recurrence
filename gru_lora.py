from jax import random, vmap, value_and_grad, jit
import jax.numpy as np
from jax.nn import sigmoid
from jax._src.util import partial
from utils import BinaryCrossEntropyLoss, SparseMatrix, jacrev
from lorax2.transform2 import LoraWeight, lora
from lorax2.helpers2 import init_lora
import time


class GRU_LORA:
    """
    GRU model integrated with LoRA (Low-Rank Adaptation) for parameter-efficient training.
    Supports both BPTT and RTRL training algorithms.
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
                 rank_constraint_w, 
                 rank_constraint_r,
                 frozen_params, 
                 logEvery=1, 
                 learning_rate=1e-3): 
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
        self.lossFunction = lossFunction
        self.jacobian_init_time = 0.0

        # Log model configuration
        print(f'GRU with {algo_type}')
        print(f'Dense GRU params: {4 * hidden_size * (input_size + hidden_size) + hidden_size * output_size}')
        print(f'Density: {recurrent_density}')
        print(f'Rank_w: {rank_constraint_w}, Rank_r: {rank_constraint_r}')
        print(f'Shape of Frozen params: {frozen_params.shape}')

        # Initialize parameters
        param_tree = self.reshape_params(frozen_params)

        # Define LoRA specification
        lora_spec = {
            'Wr': rank_constraint_w,
            'Wu': rank_constraint_w,
            'Wh': rank_constraint_w,
            'Rr': rank_constraint_r,
            'Ru': rank_constraint_r,
            'Rh': rank_constraint_r,
            'V': 1  # No rank constraint on 'V'
        }

        # Initialize LoRA parameters
        self.lora_params = init_lora(param_tree=param_tree, spec=lora_spec, rng=random.PRNGKey(0))

        # Print shapes of initialized parameters
        for key, value in self.lora_params.items():
            if isinstance(value, LoraWeight):
                print(f"{key}: w Shape: {value.w.shape}, a Shape: {value.a.shape}, b Shape: {value.b.shape}")

        # Initialize optimizer and Jacobians
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state_a, self.opt_state_b = self.init_optimizer_states()
        self.jacobian_init_time = self.initialize_jacob()
        print('Online Updates!')

        # Wrap model with LoRA
        self.lora_model = lora(self)

    def reshape_params(self, params):
        """Reshape flattened parameter array into a parameter tree."""
        Wr_shape, Wu_shape, Wh_shape = (self.input_size, self.hidden_size), (self.input_size, self.hidden_size), (self.input_size, self.hidden_size)
        Rr_shape, Ru_shape, Rh_shape = (self.hidden_size, self.hidden_size), (self.hidden_size, self.hidden_size), (self.hidden_size, self.hidden_size)
        V_shape = (self.hidden_size, self.output_size)

        sizes = [np.prod(Wr_shape), np.prod(Wu_shape), np.prod(Wh_shape),
                 np.prod(Rr_shape), np.prod(Ru_shape), np.prod(Rh_shape), 
                 np.prod(V_shape)]

        offset = 0
        param_tree = {}
        for name, shape in zip(['Wr', 'Wu', 'Wh', 'Rr', 'Ru', 'Rh', 'V'], 
                               [Wr_shape, Wu_shape, Wh_shape, Rr_shape, Ru_shape, Rh_shape, V_shape]):
            size = np.prod(shape)
            param_tree[name] = params[offset:offset + size].reshape(shape)
            offset += size
        return param_tree

    def initialize_jacob(self):
        """Initialize Jacobian matrices for 'a' and 'b' LoRA parameters."""
        start_time = time.time()
        
        total_a_params, total_b_params = 0, 0
        for key in ['Wr', 'Wu', 'Wh', 'Rr', 'Ru', 'Rh']:
            lora_weight = self.lora_params[key]
            total_a_params += np.prod(lora_weight.a.shape)
            total_b_params += np.prod(lora_weight.b.shape)

        self.J_a = SparseMatrix(m=self.hidden_size, n=total_a_params)
        self.J_b = SparseMatrix(m=self.hidden_size, n=total_b_params)

        self.J_a_data, self.J_b_data = self.J_a.init(), self.J_b.init()

        end_time = time.time()
        print(f'Jacobian initialized in {end_time - start_time:.4f}s')
        print(f'Jacobian Shape a: {self.J_a.shape}, b: {self.J_b.shape}')
        return end_time - start_time

    def init_optimizer_states(self):
        """Initialize optimizer states for 'a' and 'b' parameters."""
        a_params = self.extract_params(self.lora_params, 'a')
        b_params = self.extract_params(self.lora_params, 'b')

        flattened_a = self.flatten_params_in_order(a_params, self.param_order)
        flattened_b = self.flatten_params_in_order(b_params, self.param_order)

        opt_state_a = self.opt_init(flattened_a)
        opt_state_b = self.opt_init(flattened_b)
        return opt_state_a, opt_state_b

    def extract_params(self, params, key):
        """Extract 'a' or 'b' parameters from LoRA parameters."""
        return {k: getattr(v, key) for k, v in params.items() if isinstance(v, LoraWeight)}

    def flatten_params_in_order(self, params_dict, order):
        """Flatten parameters in the specified order."""
        return np.concatenate([params_dict[key].flatten() for key in order])

    @partial(jit, static_argnums=(0,))
    def gru(self, params, x, h):
        """Compute GRU hidden state update."""
        z = sigmoid(np.dot(x, params['Wu']) + np.dot(h, params['Ru']))  # Update gate
        r = sigmoid(np.dot(x, params['Wr']) + np.dot(h, params['Rr']))  # Reset gate
        h_tilde = np.tanh(np.dot(x, params['Wh']) + np.dot(r * h, params['Rh']))  # Candidate state
        return (1 - z) * h + z * h_tilde

    def forward_step(self, params, x, h, Jh_data_a, Jh_data_b):
        """Perform one forward step and update Jacobians."""
        (grad_h_params, grad_h_h), h = jacrev(self.lora_model, argnums=(0, 2))(params, x, h)

        grad_h_flattened_params_a = [grad_h_params[key].a.reshape(32, -1) for key in ['Wr', 'Wu', 'Wh', 'Rr', 'Ru', 'Rh']]
        grad_h_flattened_params_b = [grad_h_params[key].b.reshape(32, -1) for key in ['Wr', 'Wu', 'Wh', 'Rr', 'Ru', 'Rh']]

        grad_h_params_flat_a = jnp.concatenate(grad_h_flattened_params_a, axis=1)
        grad_h_params_flat_b = jnp.concatenate(grad_h_flattened_params_b, axis=1)

        h_Jh_a = np.dot(grad_h_h, self.J_a.toDense(Jh_data_a))[tuple(self.J_a.coords)]
        Jh_a = grad_h_params_flat_a[tuple(self.J_a.coords)] + h_Jh_a 

        h_Jh_b = np.dot(grad_h_h, self.J_b.toDense(Jh_data_b))[tuple(self.J_b.coords)]
        Jh_b = grad_h_params_flat_b[tuple(self.J_b.coords)] + h_Jh_b 

        return h, Jh_a, Jh_b

    @partial(jit, static_argnums=(0,))
    def calculate_loss(self, params, h, y):
        # Materialize 'V' parameter to dense format
        materialized_V = params.materialize()
        # Compute the output
        output = self.activation(np.dot(materialized_V.T, h))
        # Compute loss using the provided loss function
        loss = self.lossFunction(output, y)
        return loss

    @partial(jit, static_argnums=(0,))
    def combineGradients_a(self, grad_h, grad_out_params_a, Jh_data_a):
        # Combine gradients for parameter 'a'
        grad_rec_params = np.dot(grad_h, self.J_a.toDense(Jh_data_a))
        grad_out_params_flat = grad_out_params_a.flatten()
        return np.concatenate((grad_rec_params, grad_out_params_flat))
    
    @partial(jit, static_argnums=(0,))
    def combineGradients_b(self, grad_h, grad_out_params_b, Jh_data_b):
        # Combine gradients for parameter 'b'
        grad_rec_params = np.dot(grad_h, self.J_b.toDense(Jh_data_b))
        grad_out_params_flat = grad_out_params_b.flatten()
        return np.concatenate((grad_rec_params, grad_out_params_flat))

    @partial(jit, static_argnums=(0,))
    def calculate_grads_step(self, params, x, y, h, Jh_data_a, Jh_data_b):
        # Filter parameters to exclude 'V'
        filtered_params = {k: v for k, v in params.items() if k != 'V'}
        # Perform forward step
        h, Jh_data_a, Jh_data_b = self.forward_step(filtered_params, x, h, Jh_data_a, Jh_data_b)

        # Calculate loss and gradients for 'V' parameters
        loss, (grad_out_params, grad_h) = value_and_grad(self.calculate_loss, argnums=(0, 1))(params['V'], h, y)

        # Extract gradients for 'a' and 'b'
        grad_out_params_a = grad_out_params.a
        grad_out_params_b = grad_out_params.b

        # Combine gradients
        gradient_a = self.combineGradients_a(grad_h, grad_out_params_a, Jh_data_a)
        gradient_b = self.combineGradients_b(grad_h, grad_out_params_b, Jh_data_b)
            
        return loss, gradient_a, gradient_b, h, Jh_data_a, Jh_data_b

    batch_calculate_grads_step = vmap(calculate_grads_step, in_axes=(None, None, 0, 0, 0, 0, 0))

    @partial(jit, static_argnums=(0,))
    def forward_step_BPTT(self, params, x, t, h, o):
        # Perform one forward step for BPTT
        filtered_params = {k: v for k, v in params.items() if k != 'V'}
        h = self.lora_model(filtered_params, x[t], h)

        # Compute output
        V_param = params['V']
        materialized_V = V_param.materialize()
        output = self.activation(np.dot(materialized_V.T, h))

        # Update output sequence
        o = o.at[t].set(output)
        return h, o

    def forward(self, params, x):
        # Initialize hidden state and output sequence
        h = np.zeros(self.hidden_size)
        o = np.zeros((x.shape[0], self.output_size))

        # Perform forward pass over the entire sequence
        for t in range(x.shape[0]):
            h, o = self.forward_step_BPTT(params, x, t, h, o)
            
        return o

    def predict(self, x):
        # Perform batched forward pass
        batch_forward = vmap(self.forward, in_axes=(None, 0))
        return batch_forward(self.lora_params, x)
    
    def evaluate(self, x_val, y_val):
        # Predict on validation data
        preds = self.predict(x_val)
        # Compute validation loss
        val_loss = self.lossFunction(preds, y_val)
        return val_loss
    
    def unflatten_params_from_order(self, flattened_params, param_order, lora_params, ab_key):
        """
        Converts a flattened array back into a dictionary of LoraWeight parameters.

        Args:
            flattened_params: Flattened numpy or JAX array.
            param_order: List of parameter names in order.
            lora_params: Original LoRA parameters for shape reference.
            ab_key: 'a' or 'b' to indicate which part of the LoRA weight to update.
        """
        new_params = {}
        idx = 0
        for key in param_order:
            shape = lora_params[key].a.shape if ab_key == 'a' else lora_params[key].b.shape
            size = shape[0] * shape[1]
            new_params[key] = flattened_params[idx:idx + size].reshape(shape)
            idx += size
        return new_params
    
    def update_lora_params(self, lora_params, new_flattened_a, new_flattened_b, param_order):
        """
        Updates LoRA parameters with new values for 'a' and 'b' after optimization.

        Args:
            lora_params: Dictionary of LoraWeight objects.
            new_flattened_a: Flattened updated values for 'a'.
            new_flattened_b: Flattened updated values for 'b'.
            param_order: List of parameter names in order.
        """
        # Unflatten updated parameters
        new_a_params = self.unflatten_params_from_order(new_flattened_a, param_order, lora_params, 'a')
        new_b_params = self.unflatten_params_from_order(new_flattened_b, param_order, lora_params, 'b')

        # Update LoRA parameters
        for key in param_order:
            lora_params[key] = lora_params[key].update(a=new_a_params[key], b=new_b_params[key])

        return lora_params
    
    def update_fn(self, lora_params, x, y):
        # Initialize hidden state and Jacobians
        h = np.zeros((self.batch_size, self.hidden_size))
        Jh_data_a = np.zeros((self.batch_size, self.J_a.len))
        Jh_data_b = np.zeros((self.batch_size, self.J_b.len))

        losses = []

        for t in range(x.shape[1]):  # Iterate over time steps
            # Calculate gradients and update parameters
            loss, grads_a, grads_b, h, Jh_data_a, Jh_data_b = self.batch_calculate_grads_step(
                lora_params, x[:, t, :], y[:, t, :], h, Jh_data_a, Jh_data_b
            )
            losses.append(np.mean(loss, axis=0))

            # Update optimizer states for 'a' and 'b'
            self.opt_state_a = self.opt_update(0, np.sum(grads_a, axis=0), self.opt_state_a)
            self.opt_state_b = self.opt_update(0, np.sum(grads_b, axis=0), self.opt_state_b)

            # Extract updated parameters
            new_flattened_a = self.get_params(self.opt_state_a)
            new_flattened_b = self.get_params(self.opt_state_b)

            # Update LoRA parameters
            lora_params = self.update_lora_params(lora_params, new_flattened_a, new_flattened_b, self.param_order)

        return np.mean(np.array(losses)), lora_params
    
    def run(self, epochs, data, validation_data):
        # Run the training loop
        losses = []
        validation_losses = []
        epochs_list = []

        # Start timing
        start_time = time.time()
        
        for i, k in zip(range(epochs), random.split(self.key, epochs)):
            # Get training data sample
            x, y = data.getSample(k)
            # Update model parameters
            loss, self.lora_params = self.update_fn(self.lora_params, x, y)
            losses.append(loss)

            if i % 5 == 0:  # Log every 5 epochs
                print(f'Epoch {i:04d}')
                print(f'Train Loss: {loss}')
                # Evaluate on validation data
                x_val, y_val = validation_data.getSample(k)
                val_loss = self.evaluate(x_val, y_val)
                validation_losses.append(val_loss)
                print(f'Validation Loss: {val_loss}')

        # End timing
        total_training_time = time.time() - start_time
        print(f"Total training time: {total_training_time:.2f} seconds")

        return self.lora_params, losses, validation_losses, total_training_time
