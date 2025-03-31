import jax
from jax import random, vmap, value_and_grad, jit, ops
import jax.numpy as np
import jax.random as jrandom
from jax.example_libraries import optimizers
from jax.nn import sigmoid, softmax
from jax._src.util import partial
import jax.numpy as jnp
import numpy as old_np
from scipy.linalg import svd
import time
#from sophia import SophiaG
#import lorax
from utils import CrossEntropyLoss, CrossEntropyLoss_RTRL, one_hot_encoding, calculateSnApPattern, SparseMatrix, jacrev

import os
import matplotlib.pyplot as plt

#from lorax.transform import LoraWeight
from lorax2.transform2 import LoraWeight, lora
from lorax2.helpers2 import init_lora

"""
LSTM model with BPTT and RTRL training algorithm.
"""
class GRU_LORA:

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
                 embedding_dim, 
                 rank_constraint_w, 
                 rank_constraint_r,
                 frozen_params, 
                 logEvery=1, 
                 learning_rate=1e-3): 

        self.key = key
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size= batch_size
        self.recurrent_density = recurrent_density
        self.in_out_density = in_out_density
        self.logEvery = logEvery
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

        # Define different rank constraints for each parameter
        self.rank_constraint_w = rank_constraint_w  # Example rank for input weights
        self.rank_constraint_r = rank_constraint_r  # Example rank for recurrent weights
        self.frozen_params = frozen_params

        self.jacobian_init_time = 0.0

        print('GRU with '+ self.algo_type)
        print('Dense GRU params: ', (3*self.hidden_size*(self.embedding_dim+self.hidden_size) + self.hidden_size*self.output_size))
        #print('Sparse LSTM params: ', len(self.paramsData.flatten()))
        print('Density: ', self.recurrent_density)
        print('Rank_w: ', self.rank_constraint_w)
        print('Rank_r: ', self.rank_constraint_r)
        print('Shape of Frozen params: ', self.frozen_params.shape)
        print('RTRL Jacobian total param: ', (frozen_params.shape[0]-(hidden_size*self.vocab_size))*hidden_size)

        #self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        #self.opt_init, self.opt_update, self.get_params = SophiaG(learning_rate)

        #self.opt_state = self.opt_init(self.paramsData)

        param_tree = self.reshape_params(self.frozen_params)

        lora_spec = {
            'Wr': self.rank_constraint_w,  # Rank for input weight 'Wi'
            'Wu': self.rank_constraint_w,  # Rank for input weight 'Wo'
            'Wh': self.rank_constraint_w,  # Rank for input weight 'Wf'
            'Rr': self.rank_constraint_r,  # Rank for recurrent weight 'Ri'
            'Ru': self.rank_constraint_r,  # Rank for recurrent weight 'Ro'
            'Rh': self.rank_constraint_r,  # Rank for recurrent weight 'Rf'
            'V': 1,   # Assuming 'V' is an input weight V는 그냥 rank 적용 안해야하는거 아닌가???
        }

        # Initialize LoRA parameters from the pre-trained weights
        self.lora_params = init_lora(param_tree=param_tree, spec=lora_spec, rng=random.PRNGKey(0)) 
        #여기서 w에 트레인된거 잘 들어갔나 확인

        #self.lora_params = self.init_lora_params(param_tree=param_tree, spec=lora_spec, rng=random.PRNGKey(0))
        #print("type of params in init: ", type(self.lora_params))

        self.jacobian_init_time = self.initialize_jacob()
        print('Online Updates!')
            
        for key, value in self.lora_params.items():
            print(f"Key: {key}")

            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    print(f"  Subkey: {subkey}, Shape: {subvalue.shape}")
            elif isinstance(value, LoraWeight):  # If using LoRA, check the shapes of w, a, b matrices
                print(f"  w Shape: {value.w.shape}, a Shape: {value.a.shape}, b Shape: {value.b.shape}")
        
        '''# Wrap the optimizer with LoRA
        self.optimizer = optax.adam(learning_rate)
        self.optimizer = wrap_optimizer(self.optimizer, lora_spec)'''

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(self.learning_rate, b1=0.9, b2=0.999, eps=1e-8)

        # Define the order of the keys from Wi to V
        self.param_order = ['Wr', 'Wu', 'Wh', 'Rr', 'Ru', 'Rh', 'V']

        # Extract 'a' and 'b' parameters
        a_params = self.extract_params(self.lora_params, 'a')
        b_params = self.extract_params(self.lora_params, 'b')

        # Flatten the 'a' and 'b' parameter dictionaries in the specified order
        flattened_a_params = self.flatten_params_in_order(a_params, self.param_order)
        flattened_b_params = self.flatten_params_in_order(b_params, self.param_order)

        # Print shapes to verify
        #print("Shape of flattened_a_params:", flattened_a_params.shape)  # (2056,)
        #print("Shape of flattened_b_params:", flattened_b_params.shape)  # (1376,)

        # Initialize optimizer state with flattened parameters
        self.opt_state_a = self.opt_init(flattened_a_params)
        self.opt_state_b = self.opt_init(flattened_b_params)

        #print("type of self.opt_state_a: ", type(self.opt_state_a))

        self.opt_update = jit(self.opt_update)

        # Initialize optimizer states for `a` and `b` separately
        '''self.opt_state_a = self.optimizer.init({key: value.a for key, value in self.lora_params.items() if isinstance(value, LoraWeight)})
        self.opt_state_b = self.optimizer.init({key: value.b for key, value in self.lora_params.items() if isinstance(value, LoraWeight)})'''

        # Create and store the wrapped model as self.lora_model
        self.lora_model = lora(self)
    
    def extract_params(self, params, key):
        return {k: getattr(v, key) for k, v in params.items() if isinstance(v, LoraWeight)}
    
    def flatten_params_in_order(self, params_dict, order):
        flattened_params = []
        for key in order:
            value = params_dict[key]
            flattened_params.append(value.flatten())  # Flatten each parameter
        return np.concatenate(flattened_params)

    def reshape_params(self, params):
        Wr_shape = (self.embedding_dim, self.hidden_size)
        Wu_shape = (self.embedding_dim, self.hidden_size)
        Wh_shape = (self.embedding_dim, self.hidden_size)
        Rr_shape = (self.hidden_size, self.hidden_size)
        Ru_shape = (self.hidden_size, self.hidden_size)
        Rh_shape = (self.hidden_size, self.hidden_size)
        V_shape = (self.hidden_size, self.output_size)

        # The sizes of each parameter
        sizes = [np.prod(np.array(Wr_shape)), np.prod(np.array(Wu_shape)), np.prod(np.array(Wh_shape)),
                 np.prod(np.array(Rr_shape)), np.prod(np.array(Ru_shape)), np.prod(np.array(Rh_shape)), 
                 np.prod(np.array(V_shape))]

        # Start slicing from the flattened array
        offset = 0
        Wr_data = params[offset:offset + sizes[0]].reshape(Wr_shape)
        offset += sizes[0]
        Wu_data = params[offset:offset + sizes[1]].reshape(Wu_shape)
        offset += sizes[1]
        Wh_data = params[offset:offset + sizes[2]].reshape(Wh_shape)
        offset += sizes[2]
        Rr_data = params[offset:offset + sizes[3]].reshape(Rr_shape)
        offset += sizes[3]
        Ru_data = params[offset:offset + sizes[4]].reshape(Ru_shape)
        offset += sizes[4]
        Rh_data = params[offset:offset + sizes[5]].reshape(Rh_shape)
        offset += sizes[5]
        V_data = params[offset:offset + sizes[6]].reshape(V_shape)

        param_tree = {
            'Wr': Wr_data,
            'Wu': Wu_data,
            'Wh': Wh_data,
            'Rr': Rr_data,
            'Ru': Ru_data,
            'Rh': Rh_data,
            'V': V_data,
        }

        return param_tree

    def initialize_jacob(self):
        start_time = time.time()

        # Calculate the total number of parameters for 'a' and 'b' separately from Wi to Rz
        total_a_params = 0
        total_b_params = 0
        for key in ['Wr', 'Wu', 'Wh', 'Rr', 'Ru', 'Rh']:
            lora_weight = self.lora_params[key]
            a_shape = lora_weight.a.shape  # Get the shape of the 'a' matrix
            b_shape = lora_weight.b.shape  # Get the shape of the 'b' matrix
            total_a_params += a_shape[0] * a_shape[1]  # Add the number of elements in the 'a' matrix
            total_b_params += b_shape[0] * b_shape[1]  # Add the number of elements in the 'b' matrix

        # Initialize the Jacobian matrices for 'a' and 'b' as SparseMatrix
        self.J_a = SparseMatrix(m=self.hidden_size, n=total_a_params)
        self.J_b = SparseMatrix(m=self.hidden_size, n=total_b_params)

        # Initialize the sparse matrices with random or specific data if necessary
        #self.J_a_data = self.J_a.init()
        #self.J_b_data = self.J_b.init()

        # End timing
        end_time = time.time()
        
        # Calculate elapsed time
        jacobian_time = end_time - start_time

        print('Jacobian Shape for a: ', self.J_a.shape) #(32, 2048)
        print('Jacobian Shape for b: ', self.J_b.shape) #(32, 1120)
        print('Jacobian params for a: ', self.J_a.shape[0] * self.J_a.shape[1]) #65536
        print('Jacobian params for b: ', self.J_b.shape[0] * self.J_b.shape[1]) #35840
        print('RTRL_LoRA Jacobian total params: ', (self.J_a.shape[0] * self.J_a.shape[1] + self.J_b.shape[0] * self.J_b.shape[1])) #101376

        return jacobian_time
    
    @partial(jit, static_argnums=(0,))
    def gru(self, params, x, h):
        # Update gate
        z = sigmoid(np.dot(x, params['Wu']) + np.dot(h, params['Ru']))
        # Reset gate
        r = sigmoid(np.dot(x, params['Wr']) + np.dot(h, params['Rr']))
        # Candidate hidden state
        h_tilde = np.tanh(np.dot(x, params['Wh']) + np.dot(r * h, params['Rh']))

        # Final hidden state
        h = (1 - z) * h + z * h_tilde

        return h
    
    # Define the __call__ method
    def __call__(self, params, x, h):
        return self.gru(params, x, h)
    
    def forward_step(self, params, x, h, Jh_data_a, Jh_data_b):
        # Compute gradients with respect to 'a' and 'b' parameters in LoraWeight
        x = self.embed_input(x)

        (grad_h_params, grad_h_h), h = jacrev(self.lora_model, argnums=(0,2))(params, x, h)

        order = ['Wr', 'Wu', 'Wh', 'Rr', 'Ru', 'Rh']

        grad_h_flattened_params_a = []
        grad_h_flattened_params_b = []

        for key in order:
            grad_h_value = grad_h_params[key]

            # Extract and flatten 'a' and 'b' components
            grad_h_flattened_params_a.append(grad_h_value.a.reshape(self.hidden_size, -1))
            grad_h_flattened_params_b.append(grad_h_value.b.reshape(self.hidden_size, -1))

        # Concatenate all flattened arrays along the last axis (axis=1)
        grad_h_params_flat_a = jnp.concatenate(grad_h_flattened_params_a, axis=1)
        grad_h_params_flat_b = jnp.concatenate(grad_h_flattened_params_b, axis=1)

        #print("type of self.J_a: ", type(self.J_a))
        #print("Jh_data_a: ", Jh_data_a.shape)
        # Compute Jacobian products and update Jh, Jc
        h_Jh_a = np.dot(grad_h_h, self.J_a.toDense_rtrl(Jh_data_a))
        Jh_a = grad_h_params_flat_a + h_Jh_a 

        h_Jh_b = np.dot(grad_h_h, self.J_b.toDense_rtrl(Jh_data_b))
        Jh_b = grad_h_params_flat_b + h_Jh_b 

        return h, Jh_a, Jh_b

    @partial(jit, static_argnums=(0,))
    def calculate_loss(self, params, h, y):
        #output = self.activation(np.dot(self.V.toDense(params), h))
        #print(f"Shape of params: {params.shape}") #(32, 1)
        materialized_V = params.materialize()
        #print(f"Shape of materialized_V: {materialized_V.shape}") #(32, 1)
        #print(f"Shape of h: {h.shape}") #(32,)
        output = self.activation(np.dot(materialized_V.T, h)) #이렇게 해도 되는건가?
        #print(f"Shape of output: {output.shape}") #(1,)
        loss = self.lossFunction(output, y)
        return loss

    @partial(jit, static_argnums=(0,))
    def combineGradients_a(self, grad_h, grad_out_params_a, Jh_data_a):
        grad_rec_params = np.dot(grad_h, self.J_a.toDense_rtrl(Jh_data_a))
        grad_out_params_flat = grad_out_params_a.flatten()
        return np.concatenate((grad_rec_params, grad_out_params_flat))
    
    @partial(jit, static_argnums=(0,))
    def combineGradients_b(self, grad_h, grad_out_params_b, Jh_data_b):
        grad_rec_params = np.dot(grad_h, self.J_b.toDense_rtrl(Jh_data_b))
        grad_out_params_flat = grad_out_params_b.flatten()
        return np.concatenate((grad_rec_params, grad_out_params_flat))

    @partial(jit, static_argnums=(0,))
    def calculate_grads_step(self, params, x, y, h, Jh_data_a, Jh_data_b):

        filtered_params = {k: v for k, v in params.items() if k != 'V'}
        #materialized_params = self.pre_materialize_params(filtered_params)
        h, Jh_data_a, Jh_data_b = self.forward_step(filtered_params, x, h, Jh_data_a, Jh_data_b) #jh_data_a jh_data_b, jc_data_a

        #loss 계산에는 w+b*a, 기울기 업데이트 및 계산에서는 b랑 a만
        loss, (grad_out_params, grad_h) = value_and_grad(self.calculate_loss, argnums=(0,1))(params['V'], h, y) #params['V']? 여기도 inference 안에 들어가야 하나?

        #print("shape of grad_out_params: ", grad_out_params.shape) #(32, 1)
        #print("shape of grad_h: ", grad_h.shape)  #(32,)
        grad_out_params_a = grad_out_params.a
        grad_out_params_b = grad_out_params.b
        #print("shape of grad_out_a: ", grad_out_params_a.shape)
        #print("shape of grad_out_b: ", grad_out_params_b.shape)

        gradient_a = self.combineGradients_a(grad_h, grad_out_params_a, Jh_data_a)
        gradient_b = self.combineGradients_b(grad_h, grad_out_params_b, Jh_data_b)
            
        return loss, gradient_a, gradient_b, h, Jh_data_a, Jh_data_b

    batch_calculate_grads_step = vmap(calculate_grads_step, in_axes=(None, None, 0, 0, 0, 0, 0))

    def calculate_loss_val(self, params, x, y):

        output = self.forward(params, x)
        losses = vmap(self.lossFunction)(output, y)  # losses shape: (100,)
        mean_loss = jnp.mean(losses)
        return mean_loss

    @partial(jit, static_argnums=(0,))
    def forward_step_BPTT(self, params, x, t, h, o):
        filtered_params = {k: v for k, v in params.items() if k != 'V'}
        h = self.lora_model(filtered_params, x[t], h)

        V_param = params['V']
        materialized_V = V_param.materialize()
        #print(f"Shape of materialized_V: {materialized_V.shape}") #(32, 1)
        #print(f"Shape of h: {h.shape}") #(32,)
        output = self.activation(np.dot(materialized_V.T, h))

        o = o.at[t].set(output)
        return h, o

    def forward(self, params, x):

        x = self.embed_input(x)

        h = np.zeros(self.hidden_size)
        o = np.zeros((x.shape[0], self.output_size))
        #print("h: ", h.shape)
        #print("x: ", x.shape)
        #print("x[0]: ", x.shape[0])

        for t in range(x.shape[0]):
            h, o = self.forward_step_BPTT(params, x, t, h, o) #bptt는 self.forward_step_BPTT로 바꾸기, 리턴값에서 o만 씀
            
        return o

    def predict(self, x, y):
        #print("x_shape_validation: ", x.shape)  # Expected: (batch_size, seq_length)
        # Vectorize calculate_loss_BPTT over the batch dimension of x and y:
        val_loss_fn = vmap(self.calculate_loss_val, in_axes=(None, 0, 0))
        # This applies calculate_loss_BPTT for each sample in the batch.
        return val_loss_fn(self.lora_params, x, y)

    def evaluate(self, x_val, y_val):
        # Compute the loss for each sample in the batch:
        losses = self.predict(x_val, y_val)
        # Average over the batch to get a scalar validation loss:
        return np.mean(losses)
    
    def unflatten_params_from_order(self, flattened_params, param_order, lora_params, ab_key):
        """
        Converts a flattened array back into a dictionary of LoraWeight parameters.
        
        flattened_params: flattened numpy or jax array
        param_order: List of parameter names in order (e.g., ['Wi', 'Wo', 'Wf', 'Wz', 'Ri', 'Ro', 'Rf', 'Rz'])
        lora_params: original lora_params to infer the shape and structure
        ab_key: 'a' or 'b' to indicate which part of the LoraWeight to update
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
        Updates lora_params with the new values for 'a' and 'b' after optimization.
        
        lora_params: dict of LoraWeight objects
        new_flattened_a: flattened updated values for 'a'
        new_flattened_b: flattened updated values for 'b'
        param_order: List of parameter names in order (e.g., ['Wi', 'Wo', 'Wf', 'Wz', 'Ri', 'Ro', 'Rf', 'Rz'])
        """
        # Unflatten the flattened arrays
        new_a_params = self.unflatten_params_from_order(new_flattened_a, param_order, lora_params, 'a')
        new_b_params = self.unflatten_params_from_order(new_flattened_b, param_order, lora_params, 'b')

        #print("Type of new_a_params:", type(new_a_params))

        # Update lora_params with new a and b values
        for key in param_order:
            lora_params[key] = lora_params[key].update(a=new_a_params[key], b=new_b_params[key])

        return lora_params
    
    def update_fn(self, lora_params, x, y):
        # Initialize hidden and cell states, and Jacobians for the batch
        h = np.zeros((self.batch_size, self.hidden_size))

        Jh_data_a = np.zeros((self.batch_size, self.J_a.shape[0] * self.J_a.shape[1]))
        Jh_data_b = np.zeros((self.batch_size, self.J_b.shape[0] * self.J_b.shape[1]))

        losses = []

        for t in range(x.shape[1]):  # Iterate over time steps 시퀀스 길이
            # Use the batch_calculate_grads_step method from your LSTM class
            #print("type of params in update_fn: ", type(lora_params))
            loss, grads_a, grads_b, h, Jh_data_a, Jh_data_b = self.batch_calculate_grads_step(
                lora_params, x[:, t], y[:, t], h, Jh_data_a, Jh_data_b
            )

            losses.append(np.mean(loss, axis=0))

            #print("Shape of grads_a:", grads_a.shape) #(16, 2056) -> 32896
            #print("Shape of grads_b:", grads_b.shape) #(16, 1376) -> 22016 => 54912
            #print("Type of lora_params:", type(lora_params)) #dict

            # Update `a` parameters
            '''updates_a, self.opt_state_a = self.optimizer.update(np.sum(grads_a, axis=0), self.opt_state_a, params=lora_params)

            # Update `b` parameters
            updates_b, self.opt_state_b = self.optimizer.update(np.sum(grads_b, axis=0), self.opt_state_b, params=lora_params)

            # Apply updates separately to `a` and `b`
            lora_params.a = optax.apply_updates(lora_params, updates_a)
            lora_params.b = optax.apply_updates(lora_params, updates_b)'''

            self.opt_state_a = self.opt_update(0, np.sum(grads_a, axis=0), self.opt_state_a)
            self.opt_state_b = self.opt_update(0, np.sum(grads_b, axis=0), self.opt_state_b)

            #print("Type of self.opt_state_a:", type(self.opt_state_a)) 
            #print("Shape of self.opt_state_a:", self.opt_state_a.shape)

            new_flattened_a = self.get_params(self.opt_state_a)
            new_flattened_b = self.get_params(self.opt_state_b)

            #print("Type of new_flattened_a:", type(new_flattened_a)) #ArrayImpl
            #print("Shape of new_flattened_a:", new_flattened_a.shape) #(2056,)

            # Update lora_params with the new `a` and `b` values
            lora_params = self.update_lora_params(lora_params, new_flattened_a, new_flattened_b, self.param_order)

        return np.mean(np.array(losses)), lora_params

    def embed_input(self, x):
        """Converts token IDs to embedding vectors."""
        return self.embedding_matrix[x]
    
    def sample_batch(self, token_ids, batch_size, seq_length):
        total_tokens = len(token_ids)
        #print(type(token_ids))  # Should be a list or numpy array, not a dict
        #print("First 5 elements:", token_ids[:5])
        max_start = total_tokens - seq_length - 1  # Prevents index overflow
        
        key = jax.random.PRNGKey(0)  # Ensure reproducibility
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=max_start)  # Random start points
        
        batch = np.array([token_ids[idx: idx + seq_length + 1] for idx in indices], dtype=np.int32)  # Ensure integer type

        # Expected: (batch_size, seq_length+1)?????
        return jnp.array(batch)
    
    def run(self, epochs, data, validation_data):

        losses = []
        validation_losses = []
        epochs_list = []  # To store epoch numbers

        # Start timing
        start_time = time.time()
        

        for i, k in zip(range(epochs), random.split(self.key, epochs)):

            sampled_data = self.sample_batch(data, self.batch_size, self.seq_length)
            #print("after sampling x: ", sampled_data.shape)
            x, y = sampled_data[:, :-1], sampled_data[:, 1:]

            loss, self.lora_params = self.update_fn(self.lora_params, x, y)
            #total_loss = np.sum(loss)
            losses.append(loss)

            # 검증 데이터에 대한 손실을 계산합니다.
            validation_sampled_data = self.sample_batch(validation_data, self.batch_size, self.seq_length)
            x_val, y_val = validation_sampled_data[:, :-1], validation_sampled_data[:, 1:]
            val_loss = self.evaluate(x_val, y_val)  # 검증 데이터에 대한 손실 계산
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

        return losses, validation_losses, epochs_list, total_training_time