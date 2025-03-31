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
#import lorax

from utils import BinaryCrossEntropyLoss, calculateSnApPattern, SparseMatrix, jacrev

import os
import matplotlib.pyplot as plt

#from lorax.transform import LoraWeight
from lorax2.transform2 import LoraWeight, lora
from lorax2.helpers2 import init_lora

"""
LSTM model with BPTT and RTRL training algorithm.
"""
class LSTM_LORA:

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
        self.batch_size= batch_size
        self.recurrent_density = recurrent_density
        self.in_out_density = in_out_density
        self.logEvery = logEvery
        self.activation = sigmoid
        # Define different rank constraints for each parameter
        self.rank_constraint_w = rank_constraint_w  # Example rank for input weights
        self.rank_constraint_r = rank_constraint_r  # Example rank for recurrent weights
        self.frozen_params = frozen_params

        self.jacobian_init_time = 0.0

        print('LSTM with '+ algo_type)
        print('Dense LSTM params: ', (4*hidden_size*(input_size+hidden_size) + hidden_size*output_size))
        #print('Sparse LSTM params: ', len(self.paramsData.flatten()))
        print('Density: ', recurrent_density)
        print('Rank_w: ', self.rank_constraint_w)
        print('Rank_r: ', self.rank_constraint_r)
        print('Shape of Frozen params: ', self.frozen_params.shape)

        #self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        #self.opt_init, self.opt_update, self.get_params = SophiaG(learning_rate)

        #self.opt_state = self.opt_init(self.paramsData)

        param_tree = self.reshape_params(self.frozen_params)

        lora_spec = {
            'Wi': self.rank_constraint_w,  # Rank for input weight 'Wi'
            'Wo': self.rank_constraint_w,  # Rank for input weight 'Wo'
            'Wf': self.rank_constraint_w,  # Rank for input weight 'Wf'
            'Wz': self.rank_constraint_w,  # Rank for input weight 'Wz'
            'Ri': self.rank_constraint_r,  # Rank for recurrent weight 'Ri'
            'Ro': self.rank_constraint_r,  # Rank for recurrent weight 'Ro'
            'Rf': self.rank_constraint_r,  # Rank for recurrent weight 'Rf'
            'Rz': self.rank_constraint_r,  # Rank for recurrent weight 'Rz'
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

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)

        # Define the order of the keys from Wi to V
        self.param_order = ['Wi', 'Wo', 'Wf', 'Wz', 'Ri', 'Ro', 'Rf', 'Rz', 'V']

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

        self.lossFunction = lossFunction
    
    def extract_params(self, params, key):
        return {k: getattr(v, key) for k, v in params.items() if isinstance(v, LoraWeight)}
    
    def flatten_params_in_order(self, params_dict, order):
        flattened_params = []
        for key in order:
            value = params_dict[key]
            flattened_params.append(value.flatten())  # Flatten each parameter
        return np.concatenate(flattened_params)

    def reshape_params(self, params):
        Wi_shape = (self.input_size, self.hidden_size)
        Wo_shape = (self.input_size, self.hidden_size)
        Wf_shape = (self.input_size, self.hidden_size)
        Wz_shape = (self.input_size, self.hidden_size)
        Ri_shape = (self.hidden_size, self.hidden_size)
        Ro_shape = (self.hidden_size, self.hidden_size)
        Rf_shape = (self.hidden_size, self.hidden_size)
        Rz_shape = (self.hidden_size, self.hidden_size)
        V_shape = (self.hidden_size, self.output_size)

        # The sizes of each parameter
        sizes = [np.prod(np.array(Wi_shape)), np.prod(np.array(Wo_shape)), np.prod(np.array(Wf_shape)), np.prod(np.array(Wz_shape)),
                 np.prod(np.array(Ri_shape)), np.prod(np.array(Ro_shape)), np.prod(np.array(Rf_shape)), np.prod(np.array(Rz_shape)),
                 np.prod(np.array(V_shape))]

        # Start slicing from the flattened array
        offset = 0
        Wi_data = params[offset:offset + sizes[0]].reshape(Wi_shape)
        offset += sizes[0]
        Wo_data = params[offset:offset + sizes[1]].reshape(Wo_shape)
        offset += sizes[1]
        Wf_data = params[offset:offset + sizes[2]].reshape(Wf_shape)
        offset += sizes[2]
        Wz_data = params[offset:offset + sizes[3]].reshape(Wz_shape)
        offset += sizes[3]
        Ri_data = params[offset:offset + sizes[4]].reshape(Ri_shape)
        offset += sizes[4]
        Ro_data = params[offset:offset + sizes[5]].reshape(Ro_shape)
        offset += sizes[5]
        Rf_data = params[offset:offset + sizes[6]].reshape(Rf_shape)
        offset += sizes[6]
        Rz_data = params[offset:offset + sizes[7]].reshape(Rz_shape)
        offset += sizes[7]
        V_data = params[offset:offset + sizes[8]].reshape(V_shape)

        param_tree = {
            'Wi': Wi_data,
            'Wo': Wo_data,
            'Wf': Wf_data,
            'Wz': Wz_data,
            'Ri': Ri_data,
            'Ro': Ro_data,
            'Rf': Rf_data,
            'Rz': Rz_data,
            'V': V_data,
        }

        return param_tree

    def initialize_jacob(self):
        start_time = time.time()

        # Calculate the total number of parameters for 'a' and 'b' separately from Wi to Rz
        total_a_params = 0
        total_b_params = 0
        for key in ['Wi', 'Wo', 'Wf', 'Wz', 'Ri', 'Ro', 'Rf', 'Rz']:
            lora_weight = self.lora_params[key]
            a_shape = lora_weight.a.shape  # Get the shape of the 'a' matrix
            b_shape = lora_weight.b.shape  # Get the shape of the 'b' matrix
            total_a_params += a_shape[0] * a_shape[1]  # Add the number of elements in the 'a' matrix
            total_b_params += b_shape[0] * b_shape[1]  # Add the number of elements in the 'b' matrix

        # Initialize the Jacobian matrices for 'a' and 'b' as SparseMatrix
        self.J_a = SparseMatrix(m=self.hidden_size, n=total_a_params)
        self.J_b = SparseMatrix(m=self.hidden_size, n=total_b_params)

        # Initialize the sparse matrices with random or specific data if necessary
        self.J_a_data = self.J_a.init()
        self.J_b_data = self.J_b.init()

        # End timing
        end_time = time.time()
        
        # Calculate elapsed time
        jacobian_time = end_time - start_time

        print('Jacobian Shape for a: ', self.J_a.shape) #(32, 2048)
        print('Jacobian Shape for b: ', self.J_b.shape) #(32, 1120)
        print('Jacobian params for a: ', self.J_a.len) #65536
        print('Jacobian params for b: ', self.J_b.len) #35840
        print('Jacobian total params: ', (self.J_a.len + self.J_b.len)) #101376

        return jacobian_time
    
    @partial(jit, static_argnums=(0,))
    def lstm(self, params, x, h, c):

        # Debugging shapes before toDense
        #print("Shape of wi_params before toDense:", params['Wi'].shape) #(3,32)
        #print("Shape of ri_params before toDense:", params['Ri'].shape) #(32,32)

        inputGate = sigmoid(np.dot(x, params['Wi']) + np.dot(h, params['Ri']))
        outputGate = sigmoid(np.dot(x, params['Wo']) + np.dot(h, params['Ro']))
        forgetGate = sigmoid(np.dot(x, params['Wf']) + np.dot(h, params['Rf']))

        # Cell Input
        z = np.tanh(np.dot(x, params['Wz']) + np.dot(h, params['Rz']))

        # Cell State
        c = forgetGate * c + inputGate * z

        # Cell Output
        h = outputGate * np.tanh(c)

        return h, c
    
    # Define the __call__ method
    def __call__(self, params, x, h, c):
        return self.lstm(params, x, h, c)
    
    '''def pre_materialize_params(self, params):
        materialized_param_tree = {
            'Wi': params['Wi'].materialize(),
            'Wo': params['Wo'].materialize(),
            'Wf': params['Wf'].materialize(),
            'Wz': params['Wz'].materialize(),
            'Ri': params['Ri'].materialize(),
            'Ro': params['Ro'].materialize(),
            'Rf': params['Rf'].materialize(),
            'Rz': params['Rz'].materialize(),
            'V': params['V'].materialize(),
        }
        
        return materialized_param_tree
    
    def materialize_lora_params(lora_params):
        materialized_params = {}
        for key, param_dict in lora_params.items():
            w = param_dict['w']
            a = param_dict['a']
            b = param_dict['b']
            alpha = param_dict['alpha']
            materialized_params[key] = w + (alpha / a.shape[0]) * jnp.dot(b, a)
        return materialized_params'''
    
    def forward_step(self, params, x, h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b):
        # Compute gradients with respect to 'a' and 'b' parameters in LoraWeight
        ((grad_h_params, grad_h_h, grad_h_c), (grad_c_params, grad_c_h, grad_c_c)), (h, c) = jacrev(self.lora_model, argnums=(0,2,3))(params, x, h, c)

        order = ['Wi', 'Wo', 'Wf', 'Wz', 'Ri', 'Ro', 'Rf', 'Rz']

        grad_h_flattened_params_a = []
        grad_h_flattened_params_b = []
        grad_c_flattened_params_a = []
        grad_c_flattened_params_b = []

        for key in order:
            grad_h_value = grad_h_params[key]
            grad_c_value = grad_c_params[key]

            # Extract and flatten 'a' and 'b' components
            grad_h_flattened_params_a.append(grad_h_value.a.reshape(32, -1))
            grad_h_flattened_params_b.append(grad_h_value.b.reshape(32, -1))
            
            grad_c_flattened_params_a.append(grad_c_value.a.reshape(32, -1))
            grad_c_flattened_params_b.append(grad_c_value.b.reshape(32, -1))

        # Concatenate all flattened arrays along the last axis (axis=1)
        grad_h_params_flat_a = jnp.concatenate(grad_h_flattened_params_a, axis=1)
        grad_h_params_flat_b = jnp.concatenate(grad_h_flattened_params_b, axis=1)
        grad_c_params_flat_a = jnp.concatenate(grad_c_flattened_params_a, axis=1)
        grad_c_params_flat_b = jnp.concatenate(grad_c_flattened_params_b, axis=1)

        #print("type of self.J_a: ", type(self.J_a))
        #print("Jh_data_a: ", Jh_data_a.shape)
        # Compute Jacobian products and update Jh, Jc
        h_Jh_a = np.dot(grad_h_h, self.J_a.toDense(Jh_data_a))[tuple(self.J_a.coords)]
        h_Jc_a = np.dot(grad_h_c, self.J_a.toDense(Jc_data_a))[tuple(self.J_a.coords)]
        Jh_a = grad_h_params_flat_a[tuple(self.J_a.coords)] + h_Jh_a + h_Jc_a

        h_Jh_b = np.dot(grad_h_h, self.J_b.toDense(Jh_data_b))[tuple(self.J_b.coords)]
        h_Jc_b = np.dot(grad_h_c, self.J_b.toDense(Jc_data_b))[tuple(self.J_b.coords)]
        Jh_b = grad_h_params_flat_b[tuple(self.J_b.coords)] + h_Jh_b + h_Jc_b

        c_Jh_a = np.dot(grad_c_h, self.J_a.toDense(Jh_data_a))[tuple(self.J_a.coords)]
        c_Jc_a = np.dot(grad_c_c, self.J_a.toDense(Jc_data_a))[tuple(self.J_a.coords)]
        Jc_a = grad_c_params_flat_a[tuple(self.J_a.coords)] + c_Jh_a + c_Jc_a

        c_Jh_b = np.dot(grad_c_h, self.J_b.toDense(Jh_data_b))[tuple(self.J_b.coords)]
        c_Jc_b = np.dot(grad_c_c, self.J_b.toDense(Jc_data_b))[tuple(self.J_b.coords)]
        Jc_b = grad_c_params_flat_b[tuple(self.J_b.coords)] + c_Jh_b + c_Jc_b

        return h, c, Jh_a, Jh_b, Jc_a, Jc_b

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
        grad_rec_params = np.dot(grad_h, self.J_a.toDense(Jh_data_a))
        grad_out_params_flat = grad_out_params_a.flatten()
        return np.concatenate((grad_rec_params, grad_out_params_flat))
    
    @partial(jit, static_argnums=(0,))
    def combineGradients_b(self, grad_h, grad_out_params_b, Jh_data_b):
        grad_rec_params = np.dot(grad_h, self.J_b.toDense(Jh_data_b))
        grad_out_params_flat = grad_out_params_b.flatten()
        return np.concatenate((grad_rec_params, grad_out_params_flat))

    @partial(jit, static_argnums=(0,))
    def calculate_grads_step(self, params, x, y, h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b):

        filtered_params = {k: v for k, v in params.items() if k != 'V'}
        #materialized_params = self.pre_materialize_params(filtered_params)
        h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b = self.forward_step(filtered_params, x, h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b) #jh_data_a jh_data_b, jc_data_a

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
            
        return loss, gradient_a, gradient_b, h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b

    batch_calculate_grads_step = vmap(calculate_grads_step, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0))
    
    '''def forward_evel(self, params, x):

        h = np.zeros((self.batch_size, self.hidden_size))
        c = np.zeros((self.batch_size, self.hidden_size))

        h, c = self.lstm(params[:self.Rz.end,], x, h, c)
        output = self.activation(np.dot(params[self.Rz.end:,], h))
            
        return output'''

    @partial(jit, static_argnums=(0,))
    def forward_step_BPTT(self, params, x, t, h, c, o):
        filtered_params = {k: v for k, v in params.items() if k != 'V'}
        h, c = self.lora_model(filtered_params, x[t], h, c)

        V_param = params['V']
        materialized_V = V_param.materialize()
        #print(f"Shape of materialized_V: {materialized_V.shape}") #(32, 1)
        #print(f"Shape of h: {h.shape}") #(32,)
        output = self.activation(np.dot(materialized_V.T, h))

        o = o.at[t].set(output)
        return h, c, o

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
        return batch_forward(self.lora_params, x)
    
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
        c = np.zeros((self.batch_size, self.hidden_size))

        Jh_data_a = np.zeros((self.batch_size, self.J_a.len))
        Jh_data_b = np.zeros((self.batch_size, self.J_b.len))
        Jc_data_a = np.zeros((self.batch_size, self.J_a.len))
        Jc_data_b = np.zeros((self.batch_size, self.J_b.len))

        losses = []

        for t in range(x.shape[1]):  # Iterate over time steps
            # Use the batch_calculate_grads_step method from your LSTM class
            #print("type of params in update_fn: ", type(lora_params))
            loss, grads_a, grads_b, h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b = self.batch_calculate_grads_step(
                lora_params, x[:, t, :], y[:, t, :], h, c, Jh_data_a, Jh_data_b, Jc_data_a, Jc_data_b
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
    
    def run(self, epochs, data, validation_data):

        losses = []
        validation_losses = []
        achieved_sequence_lengths = []
        epochs_list = []  # To store epoch numbers

        # Start timing
        start_time = time.time()
        
        for i, k in zip(range(epochs), random.split(self.key, epochs)):

            x, y = data.getSample(k)

            #print("type of params in cal grads: ", type(self.lora_params))
            loss, self.lora_params = self.update_fn(self.lora_params, x, y)
            #total_loss = np.sum(loss)
            losses.append(loss)

            #이때 파라미터 (4480,)으로 바꿔야하나

            # Check if the condition to increase L is met
            '''if loss < 0.15:
                data.maxSeqLength += 1
                validation_data.maxSeqLength += 1
                # Make sure to update the data generation to reflect new maxSeqLength if necessary'''

            if i % 5 == 0:
                print('Epoch', "{:04d}".format(i))
                print('Train Loss ', loss)

                # 검증 데이터에 대한 손실을 계산합니다.
                x_val, y_val = validation_data.getSample(k)
                val_loss = self.evaluate(x_val, y_val)  # 검증 데이터에 대한 손실 계산
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

        return self.lora_params, losses, validation_losses, epochs_list, achieved_sequence_lengths, total_training_time
