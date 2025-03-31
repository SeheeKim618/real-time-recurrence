from jax import random, vmap, jit
import jax.numpy as np
from jax.lax import log
from jax import linear_util as lu
from jax.tree_util import tree_map, tree_transpose, tree_structure
from jax._src.util import partial
from jax.api_util import argnums_partial
from jax._src.api import _unravel_array_into_pytree, _std_basis
from jax._src.api import _check_input_dtype_jacrev, _vjp, _check_output_dtype_jacrev
import numpy as old_np
from functools import lru_cache
import jax.numpy as jnp

"""
Calculate Sparse n-step pattern - Influence of parameters on hidden states in n timesteps.
"""
def calculateSnApPattern(snapLevel, weightRows, weightCols, recurrentRows, recurrentCols):

        @lru_cache(maxsize=None)
        def getInfluence(state):
            influence = np.where(recurrentRows == state) # where hidden state SnAP_rows[idx] influences other state
            influencedState = recurrentCols[influence] # next influenced state
            return influencedState

        SnAP_rows, SnAP_cols = [], []

        rows = np.concatenate((weightRows, recurrentRows))
        cols = np.concatenate((weightCols, recurrentCols))

        SnAP_rows.extend(cols[np.arange(len(rows))])
        SnAP_cols.extend(np.arange(len(rows)))

        if (snapLevel == 1):
            return SnAP_rows, SnAP_cols

        #reduce duplicates in recurrents
        coords = np.vstack((np.array(recurrentRows), np.array(recurrentCols)))
        [recurrentRows, recurrentCols] = old_np.unique(coords, axis=1)

        for s in range(1, snapLevel): #SnAP Level
            for idx in range(len(SnAP_rows)):
                '''if isinstance(SnAP_rows[idx], jnp.ndarray) and SnAP_rows[idx].ndim == 0:
                    # 스칼라 값인 경우, 스칼라를 튜플로 만들어 전달
                    influencedState = getInfluence((SnAP_rows[idx].item(),))
                else:
                    # 그렇지 않은 경우, 배열을 튜플로 만들어 전달
                    #influencedState = getInfluence(tuple(SnAP_rows[idx]))
                    influencedState = getInfluence((SnAP_rows[idx],))'''

                '''# getInfluence 함수 내에서 tuple 형태로 전달하기 전에 SnAP_rows[idx]의 타입을 체크합니다.
                if isinstance(SnAP_rows[idx], int):
                    influencedState = getInfluence((SnAP_rows[idx],))  # 정수를 단일 항목 튜플로 만듦
                else:
                    influencedState = getInfluence(tuple(SnAP_rows[idx]))  # 배열이나 리스트를 튜플로 변환'''
                #influencedState = getInfluence(SnAP_rows[idx])
                '''if SnAP_rows[idx].ndim == 0:
                    influencedState = getInfluence(SnAP_rows[idx].item())
                else:
                    influencedState = getInfluence(tuple(SnAP_rows[idx]))'''
                #influencedState = getInfluence((SnAP_rows[idx].item(),))
                if isinstance(SnAP_rows[idx], (np.ndarray, jnp.ndarray)) and SnAP_rows[idx].ndim == 0:
                    # SnAP_rows[idx]가 0차원 배열일 경우 .item()으로 스칼라 값을 추출
                    influencedState = getInfluence((SnAP_rows[idx].item(),))
                else:
                    #print("hereeeeeeeeee")
                    # SnAP_rows[idx]가 이미 정수 타입일 경우, 직접 튜플로 변환
                    influencedState = getInfluence((SnAP_rows[idx],))
                    
                SnAP_rows.extend(influencedState)
                SnAP_cols.extend(np.full((len(influencedState),), SnAP_cols[idx]))       

            coords = np.vstack((np.array(SnAP_rows), np.array(SnAP_cols)))
            [SnAP_rows, SnAP_cols] = old_np.unique(coords, axis=1)

            SnAP_rows = SnAP_rows.tolist()
            SnAP_cols = SnAP_cols.tolist()

        return np.array(SnAP_rows), np.array(SnAP_cols)

def jacrev(fun, argnums, holomorphic = False, allow_int = False):

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
    y, pullback = _vjp(f_partial, *dyn_args)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac = tree_map(partial(_unravel_array_into_pytree, y, 0, None), jac)
    jac = tree_transpose(tree_structure(example_args), tree_structure(y), jac)
    return jac, y

  return jacfun

class SparseMatrix:

    def __init__(self, key=random.PRNGKey(1), m=10, n=10, density=1, start=0):
        self.key = key
        self.density = density
        self.shape = (m, n) 
        self.start = start

    def jacobian(self, rows, cols, shape, start):
        self.rows = rows 
        self.cols = cols 
        self.shape = shape 
        self.start = start
        self.end = start + len(rows) 
        self.coords = (rows, cols)
        self.len = len(rows)
        self.density = self.len / (shape[0] * shape[1])
        
    def init(self):
        k1, k2 = random.split(self.key, 2)
        (m, n) = self.shape
        mn = m * n

        bound = np.sqrt(1/m)

        # Number of non zero values
        k = int(round(self.density * m * n))

        # flat index
        ind = random.choice(k1, mn, shape=(k,), replace=False).sort()

        row = np.floor(ind * 1. / n).astype(np.int16)
        col = (ind - row * n).astype(np.int16)
        #data = random.normal(self.key, (k,))
        data = random.uniform(self.key, (k,), minval=-bound, maxval=bound)

        self.rows = np.asarray(row) 
        self.cols = np.asarray(col)
        self.len = len(self.rows)
        self.end = self.start + self.len
        self.coords = (self.rows, self.cols)

        return np.asarray(data)

    @partial(jit, static_argnums=(0,))
    def toDense(self, data):

        #print(f"Shape of data before flatten: {data.shape}")
        #if data.size == np.prod(self.shape) and data.shape != self.shape:
        #data = data.flatten()

        #print(f"Shape of data after flatten: {data.shape}")
        #print(f"Expected shape: {self.shape}")
        #print(f"Coordinates: {self.coords}")
        return np.zeros(self.shape).at[tuple(self.coords)].add(data)
    
    @partial(jit, static_argnums=(0,))
    def toDense_rtrl(self, data):

        return data.reshape(self.shape)

@jit
def BinaryCrossEntropyLoss(y_hat, y):
    #print("y: ", y.shape)
    #print("y_hat", y_hat.shape)

    loss =  -(y * log(y_hat) + (1-y)* log(1-y_hat))
    return np.mean(loss)

@jit
def CrossEntropyLoss(y_hat, y):

    #print("y: ", y.shape) #-> (100,)
    #print("y_hat", y_hat.shape) #-> (100, 1013)

    epsilon = 1e-10  # Small constant to avoid log(0)
    
    # Select only the predicted probability for the correct class
    correct_probs = y_hat[np.arange(y.shape[0]), y]  # Shape: (batch_size,)

    # Compute cross-entropy loss
    loss = -np.mean(np.log(correct_probs + epsilon))  # the average loss over all 100 time steps.

    return loss

@jit
def CrossEntropyLoss_RTRL(y_hat, y):
    #print("This is CrossEntropyLoss_RTRL")
    #print("y: ", y.shape) #-> ()
    #print("y_hat", y_hat.shape) #-> (1013,)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # y is a scalar index, and y_hat is a vector of probabilities/logits.
    # Compute negative log likelihood for the correct class:
    loss = -jnp.log(y_hat[y] + epsilon)
    return loss


def one_hot_encoding(y, vocab_size):
    return jnp.eye(vocab_size)[y.squeeze()]

def compute_ema(data, alpha):
    """Compute the Exponential Moving Average (EMA) of a data series."""
    ema = [data[0]]  # Initialize the EMA with the first data point
    for t in range(1, len(data)):
        ema_t = alpha * data[t] + (1 - alpha) * ema[-1]
        ema.append(ema_t)
    return np.array(ema)

def load_model(filename="model.npz"):
    """ Load model parameters from a file """
    data = np.load(filename, allow_pickle=True)
    return {k: data[k] for k in data}

