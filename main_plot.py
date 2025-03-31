import os
import sys
import matplotlib.pyplot as plt
import argparse

from jax import random
import jax.numpy as np
#import numpy as np

from CopyTaskData import CopyTaskData
from lstm import LSTM
from gru import GRU
from lstm_lora import LSTM_LORA
from gru_lora import GRU_LORA

from utils import BinaryCrossEntropyLoss, compute_ema, load_model

def main(args):

    key = random.PRNGKey(1)
    np.set_printoptions(formatter={'float_kind':"{:.5f}".format})

    epochs = 50
    logEvery = 1
    learning_rate = 1e-3
    batch_size = 16

    maxSeqLength = 10
    minSeqLength = np.maximum(1, (maxSeqLength-5))
    bits = 1
    padding = 0
    lowValue = 0
    highValue = 1

    input_size = bits+2
    output_size = bits
    hidden_size = 32

    seq_length = 2*maxSeqLength + 2
    
    lossFunction = BinaryCrossEntropyLoss

    # Define different rank constraints for each parameter
    #rank_constraint_w = 2  
    #rank_constraint_r = 8

    # Load the pre-trained model parameters
    trained_model = np.load('./saved_model/trained_model_gru_32_bptt_7.npz')
    frozen_params = trained_model['params']

    data = CopyTaskData(batch_size, maxSeqLength, minSeqLength, bits, padding, lowValue, highValue)
    validation_data = CopyTaskData(batch_size, maxSeqLength, minSeqLength, bits, padding, lowValue, highValue)
    
    validation_key = random.PRNGKey(2)
    validation_data.key = validation_key

    model_configs = {
        #'BPTT': {'type': 'bptt', 'level': 2, 'recurrent_density': 1, 'inout_density': 1},
        #'RTRL': {'type': 'rtrl', 'level': 2, 'recurrent_density': 1, 'inout_density': 1},  # No specific args for rtrl
        #'SnAp-1': {'type': 'snap', 'level': 1, 'recurrent_density': 1, 'inout_density': 1},
        #'SnAp-2 (d=0.6)': {'type': 'snap', 'level': 2, 'recurrent_density': 0.6, 'inout_density': 1},
        #'LoRA-RTRL1': {'type': 'lora_rtrl', 'level': 2, 'recurrent_density': 1, 'inout_density': 1, 'rank_constraint_w': 2, 'rank_constraint_r': 2},
        'LoRA-RTRL2': {'type': 'lora_rtrl', 'level': 2, 'recurrent_density': 1, 'inout_density': 1, 'rank_constraint_w': 2, 'rank_constraint_r': 2},
        'LoRA-RTRL3': {'type': 'lora_rtrl', 'level': 2, 'recurrent_density': 1, 'inout_density': 1, 'rank_constraint_w': 3, 'rank_constraint_r': 30}
    }

    #'LoRA-RTRL': {'type': 'lora_rtrl', 'level': 2, 'recurrent_density': 1, 'inout_density': 1}
    #'SnAp-2 (d=0.25)': {'type': 'snap', 'level': 2, 'recurrent_density': 0.25, 'inout_density': 1}

    validation_losses_dict = {
        #'BPTT': [],
        #'RTRL': [],
        #'SnAp-1': [],
        #'SnAp-2 (d=0.6)': [],
        #'LoRA-RTRL1': [],
        'LoRA-RTRL2': [],
        'LoRA-RTRL3': []
    }

    total_training_times = {
        #'BPTT': [],
        #'RTRL': [],
        #'SnAp-1': [],
        #'SnAp-2 (d=0.6)': [],
        #'LoRA-RTRL1': [],
        'LoRA-RTRL2': [],
        'LoRA-RTRL3': []
    }

    jacobian_init_times = {}

    # Train different models
    for model_name, config in model_configs.items():
        # Set args according to the current model configuration
        algo_type = config['type']
        level = config['level']
        recurrent_density = config['recurrent_density']
        inout_density = config['inout_density']
        rank_constraint_w = config['rank_constraint_w']
        rank_constraint_r = config['rank_constraint_r']
       
        print(f"Running: Model=GRU, Type={algo_type}, Level={level}, Recurrent Density={recurrent_density}, Input and Output Density={inout_density}")

        if algo_type in ['rtrl', 'snap', 'bptt']:
            model = GRU(key, input_size, output_size, hidden_size, batch_size, 
                        recurrent_density, inout_density, level, 
                        lossFunction, algo_type, logEvery, learning_rate, args.online)
        elif algo_type == 'lora_rtrl':
            model = GRU_LORA(key, input_size, output_size, hidden_size, batch_size, 
                            recurrent_density,inout_density, level, 
                            lossFunction, algo_type, rank_constraint_w, rank_constraint_r, 
                            frozen_params, logEvery, learning_rate)
        else:
            print("Error: Training algorithm Type is not provided.")

        # Run the model
        params, losses, validation_losses, epochs_list, achieved_sequence_lengths, total_training_time = model.run(epochs, data, validation_data)

        # Compute EMA for validation losses
        alpha = 0.1  # Smoothing factor, can be adjusted between 0 and 1
        validation_loss_ema = compute_ema(validation_losses, alpha)

        # Store the collected validation losses
        validation_losses_dict[model_name] = (epochs_list, validation_loss_ema)

        total_training_times[model_name].append(total_training_time)

        jacobian_init_times[model_name] = model.jacobian_init_time

    '''# To demonstrate loading:
    loaded_params = load_model("trained_model.npz")
    print("Loaded model parameters successfully.")'''

    '''processed_sequence_lengths = []
    for i, item in enumerate(achieved_sequence_lengths):
        epoch_number = i * 5
        if isinstance(item, int):
            processed_sequence_lengths.append((epoch_number, item))
        else: 
            value = item.item() 
            processed_sequence_lengths.append((epoch_number, value))

    epochs_plot, sequence_lengths = zip(*processed_sequence_lengths)

    # Achieved Sequence Lengths Plot
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_plot, sequence_lengths, label=args.type)
    plt.xlabel('Epochs')
    plt.ylabel('Achieved Sequence Length')
    plt.title('Achieved Sequence Length over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('copy_task_result.png')'''

    print("----------train is done.-----------")
    print("total training time: ", total_training_times)
    print("Jacobian initialization times:")
    for model_name, init_time in jacobian_init_times.items():
        print(f"{model_name}: {init_time:.4f} seconds")

    # Plotting validation losses for each model type
    plt.figure()

    for model_name, (epoch_nums, val_losses) in validation_losses_dict.items():
        plt.plot(epoch_nums, val_losses, label=model_name)

    plt.xlabel('Validation epochs', fontsize=14)
    plt.ylabel('Cross Entropy Loss', fontsize=14)
    plt.title('GRU 32', fontsize=22)
    plt.legend(fontsize=12)  # Increase legend font size
    plt.xticks(fontsize=12)  # Increase x-axis tick font size
    plt.yticks(fontsize=12)  # Increase y-axis tick font size
    plt.grid(True)
    plt.savefig(f'copy_task_loss_gru_32_integrated_lora_rank.png', bbox_inches='tight')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, required=True, help='Type of the model')
    parser.add_argument('--type', type=str, required=True, help='Type of the training algorithm type')
    parser.add_argument('--level', type=int, required=True, help='Level of the algorithm')
    parser.add_argument('--online', type=lambda x: (str(x).lower() == 'true'), help='Run the algorithm online or not')
    parser.add_argument('--recurrent-density', type=float, required=True, help='Density of recurrent connections')
    parser.add_argument('--inout-density', type=float, required=True, help='Density of input/output connections')

    args = parser.parse_args()
    main(args)