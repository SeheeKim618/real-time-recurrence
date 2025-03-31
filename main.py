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
    print(f"Running with settings: Model={args.model}, Type={args.type}, Level={args.level}, Online={args.online}, Recurrent Density={args.recurrent_density}, Input and Output Density={args.inout_density}")

    key = random.PRNGKey(1)
    np.set_printoptions(formatter={'float_kind':"{:.5f}".format})

    epochs = 500
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
    rank_constraint_w = 2  
    rank_constraint_r = 8 

    # Load the pre-trained model parameters
    trained_model = np.load('./saved_model_copytask/trained_model_gru_32_bptt_7.npz')
    frozen_params = trained_model['params']

    data = CopyTaskData(batch_size, maxSeqLength, minSeqLength, bits, padding, lowValue, highValue)
    validation_data = CopyTaskData(batch_size, maxSeqLength, minSeqLength, bits, padding, lowValue, highValue)
    
    validation_key = random.PRNGKey(2)
    validation_data.key = validation_key

    if args.model == 'lstm':
        if args.type in ['rtrl', 'snap', 'bptt']:
            model = LSTM(key,
                    input_size, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction, 
                    args.type, 
                    logEvery, 
                    learning_rate, 
                    args.online)
            
        elif args.type == 'lora_rtrl':
            model = LSTM_LORA(key,
                    input_size, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction, 
                    args.type,
                    rank_constraint_w,
                    rank_constraint_r,
                    frozen_params, 
                    logEvery, 
                    learning_rate)                
        else:
            print("Error: Training algorithm Type is not provided.")

    elif args.model == 'gru':
        if args.type in ['rtrl', 'snap', 'bptt']:
            model = GRU(key,
                    input_size, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction, 
                    args.type, 
                    logEvery, 
                    learning_rate, 
                    args.online)
            
        elif args.type == 'lora_rtrl':
            model = GRU_LORA(key,
                    input_size, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction, 
                    args.type,
                    rank_constraint_w,
                    rank_constraint_r,
                    frozen_params, 
                    logEvery, 
                    learning_rate)                
        else:
            print("Error: Training algorithm Type is not provided.")
    else:
        print("Error: Model Type is not provided.")
    
    params, losses, validation_losses, epochs_list, achieved_sequence_lengths, total_training_time = model.run(epochs, data, validation_data)

    # Save the trained model parameters
    np.savez(f'trained_model_{args.model}_{args.type}_{maxSeqLength}.npz', params=np.array(params))
    print(f'trained_model_{args.model}_{args.type}_{maxSeqLength} saved successfully.')

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
    
    # Smoothing Loss
    alpha = 0.05  # Smoothing factor, can be adjusted between 0 and 1
    training_loss_ema = compute_ema(losses, alpha)
    validatioin_loss_ema = compute_ema(validation_losses, alpha)

    # Loss Plot
    epochs_range = range(epochs)
    validation_epochs_range = range(0, epochs, 5)

    if args.type == 'snap':
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, training_loss_ema, label=f"{args.type}_{args.level}_{args.recurrent_density}")
        plt.plot(validation_epochs_range, validatioin_loss_ema, label='Validation Loss', linestyle='--')
        plt.title(f'{args.model}')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'copy_task_loss_{args.model}_{args.type}_{args.level}_{args.recurrent_density}.png')
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, training_loss_ema, label=f"{args.type}_{args.recurrent_density}")
        plt.plot(validation_epochs_range, validatioin_loss_ema, label='Validation Loss', linestyle='--')
        plt.title(f'{args.model}')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'copy_task_loss_{args.model}_{args.type}_{args.recurrent_density}.png')
    

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
