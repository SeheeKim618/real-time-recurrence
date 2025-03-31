import os
import sys
import matplotlib.pyplot as plt
import argparse

from jax import random
import jax.numpy as np
#import numpy as np

from CopyTaskData import CopyTaskData
from lstm import LSTM
from gru_wiki import GRU
from lstm_lora import LSTM_LORA
from gru_lora_wiki import GRU_LORA
from gru_rtrl_wiki import GRU_RTRL
from gru_tbptt_wiki import GRU_TBPTT
from utils import CrossEntropyLoss, CrossEntropyLoss_RTRL, compute_ema, load_model
from datasets import load_dataset

def tokenize_function(dataset):
    """Convert dataset text into a flat array of character-level tokens."""
    all_text = "".join(sample['text'] for sample in dataset)  # Extract text & concatenate
    return np.array([ord(char) for char in all_text], dtype=np.int32)  # Convert to ASCII

def get_vocab_size(dataset):
    """Calculate the number of unique characters in the dataset."""
    all_text = "".join(sample['text'] for sample in dataset)  # Extract and merge text
    unique_chars = sorted(set(all_text))  # Find unique characters
    return len(unique_chars)

def main(args):
    print(f"Running with settings: Model={args.model}, Type={args.type}, Level={args.level}, Online={args.online}, Recurrent Density={args.recurrent_density}, Input and Output Density={args.inout_density}")

    key = random.PRNGKey(1)
    np.set_printoptions(formatter={'float_kind':"{:.5f}".format})

    epochs = 150

    epochs_pretrain = 150
    epochs_finetune = 200

    if args.type == "lora_rtrl":
        epochs = epochs_finetune
    else:
        epochs = epochs
    
    if args.type == "bptt":
        learning_rate = 1e-2
    else:
        learning_rate = 5e-3

    logEvery = 1
    batch_size = 32

    bits = 1
    padding = 0
    lowValue = 0
    highValue = 1

    hidden_size = 32

    seq_length = 100
    trunc_length = 10

    embedding_dim = 32
    
    if args.type in ['bptt', 'tbptt']:
        lossFunction = CrossEntropyLoss
    elif args.type == "lora_rtrl":
        lossFunction_pretrain = CrossEntropyLoss
        lossFunction_finetune = CrossEntropyLoss_RTRL
    else:
        lossFunction = CrossEntropyLoss_RTRL

    # Define different rank constraints for each parameter
    rank_constraint_w = 8  
    rank_constraint_r = 8 

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    # Check the dataset structure
    print(dataset)

    # Extract the train, validation, and test datasets
    train_data = dataset['train']
    validation_data = dataset['validation']
    #test_dataset = dataset['test']

    # Get vocabulary size
    vocab_size = get_vocab_size(train_data)
    print(f"VOCAB_SIZE: {vocab_size}")
    output_size = vocab_size

    if args.type == "lora_rtrl":
        # Shuffle the training data
        train_data_shuffled = train_data.shuffle(seed=42)

        # Split the training data into pre-training (70%) and fine-tuning (30%) subsets
        split_data = train_data_shuffled.train_test_split(test_size=0.3, seed=42)
        pretrain_data = split_data['train']
        finetune_data = split_data['test']

        print("Pre-training data: ", pretrain_data.shape[0])
        print("Fine-tuning data: ", finetune_data.shape[0])

        # Tokenize the datasets
        pretrain_id = tokenize_function(pretrain_data)
        finetune_id = tokenize_function(finetune_data)
        validation_id = tokenize_function(validation_data)

    else:
        train_id = tokenize_function(train_data)
        validation_id = tokenize_function(validation_data)
    
    validation_key = random.PRNGKey(2)
    validation_data.key = validation_key

    if args.model == 'lstm':
        if args.type in ['rtrl', 'snap', 'bptt']:
            model = LSTM(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction, 
                    args.type,
                    seq_length,
                    vocab_size, 
                    embedding_dim, 
                    logEvery, 
                    learning_rate, 
                    args.online)
            params, losses, validation_losses, epochs_list, total_training_time = model.run(epochs, train_id, validation_id)
            
        elif args.type == 'lora_rtrl':
            model = LSTM_LORA(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction, 
                    args.type,
                    seq_length,
                    vocab_size, 
                    embedding_dim,
                    rank_constraint_w,
                    rank_constraint_r,
                    frozen_params, 
                    logEvery, 
                    learning_rate) 
            params, losses, validation_losses, epochs_list, total_training_time = model.run(epochs, train_id, validation_id)               
        else:
            print("Error: Training algorithm Type is not provided.")

    elif args.model == 'gru':
        if args.type in ['snap', 'bptt']:
            model = GRU(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction, 
                    args.type,
                    seq_length,
                    vocab_size, 
                    embedding_dim, 
                    logEvery, 
                    learning_rate, 
                    args.online)

            params, losses, validation_losses, epochs_list, total_training_time = model.run(epochs, train_id, validation_id)

        elif args.type == 'tbptt':
            model = GRU_TBPTT(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction, 
                    args.type,
                    seq_length,
                    vocab_size,
                    trunc_length, 
                    embedding_dim, 
                    logEvery, 
                    learning_rate, 
                    args.online)
            params, losses, validation_losses, epochs_list, total_training_time = model.run(epochs, train_id, validation_id)

        elif args.type == 'rtrl':
            model = GRU_RTRL(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction, 
                    args.type,
                    seq_length,
                    vocab_size, 
                    embedding_dim, 
                    logEvery, 
                    learning_rate, 
                    args.online)

            params, losses, validation_losses, epochs_list, total_training_time = model.run(epochs, train_id, validation_id)

        elif args.type == 'lora_rtrl':
            '''print("Pre-training starts!")
            model = GRU(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction_pretrain, 
                    args.type,
                    seq_length,
                    vocab_size, 
                    embedding_dim, 
                    logEvery, 
                    learning_rate, 
                    args.online)
            params, losses_pretrain, validation_losses_pretrain, epochs_list, total_training_time = model.run(epochs_pretrain, pretrain_id, validation_id)
            
            print("Final 10 values in validation_loss array:", validation_losses_pretrain[-10:])
            # Smoothing Loss
            alpha = 0.5  # Smoothing factor, can be adjusted between 0 (more smooth) and 1 (less smooth)
            training_loss_ema_pretrain = compute_ema(losses_pretrain, alpha)
            validatioin_loss_ema_pretrain = compute_ema(validation_losses_pretrain, alpha)

            # Loss Plot
            epochs_range_pretrain = range(epochs_pretrain)
            validation_epochs_range_pretrain = range(0, epochs_pretrain, 5)

            plt.figure(figsize=(12, 6))
            plt.plot(epochs_range_pretrain, training_loss_ema_pretrain, label=f"Training Loss {args.type}_{args.recurrent_density}")
            plt.plot(validation_epochs_range_pretrain, validatioin_loss_ema_pretrain, label='Validation Loss', linestyle='--')
            plt.title(f'{args.model}')
            plt.xlabel('Epochs')
            plt.ylabel('Losses')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'wiki_pretrain_loss_{args.model}_{args.type}_{args.recurrent_density}.png')   

            # Save the trained model parameters
            np.savez(f'trained_model_wiki_{args.model}_{args.type}.npz', params=np.array(params))
            print(f'trained_model_wiki_{args.model}_{args.type} saved successfully.')

            # Load the pre-trained model parameters
            frozen_params = params'''

            trained_model = np.load('trained_model_wiki_gru_lora_rtrl.npz')
            frozen_params = trained_model['params']

            print("-----------------------")
            print("Fine-tuning starts!")
            model = GRU_LORA(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    args.recurrent_density, 
                    args.inout_density, 
                    args.level, 
                    lossFunction_finetune, 
                    args.type,
                    seq_length,
                    vocab_size, 
                    embedding_dim,
                    rank_constraint_w,
                    rank_constraint_r,
                    frozen_params, 
                    logEvery, 
                    learning_rate)

            losses, validation_losses, epochs_list, total_training_time = model.run(epochs_finetune, finetune_id, validation_id)
                       
        else:
            print("Error: Training algorithm Type is not provided.")
    else:
        print("Error: Model Type is not provided.")
    
    # Smoothing Loss
    alpha = 0.5  # Smoothing factor, can be adjusted between 0 and 1
    training_loss_ema = compute_ema(losses, alpha)
    validatioin_loss_ema = compute_ema(validation_losses, alpha)

    # Loss Plot
    epochs_range = range(epochs)
    validation_epochs_range = range(0, epochs, 5)

    if args.type == 'snap':
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, training_loss_ema, label=f'Training Loss {args.type}_{args.level}_{args.recurrent_density}')
        plt.plot(validation_epochs_range, validatioin_loss_ema, label='Validation Loss', linestyle='--')
        plt.title(f'{args.model}')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'wiki_loss_{args.model}_{args.type}_{args.level}_{args.recurrent_density}.png')
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, training_loss_ema, label=f"Training Loss {args.type}_{args.recurrent_density}")
        plt.plot(validation_epochs_range, validatioin_loss_ema, label='Validation Loss', linestyle='--')
        plt.title(f'{args.model}')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'wiki_loss_{args.model}_{args.type}_{args.recurrent_density}.png')
    

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