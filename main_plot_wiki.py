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

    key = random.PRNGKey(1)
    np.set_printoptions(formatter={'float_kind':"{:.5f}".format})

    epochs = 200 #128

    epochs_pretrain = 20 #70 
    epochs_finetune = 180 #58

    logEvery = 1
    batch_size = 32
    learning_rate = 1e-2 #1e-3 /4

    bits = 1
    padding = 0
    lowValue = 0
    highValue = 1

    hidden_size = 32

    seq_length = 100
    trunc_length = 10

    embedding_dim = 32

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

    train_id = tokenize_function(train_data)
    validation_id = tokenize_function(validation_data)

    validation_key = random.PRNGKey(2)
    validation_data.key = validation_key

    model_configs = {
        'BPTT': {'type': 'bptt', 'level': 2, 'online': False, 'recurrent_density': 1, 'inout_density': 1},
        'TBPTT': {'type': 'tbptt', 'level': 2, 'online': False, 'recurrent_density': 1, 'inout_density': 1},
        'RTRL': {'type': 'rtrl', 'level': 2, 'online': True, 'recurrent_density': 1, 'inout_density': 1},  # No specific args for rtrl
        'SnAp-1': {'type': 'snap', 'level': 1, 'online': True, 'recurrent_density': 1, 'inout_density': 1},
        'SnAp-2 (d=0.6)': {'type': 'snap', 'level': 2, 'online': True, 'recurrent_density': 0.6, 'inout_density': 0.6},
        'LoRA-RTRL': {'type': 'lora_rtrl', 'level': 2, 'online': False, 'recurrent_density': 1, 'inout_density': 1},
        #'LoRA-RTRL2': {'type': 'lora_rtrl', 'level': 2, 'recurrent_density': 1, 'inout_density': 1, 'rank_constraint_w': 2, 'rank_constraint_r': 2},
        #'LoRA-RTRL3': {'type': 'lora_rtrl', 'level': 2, 'recurrent_density': 1, 'inout_density': 1, 'rank_constraint_w': 3, 'rank_constraint_r': 30}
    }

    #'LoRA-RTRL': {'type': 'lora_rtrl', 'level': 2, 'recurrent_density': 1, 'inout_density': 1}
    #'SnAp-2 (d=0.25)': {'type': 'snap', 'level': 2, 'recurrent_density': 0.25, 'inout_density': 1}

    validation_losses_dict = {
        'BPTT': [],
        'TBPTT': [],
        'RTRL': [],
        'SnAp-1': [],
        'SnAp-2 (d=0.6)': [],
        'LoRA-RTRL': [],
        #'LoRA-RTRL2': [],
        #'LoRA-RTRL3': []
    }

    total_training_times = {
        'BPTT': [],
        'TBPTT': [],
        'RTRL': [],
        'SnAp-1': [],
        'SnAp-2 (d=0.6)': [],
        'LoRA-RTRL': [],
        #'LoRA-RTRL2': [],
        #'LoRA-RTRL3': []
    }

    jacobian_init_times = {}

    # Train different models
    for model_name, config in model_configs.items():

        # Set args according to the current model configuration
        algo_type = config['type']
        level = config['level']
        online = config['online']
        recurrent_density = config['recurrent_density']
        inout_density = config['inout_density']
        #rank_constraint_w = config['rank_constraint_w']
        #rank_constraint_r = config['rank_constraint_r']

        if algo_type in ['bptt', 'tbptt']:
            lossFunction = CrossEntropyLoss
            #learning_rate = 1e-2
        elif algo_type == "lora_rtrl":
            lossFunction_pretrain = CrossEntropyLoss
            lossFunction_finetune = CrossEntropyLoss_RTRL
            #learning_rate_pretrain = 1e-2
            #learning_rate_finetune = 5e-3
        else:
            lossFunction = CrossEntropyLoss_RTRL
            #learning_rate = 5e-3
       
        print(f"Running with settings: Model={args.model}, Type={algo_type}, Level={level}, Online={online}, Recurrent Density={recurrent_density}, Input and Output Density={inout_density}")

        if algo_type in ['snap', 'bptt']:
            model = GRU(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    recurrent_density, 
                    inout_density, 
                    level, 
                    lossFunction, 
                    algo_type,
                    seq_length,
                    vocab_size, 
                    embedding_dim, 
                    logEvery, 
                    learning_rate, 
                    online)

            params, losses, validation_losses, epochs_list, total_training_time = model.run(epochs, train_id, validation_id)

            # Compute EMA for validation losses
            alpha = 0.2  # Smoothing factor, can be adjusted between 0 and 1
            validation_loss_ema = compute_ema(validation_losses, alpha)

            # Store the collected validation losses
            validation_losses_dict[model_name] = (epochs_list, validation_loss_ema)
            total_training_times[model_name].append(total_training_time)
            #jacobian_init_times[model_name] = model.jacobian_init_time

        elif algo_type == 'tbptt':
            model = GRU_TBPTT(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    recurrent_density, 
                    inout_density, 
                    level, 
                    lossFunction, 
                    algo_type,
                    seq_length,
                    vocab_size,
                    trunc_length, 
                    embedding_dim, 
                    logEvery, 
                    learning_rate, 
                    online)
            params, losses, validation_losses_tbptt, epochs_list_tbptt, total_training_time_tbptt = model.run(epochs, train_id, validation_id)

            # Compute EMA for validation losses
            alpha = 0.2  # Smoothing factor, can be adjusted between 0 and 1
            validation_loss_ema = compute_ema(validation_losses_tbptt, alpha)

            # Store the collected validation losses
            validation_losses_dict[model_name] = (epochs_list_tbptt, validation_loss_ema)
            total_training_times[model_name].append(total_training_time_tbptt)
            #jacobian_init_times[model_name] = model.jacobian_init_time

        elif algo_type == 'rtrl':
            model = GRU_RTRL(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    recurrent_density, 
                    inout_density, 
                    level, 
                    lossFunction, 
                    algo_type,
                    seq_length,
                    vocab_size, 
                    embedding_dim, 
                    logEvery, 
                    learning_rate, 
                    online)

            params, losses, validation_losses_rtrl, epochs_list_rtrl, total_training_time_rtrl = model.run(epochs, train_id, validation_id)

            # Compute EMA for validation losses
            alpha = 0.2  # Smoothing factor, can be adjusted between 0 and 1
            validation_loss_ema = compute_ema(validation_losses_rtrl, alpha)

            # Store the collected validation losses
            validation_losses_dict[model_name] = (epochs_list_rtrl, validation_loss_ema)
            total_training_times[model_name].append(total_training_time_rtrl)
            #jacobian_init_times[model_name] = model.jacobian_init_time

        elif algo_type == 'lora_rtrl':
            print("Pre-training starts!")
            model = GRU(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    recurrent_density, 
                    inout_density, 
                    level, 
                    lossFunction_pretrain, 
                    algo_type,
                    seq_length,
                    vocab_size, 
                    embedding_dim, 
                    logEvery, 
                    learning_rate, 
                    online)
            params, losses, validation_losses_pretrain, epochs_list_pretrain, total_training_time_pretrain = model.run(epochs_pretrain, pretrain_id, validation_id)  

            # Compute EMA for validation losses
            alpha = 0.2  # Smoothing factor, can be adjusted between 0 and 1
            validation_loss_pretrain_ema = compute_ema(validation_losses_pretrain, alpha)

            # Store the collected validation losses
            #validation_losses_dict[model_name] = (epochs_list_pretrain, validation_loss_pretrain_ema)
            #total_training_times[model_name].append(total_training_time_pretrain)
            #jacobian_init_times[model_name] = model.jacobian_init_time

            # Save the trained model parameters
            np.savez(f'trained_model_wiki_{args.model}_{algo_type}.npz', params=np.array(params))
            print(f'trained_model_wiki_{args.model}_{algo_type} saved successfully.')

            # Load the pre-trained model parameters
            frozen_params = params

            #trained_model = np.load('trained_model_wiki_gru_lora_rtrl.npz')
            #frozen_params = trained_model['params']

            print("-----------------------")
            print("Fine-tuning starts!")
            model = GRU_LORA(key, 
                    output_size, 
                    hidden_size, 
                    batch_size, 
                    recurrent_density, 
                    inout_density, 
                    level, 
                    lossFunction_finetune, 
                    algo_type,
                    seq_length,
                    vocab_size, 
                    embedding_dim,
                    rank_constraint_w,
                    rank_constraint_r,
                    frozen_params, 
                    logEvery, 
                    learning_rate)

            losses, validation_losses_finetune, epochs_list_finetune, total_training_time_finetune = model.run(epochs_finetune, finetune_id, validation_id)

            # Get the last epoch of pre-training
            last_epoch = epochs_pretrain  # Last epoch from pre-training (70 in this case)
            # Adjust fine-tuning epochs to continue from the last pre-training epoch
            epochs_list_finetune_shifted = [e + last_epoch for e in epochs_list_finetune]

            validation_loss_finetune_ema = compute_ema(validation_losses_finetune, alpha)

            # Store the collected validation losses
            # Merge pre-training and fine-tuning values
            updated_epochs = epochs_list_pretrain + epochs_list_finetune_shifted  # Append fine-tuning epochs
            updated_losses = np.concatenate((validation_loss_pretrain_ema, validation_loss_finetune_ema))  # Append fine-tuning losses
            # Store the updated values back
            validation_losses_dict[model_name] = (updated_epochs, updated_losses)
            #validation_losses_dict[model_name].extend(zip(epochs_list_finetune_shifted, validation_loss_finetune_ema))
            total_training_time = total_training_time_pretrain + total_training_time_finetune
            total_training_times[model_name].append(total_training_time)

            #jacobian_init_times[model_name] = model.jacobian_init_time
  
        else:
            print("Error: Training algorithm Type is not provided.")

    print("----------train is done.-----------")
    print("total training time: ", total_training_times)

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
    plt.savefig(f'wiki_loss_gru_32_integrated.png', bbox_inches='tight')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, required=True, help='Type of the model')

    args = parser.parse_args()
    main(args)