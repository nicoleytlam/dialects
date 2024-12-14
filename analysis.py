import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import pandas as pd
import seaborn as sns

def plot_filtered_validation_and_test_accuracy(csv_file='accuracy.csv', dataset='standard', config='B'):
    """
    Filters the dataset by the first and second columns and plots the distribution 
    of validation accuracy for each of the four epochs and the final test accuracy.
    
    Parameters:
    - csv_file: str, path to the CSV file
    - dataset: value to filter the first column
    - config: value to filter the second column
    
    Returns:
    - None
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Ensure the file has at least two columns for filtering
    if data.shape[1] < 2:
        raise ValueError("The CSV file does not have enough columns for filtering.")
    
    # Filter the dataset based on the first and second columns
    filtered_data = data[(data.iloc[:, 0] == dataset) & (data.iloc[:, 1] == config)]
    
    # Ensure there is data left after filtering
    if filtered_data.empty:
        print(f"No data found for dataset '{dataset}' and config '{config}'.")
        return
    
    # Columns for epochs and test accuracy 
    epoch_columns = ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4']
    test_column = 'Test Accuracy'
    
    # Reshape the data for visualization
    validation_data = filtered_data[epoch_columns].melt(var_name="Epoch", value_name="Validation Accuracy")
    test_data = pd.DataFrame({
        "Epoch": ["Test"] * len(filtered_data[test_column]), 
        "Validation Accuracy": filtered_data[test_column]
    })

    combined_data = pd.concat([validation_data, test_data], ignore_index=True)
    
    # Plot using violin plot with swarm overlay
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Epoch", y="Validation Accuracy", data=combined_data, inner="quartile", palette="Set2")
    sns.swarmplot(x="Epoch", y="Validation Accuracy", data=combined_data, color="k", alpha=0.5)
    plt.title(f"Validation Accuracy Per Epoch and Test Accuracy\nDataset: '{dataset}', Config: '{config}'")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch/Test")
    plt.tight_layout()
    plt.show()

def plot_boxplots(dataset = 'standard', config = 'B'):
    data = pd.read_csv('accuracy.csv')
    data = data[(data['Dataset'] == dataset)] 
    data = data[(data['Config'] == config)]

    boxplot_data = data.iloc[:, [2, 3, 4, 5]]
    
    # Create box plots
    plt.figure(figsize=(10, 6))
    boxplot_data.boxplot()
    plt.title("Distribution of Validation Accuracy per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Values")

    plt.xticks(ticks=range(1, len(boxplot_data.columns) + 1), labels=boxplot_data.columns, rotation=45)
    plt.tight_layout()
    plt.show()



def test_example(model, vocab, example, target, heatmap: bool = False, 
                 rotate_x_labels: bool = False):
    model.eval()
    example_list = example.split()
    target_list = target.split()    
    encoded_example = torch.LongTensor([[vocab.get_index(w) for w in example.split()]])
    length = len(target_list)
    output, attention = model(encoded_example, maxlen=length)
    output = output.argmax(dim=-1)
    if heatmap:        
        attention_array = attention.detach()[0].T.numpy()
        output_list = [vocab.get_form(i) for i in output[0]]
        plot_heatmap(attention_array, example_list, output_list, target_list, 
                     rotate_x_labels=rotate_x_labels)
    else: 
        print(" ".join(vocab.get_form(i) for i in output[0]))

def test_copy_example(model, vocab, example, heatmap: bool = False):
    test_example(model, vocab, example, example, heatmap=heatmap)
           
def test_reverse_example(model, vocab, example, heatmap: bool = False):
    target = " ".join(example.split()[::-1])
    test_example(model, vocab, example, target, heatmap=heatmap)    

def test_tense_example(model, vocab, example, target, heatmap: bool = False):
     test_example(model, vocab, example, target, heatmap=heatmap,
                   rotate_x_labels=True)

def plot_heatmap(array, input, output, target, rotate_x_labels: bool = False):
        plt.imshow(array, cmap='hot')
        plt.xlabel("Output")
        plt.ylabel("Input")        
        plt.yticks(np.arange(len(input)), input)  
        if rotate_x_labels:
             plt.xticks(np.arange(len(output)), output, rotation = 90)
        else:
            plt.xticks(np.arange(len(output)), output)
        output_correct = [output[i] == target[i] 
                          for i in range(len(output))]
        tick_colors = map(lambda x: 'g' if x else 'r', 
                         output_correct)                          
        for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), tick_colors):
            ticklabel.set_color(tickcolor)
        plt.colorbar()
        plt.show()

def test_verbs(model, vocab, data):
    verbs = ['giggles', 'smiles', 	'sleeps', 'swims', 'waits', 'moves', 'changes',
              'reads', 'eats', 'entertains', 'amuses', 'high_fives', 'applauds', 'confuses', 'admires', 'accepts', 'remembers', 'comforts', 'giggle', 'smile', 'sleep', 'swim', 'wait', 'move', 'change', 'read', 'eat', 'entertain', 'amuse', 'high_five', 'applaud', 'confuse', 'admire', 'accept', 'remember', 'comfort']
    verb_indices =  [vocab.get_index(w) for w in verbs]
    total_verbs = 0
    correct_verbs = 0
    correct_sentences = []
    incorrect_sentences = []
    model.eval()
    for input, target in data.get_batches(1):
        verb_positions = np.where(np.isin(target[0], verb_indices))[0]
        if len(verb_positions) > 0:
            total_verbs += len(verb_positions)
            output, _ = model(input, target)
            output = output[0].argmax(dim=1)
            target = target.squeeze()
            if (output[verb_positions] == target[verb_positions]).any():
                i_sent = " ".join([vocab.get_form(i) for i in input[0]])
                t_sent = " ".join([vocab.get_form(i) for i in target])
                o_sent = " ".join([vocab.get_form(i) for i in output])
                if (target == output).all():
                    correct_sentences += [(i_sent, t_sent)]

                else:
                    incorrect_sentences += [(i_sent, t_sent)]
            correct_verbs += sum(output[verb_positions] == target[verb_positions])
    print("{} verbs correct out of {} ({:.1f}%)".format(correct_verbs.item(), 
                                                        total_verbs, 
                                                        (correct_verbs/total_verbs * 100).item()))
    print("{} sentences perfect out of {} ({:.1f}%)".format(len(correct_sentences), 
                                                            len(data),
                                                            (len(correct_sentences)/len(data))* 100))
    return correct_sentences, incorrect_sentences
         
         
     