import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from ModelClass import SimpleGPT2SequenceClassifier

from torch import nn
from torch.optim import Adam
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm





def load_dataset(label_path, train_path, test_path):

    label_ids = pd.read_csv(label_path)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_test = pd.merge(df_test, label_ids, how = 'inner', left_on = ['ArticleId'], right_on = ['ArticleId'])


    return label_ids, df_train, df_test

def load_tokenizer(model_name, padding_side):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_label_encoding():
    labels = { "business": 0, "entertainment": 1, "sport": 2, "tech": 3, "politic" : 4}

    return labels    


def create_train_test_split(df_train, df_test):
    df_train, df_val = train_test_split(df_train, test_size = 0.2, random_state = 412)
    return df_train, df_val, df_test



class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, labels, df):
        self.labels = [labels[label] for label in df['Category']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors="pt") for text in df['Text']]
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def save_model(model , model_directory):


    torch.save(model.state_dict(), model_directory)
    print("Model Saved To Storage")

def train(model, train_data, val_data, learning_rate, epochs, model_directory):
    train, val = Dataset(train_data), Dataset(val_data)
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            
            model.zero_grad()

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1)==train_label).sum().item()
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        
        with torch.no_grad():
            
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                
                output = model(input_id, mask)
                
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1)==val_label).sum().item()
                total_acc_val += acc
                
            print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}")

    save_model(model, model_directory)
    print("Model Saved to " + model_directory)


def evaluate(model_directory, test_data, hidden_size_parameter, num_classes_parameter, max_seq_len_parameter, model_name_parameter):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        
        model = SimpleGPT2SequenceClassifier(hidden_size = hidden_size_parameter, num_classes = num_classes_parameter, max_seq_len = max_seq_len_parameter, gpt_model_name = model_name_parameter)
        model.load_state_dict(torch.load(model_directory))
        model = model.cuda()
        print("Model Loaded")

        
    # Tracking variables
    predictions_labels = []
    true_labels = []
    
    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
            
            # add original labels
            true_labels += test_label.cpu().numpy().flatten().tolist()
            # get predicitons to list
            predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return true_labels, predictions_labels
    


def master_function(args):


    train_path = str(args['Train_Dataset_Path'])
    test_path = str(args['Test_Dataset_Path'])
    label_path = str(args['Label_Ids_Path'])
    LR = float(args['Learning_Rate'])
    EPOCHS = int(args['Epochs'])
    model_directory = str(args['Model_Directory'])

    hidden_size_parameter = int(args['Hidden_Size'])
    num_classes_parameter = int(args['Num_Classes'])
    max_seq_len_parameter = int(args['Max_Seq_Len'])
    model_name_parameter = str(args['Model_Name'])
    padding_side = str(args['Padding_Side'])




    model = SimpleGPT2SequenceClassifier(hidden_size=hidden_size_parameter, num_classes=num_classes_parameter, max_seq_len = max_seq_len_parameter, gpt_model_name= model_name_parameter)
    
    label_ids, df_train, df_test = load_dataset(label_path, train_path, test_path)


    tokenizer = load_tokenizer(model_name_parameter, padding_side)

    labels = get_label_encoding()

    df_train, df_val, df_test = create_train_test_split(df_train, df_test)

    train(model, df_train, df_val, LR, EPOCHS, model_directory)
    true_labels, pred_labels = evaluate(model_directory, df_test, hidden_size_parameter, num_classes_parameter, max_seq_len_parameter, model_name_parameter)
    
    

#df-train = "../data/BBC_News_Train.csv"
# model_directory = "../model/gpt2-text-classifier-model.pt"           
# EPOCHS = 5
# LR = 1e-5
# padding_side = "left"
# model_name = "gpt2"


if __name__ == "__main__":
    
    parser = argparse.ArgumentPArser(description = "Arguments for Training GPT2")
    
    parser.add_argument("--label-ids_path", dest = "Label_Ids_Path", type = str, required = True, help = "Path to labels ids of the text datapoints")
    
    parser.add_argument("--df-train", dest = "Train_Dataset_Path", type = str, required = True, help = "Path to Train Dataset")
    parser.add_argument("--df-test", dest = "Test_Dataset_Path", type = str, required = True, help = "Path to Test Dataset")
    parser.add_argument("--model_directory", dest = "Model_Directory", type = str, required = True, help = "Path to Save Model Object")
    parser.add_argument("--epochs", dest = "Epochs", type = int, required = True, help = "Number of Epochs to Train the Model")
    parser.add_argument("--learning_rate", dest = "Learning_Rate", type = float, required = True, help = "Learning Rate to Train the Model")
    parser.add_argument("--hidden_size", dest = "Hidden_Size", type = int, required = True, help = "Number of Hidden Layers in the Model")
    parser.add_argument("--num_classes", dest = "Num_Classes", type = int, required = True, help = "Number of Classes To Predict by the Model")
    parser.add_argument("--padding_side", dest = "Padding_Side", type = str, required = True, help = "Padding Side To Be Used For Data Preparation")
    parser.add_argument("--max_seq_len", dest = "Max_Seq_Len", type = int, required = True, help = "Max Sequence Length to be Used By The Model")
    parser.add_argument("--model_name", dest = "Model_Name", type = int, required = True, help = "Name Of the Model to Train")

    
    args = parser.parse_args()
    
    master_function(args)
    
    
    
    
    


