import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as transforms
import argparse
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm  # For progress bar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image dataset
class ImageDataset(Dataset):
    def __init__(self, images_df, labels_df, transform=None, target_size=(224, 224)):
        """
        Args:
            images_df (pd.DataFrame): DataFrame where each row is a NumPy array representing an image.
            labels_df (pd.DataFrame or pd.Series): DataFrame or Series containing the labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_df = images_df  # Images are stored as numpy arrays
        self.labels_df = labels_df  # Labels could be a pandas Series or a DataFrame
        self.transform = transform  # Optional transform (e.g., normalization, augmentation)
        self.target_size = target_size # Target size for padding the images

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images_df)

    def __getitem__(self, idx):
        """Fetch a single sample at the specified index."""
        # Get the image (NumPy array) at the index
        image = self.images_df.iloc[idx]  # Convert row to numpy array
        label = self.labels_df.iloc[idx]  # Fetch the label at the index

        # Ensure image is a valid NumPy array with correct dimensions
        if isinstance(image, np.ndarray):
            # Check if the image has 4 channels (RGBA) and convert to RGB by removing the alpha channel
            if image.ndim == 3 and image.shape[2] == 4:
                image = image[:, :, :3]  # Select the first 3 channels (RGB)
            # Convert grayscale to RGB
            elif image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            # Check if the image is 3D (height, width, channels)
            elif image.ndim != 3 or image.shape[2] != 3:
                print(image.ndim)
                print(image.shape)
                raise ValueError(f"Image at index {idx} does not have the expected shape (height, width, 3) or (height, width, 4).")
            # Convert NumPy array to PIL Image
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        else:
            raise ValueError(f"Image at index {idx} is not a valid NumPy array.")

        # Convert NumPy array to PIL Image
        #image = Image.fromarray(image.astype('uint8'), 'RGB')

        # Resize the image to the target size (TODO: check this properly whether do resizing or padding)
        image = image.resize(self.target_size)

        # Convert the image to a PyTorch tensor
        image = transforms.ToTensor()(image)

        # Apply any transformations (e.g., normalization, augmentation)
        if self.transform:
            image = self.transform(image)

        return image, label

# Image dataset
class TextDataset(Dataset):
    def __init__(self, text_df, label_df, max_length=120):
        self.text_df = text_df
        self.labels_df = label_df
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.text_df)

    def __getitem__(self, idx):
        text = self.text_df.iloc[idx]
        label = self.labels_df.iloc[idx]

        # Tokenize the text using BERT tokenizer
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_attention_mask=True, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return input_ids, attention_mask, label

class CombinedDataset(Dataset):
    def __init__(self, images_df, text_df, labels_df, target_size=(224, 224), max_length=120):
        self.images_df = images_df  # Images are stored as numpy arrays
        self.labels_df = labels_df  # Labels could be a pandas Series or a DataFrame
        self.target_size = target_size # Target size for padding the images
        self.text_df = text_df
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images_df)

    def __getitem__(self, idx):

        """Fetch a single sample at the specified index."""
        # Get the image (NumPy array) at the index
        image = self.images_df.iloc[idx]  # Convert row to numpy array
        text = self.text_df.iloc[idx]
        label = self.labels_df.iloc[idx]  # Fetch the label at the index

        # Ensure image is a valid NumPy array with correct dimensions
        if isinstance(image, np.ndarray):
            # Check if the image has 4 channels (RGBA) and convert to RGB by removing the alpha channel
            if image.ndim == 3 and image.shape[2] == 4:
                image = image[:, :, :3]  # Select the first 3 channels (RGB)
            # Convert grayscale to RGB
            elif image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            # Check if the image is 3D (height, width, channels)
            elif image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Image at index {idx} does not have the expected shape (height, width, 3) or (height, width, 4).")
            # Convert NumPy array to PIL Image
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        else:
            raise ValueError(f"Image at index {idx} is not a valid NumPy array.")

        # Resize the image to the target size (TODO: check this properly whether do resizing or padding)
        image = image.resize(self.target_size)

        # Convert the image to a PyTorch tensor
        image = transforms.ToTensor()(image)

        # Tokenize the text using BERT tokenizer
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_attention_mask=True, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return image, input_ids, attention_mask, label

class LateFusion(nn.Module):
    def __init__(self, resnet_model, bert_model):
        super().__init__()
        self.resnet_model = resnet_model
        resnet_output_size = 100
        self.resnet_model.fc = nn.Linear(resnet_model.fc.in_features, resnet_output_size)

        # Freeze resnet
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        #bert_output_size = 100
        #bert_model.classifier = torch.nn.Linear(bert_model.config.hidden_size, bert_output_size)
        self.bert_model = bert_model.bert
        bert_output_size = bert_model.classifier.in_features

        # Freeze bert
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(resnet_output_size+bert_output_size, 1)

    def forward(self, image, input_ids, attention_mask, label):
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)
        cls_vector = bert_output.pooler_output
        resnet_output = self.resnet_model(image)

        #print(bert_output)
        #print(cls_vector.shape)
        #print(resnet_output.shape)

        op = self.linear(torch.cat((cls_vector, resnet_output), dim=1))

        return op

# Train the image part
def train_image_network(image_data, labels):

    # Instantiate the custom dataset
    train_dataset = ImageDataset(images_df=image_data, labels_df=labels)

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Example usage: iterate through the data
    #for images, labels in train_dataloader:
    #    print(f"Images batch shape: {images.shape}")
    #    print(f"Labels batch shape: {labels.shape}")

    # Use a pre-trained ResNet model
    model = models.resnet18(pretrained=True)

    # Modify the final layer for the specific number of classes (e.g., 10 classes)
    num_classes = len(labels.unique())  # Number of unique classes in the labels
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_dataloader, desc=f"Image Training Epoch {epoch + 1}"):
            # Move images and labels to the same device as the model (GPU or CPU)
            images, lbls = [x.to(device) for x in batch]

            # Zero the gradients
            optimizer.zero_grad()

            #print(f"Model device: {next(model.parameters()).device}")
            #print(f"Input device: {images.device}")

            # Forward pass
            outputs = model(images)
            #outputs.to(device)

            # Compute the loss
            loss = criterion(outputs, lbls)

            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()

            # For statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()
            print("loss is ", loss.item())

        # Print statistics after each epoch
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return model


# Train the text part
def train_text_network(text_data, labels):
    # Instantiate the custom dataset
    train_dataset = TextDataset(text_df=text_data, label_df=labels)

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_classes = len(labels.unique())  # Number of unique classes in the labels

    # Initialize Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=num_classes)
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 2

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0

        # Training
        for batch in tqdm(train_dataloader, desc=f"Text Training Epoch {epoch + 1}"):
            # Move inputs to the correct device
            input_ids, attention_mask, lbls = [x.to(device) for x in batch]

            # Zero gradients from previous step
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lbls)

            # Get loss
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass (compute gradients)
            loss.backward()

            # Update model parameters
            optimizer.step()

            print("loss is ", loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}: Average Training Loss: {avg_train_loss}")

        # Validation
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        correct_predictions = 0

        with torch.no_grad():  # Disable gradient calculation for validation
            for batch in tqdm(train_dataloader, desc=f"Text Evaluating Epoch {epoch + 1}"):
                input_ids, attention_mask, lbls = [x.to(device) for x in batch]

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lbls)

                # Get loss and predictions
                loss = outputs.loss
                logits = outputs.logits
                total_val_loss += loss.item()

                # Get predicted class
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == lbls).sum().item()

        avg_val_loss = total_val_loss / len(train_dataloader)
        accuracy = correct_predictions / len(train_dataset)
        print(f"Epoch {epoch + 1}: Average Validation Loss: {avg_val_loss}, Accuracy: {accuracy * 100}%")

    return model

def train_late_fusion(image_data, text_data, labels, resnet_model, bert_model):

    # Instantiate the custom dataset
    train_dataset = CombinedDataset(images_df=image_data, text_df=text_data, labels_df=labels)

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = LateFusion(resnet_model, bert_model)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        epoch_acc = []
        epoch_loss = []

        # Training
        for batch in tqdm(train_dataloader, desc=f"Combined Training Epoch {epoch + 1}"):

            # Move inputs to the correct device
            image, input_ids, attention_mask, lbls = [x.to(device) for x in batch]

            # Ensure labels are of type float for BCEWithLogitsLoss
            lbls = lbls.float()  # Convert labels to float

            outputs = model(image, input_ids, attention_mask, lbls)
            preds = outputs > 0.5

            acc = (preds.squeeze() == lbls).float().sum() / len(lbls)
            epoch_acc.append(acc.item())

            #print("output shape in late fusion is ", outputs.shape)
            #print("output type in late fusion is ", outputs.type())
            #print("labels shape in late fusion is ", lbls.shape)
            #print("labels type in late fusion is ", lbls.type())

            loss = criterion(outputs.squeeze(), lbls)
            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        print(f'\nEpoch: {epoch + 1}/{num_epochs} done, loss: {np.mean(epoch_loss)}, acc: {np.mean(epoch_acc)}')

if __name__ == '__main__':

    train_df = pd.read_pickle('train_df.pkl')
    #test_df = pd.read_pickle('test_df.pkl')
    #validate_df = pd.read_pickle('validate_df.pkl')

    df_under_review = train_df
    df = df_under_review
    #df = df_under_review.head(5000)
    print(df.index)
    print(df.columns)
    print(df.shape)
    print(df['image_path'][0])
    print(df['clean_title'][0])
    print(df['2_way_label'][0])

    image_data = df['Image_Data']
    text_data = df['clean_title']
    labels = df['2_way_label']

    print(image_data.shape)
    print(text_data.shape)
    print(labels.shape)

    image_model = train_image_network(image_data, labels)
    text_model = train_text_network(text_data, labels)

    #image_model = models.resnet18(pretrained=True)
    #image_model.fc = nn.Linear(image_model.fc.in_features, 5)

    #text_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
    #print(image_model.fc.in_features)
    #print(text_model.classifier.in_features)
    #print(text_model.config.hidden_size)

    train_late_fusion(image_data, text_data, labels, image_model, text_model)
