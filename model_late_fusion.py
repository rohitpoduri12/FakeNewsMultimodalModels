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
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, pipeline
from tqdm import tqdm  # For progress bar
import random
from sklearn.metrics import classification_report, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def augment_text(text, method="synonym"):

    if method == "paraphrase":
        paraphraser = pipeline("text2text-generation", model="t5-small", framework="pt", device=0)
        return paraphraser(text, max_length=50, num_beams=5, do_sample=False)[0]['generated_text']
    else:
        return text

# Image dataset
class ImageDataset(Dataset):
    def __init__(self, images_df, labels_df, target_size=(224, 224)):
        """
        Args:
            images_df (pd.DataFrame): DataFrame where each row is a NumPy array representing an image.
            labels_df (pd.DataFrame or pd.Series): DataFrame or Series containing the labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_df = images_df  # Images are stored as numpy arrays
        self.labels_df = labels_df  # Labels could be a pandas Series or a DataFrame
        self.target_size = target_size # Target size for padding the images

        self.image_transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(15),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                transforms.RandomAdjustSharpness(2, p=0.5)
                                ])

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

        # Apply image transformation
        image = self.image_transform(image)

        # Convert the image to a PyTorch tensor
        image = transforms.ToTensor()(image)

        return image, label

# Image dataset
class TextDataset(Dataset):
    def __init__(self, text_df, label_df, max_length=20):
        self.text_df = text_df
        self.labels_df = label_df
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.text_augmentation_methods = ["synonym", "random_insert", "paraphrase"]

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.text_df)

    def __getitem__(self, idx):
        text = self.text_df.iloc[idx]
        label = self.labels_df.iloc[idx]

        # Apply text augmentation
        augment_method = random.choice(self.text_augmentation_methods)
        text = augment_text(text, method=augment_method)

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

        self.image_transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(15),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                transforms.RandomAdjustSharpness(2, p=0.5)
                                ])

        self.text_augmentation_methods = ["synonym", "random_insert", "paraphrase"]

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

        # Apply image transformation (not needed for test)
        #image = self.image_transform(image)

        # Convert the image to a PyTorch tensor
        image = transforms.ToTensor()(image)

        # Apply text augmentation (not needed for test)
        #augment_method = random.choice(self.text_augmentation_methods)
        #text = augment_text(text, method=augment_method)

        # Tokenize the text using BERT tokenizer
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_attention_mask=True, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return image, input_ids, attention_mask, text, label

class LateFusion(nn.Module):
    def __init__(self, resnet_model, bert_model):
        super().__init__()
        self.resnet_model = resnet_model

        # Freeze resnet
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        self.bert_model = bert_model

        # Freeze bert
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, image, input_ids, attention_mask, label):
        resnet_output = self.resnet_model(image)
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)
        op = torch.maximum(resnet_output, bert_output.logits)
        return op

# Train the image part
def train_image_network(train_dataloader, valid_dataloader):

    # Use a pre-trained ResNet model
    model = models.resnet18(pretrained=True)

    # Modify the final layer for the specific number of classes (e.g., 10 classes)
    num_classes = 3  # Number of unique classes in the labels
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

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

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader)}")

        # Validation phase

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, lbls in valid_dataloader:
                images = images.to(device)
                lbls = lbls.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(valid_dataloader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    return model

# Train the text part
def train_text_network(train_dataloader, valid_dataloader):

    num_classes = 3  # Number of unique classes in the labels

    # Initialize Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=num_classes)
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

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
            running_loss += loss.item()

            # Backward pass (compute gradients)
            loss.backward()

            # Update model parameters
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader)}")

        # Validation phase

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for input_ids, attention_mask, lbls in valid_dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                lbls = lbls.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lbls)
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                loss = outputs.loss
                val_loss += loss.item()
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(valid_dataloader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    return model

def late_fusion_output(test_combined_dataloader, resnet_model, bert_model):
    num_classes = 3
    model = LateFusion(resnet_model, bert_model)
    model.to(device)

    model.eval()
    correct = 0
    total = 0
    all_texts = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for image, input_ids, attention_mask, text, lbls in tqdm(test_combined_dataloader, desc=f"Combined test evaluation"):
            # Move inputs to the correct device
            image = image.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            lbls = lbls.to(device)

            outputs = model(image, input_ids, attention_mask, lbls)
            predictions = torch.argmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

            # Collect data for csv
            all_texts.extend(text)
            all_labels.extend(lbls.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    # Calculate test accuracy and loss
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Calculate overall accuracy
    accuracy = accuracy_score(all_labels, all_predictions)

    # Generate class-wise statistics
    class_report = classification_report(
                    all_labels,
                    all_predictions,
                    target_names = [f"Class {i}" for i in range(num_classes)],
                    output_dict=True)

    # Print statistics
    print(f"Overall Accuracy: {accuracy: .4f}")
    print("Prediction Statistics for Each Class:")
    for class_name, stats in class_report.items():
        if class_name in ["accuracy", "macro_avg", "weighted_avg"]:
            continue
        print(f"{class_name}: Precision={stats['precision']:.4f}, Recall={stats['recall']:.4f}, F1-Score={stats['f1-score']:.4f}")

    # Convert data to Dataframe
    df = pd.DataFrame({
        "Input Text (Tokenized)": all_texts,
        "Ground Truth Label": all_labels,
        "Predicted Label": all_predictions
        })

    output_csv = 'late_fusion_results.csv'

    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == '__main__':

    train_df = pd.read_pickle('train_df.pkl')
    test_df = pd.read_pickle('test_df.pkl')
    validate_df = pd.read_pickle('validate_df.pkl')

    #train_df = pd.read_pickle('small.pkl')
    #test_df = pd.read_pickle('small.pkl')
    #validate_df = pd.read_pickle('small.pkl')

    train_image_data = train_df['Image_Data']
    train_text_data = train_df['clean_title']
    train_labels = train_df['3_way_label']

    valid_image_data = validate_df['Image_Data']
    valid_text_data = validate_df['clean_title']
    valid_labels = validate_df['3_way_label']

    test_image_data = test_df['Image_Data']
    test_text_data = test_df['clean_title']
    test_labels = test_df['3_way_label']

    # Instantiate the image train dataset
    train_image_dataset = ImageDataset(images_df=train_image_data, labels_df=train_labels)
    # Create DataLoader
    train_image_dataloader = DataLoader(train_image_dataset, batch_size=32, shuffle=True)

    # Instantiate the image test dataset
    test_image_dataset = ImageDataset(images_df=test_image_data, labels_df=test_labels)
    # Create DataLoader
    test_image_dataloader = DataLoader(test_image_dataset, batch_size=32, shuffle=True)

    # Instantiate the image valid dataset
    valid_image_dataset = ImageDataset(images_df=valid_image_data, labels_df=valid_labels)
    # Create DataLoader
    valid_image_dataloader = DataLoader(valid_image_dataset, batch_size=32, shuffle=True)

    # Instantiate the text train dataset
    train_text_dataset = TextDataset(text_df=train_text_data, label_df=train_labels)
    # Create DataLoader
    train_text_dataloader = DataLoader(train_text_dataset, batch_size=32, shuffle=True)

    # Instantiate the text test dataset
    test_text_dataset = TextDataset(text_df=test_text_data, label_df=test_labels)
    # Create DataLoader
    test_text_dataloader = DataLoader(test_text_dataset, batch_size=32, shuffle=True)

    # Instantiate the text valid dataset
    valid_text_dataset = TextDataset(text_df=valid_text_data, label_df=valid_labels)
    # Create DataLoader
    valid_text_dataloader = DataLoader(valid_text_dataset, batch_size=32, shuffle=True)

    image_model = train_image_network(train_image_dataloader, valid_image_dataloader)
    text_model = train_text_network(train_text_dataloader, valid_text_dataloader)

    # Instantiate the combined test dataset
    test_combined_dataset = CombinedDataset(images_df=test_image_data, text_df=test_text_data, labels_df=test_labels)
    # Create DataLoader
    test_combined_dataloader = DataLoader(test_combined_dataset, batch_size=32, shuffle=True)

    #image_model = models.resnet18(pretrained=True)
    #image_model.fc = nn.Linear(image_model.fc.in_features, 3)

    #text_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    #print(image_model.fc.in_features)
    #print(text_model.classifier.in_features)
    #print(text_model.config.hidden_size)

    late_fusion_output(test_combined_dataloader, image_model, text_model)
