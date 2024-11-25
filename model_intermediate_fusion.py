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
from transformers import BertTokenizer, pipeline, BertModel
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

# Multimodal Dataset
class CombinedDataset(Dataset):
    def __init__(self, images_df, text_df, labels_df, target_size=(224, 224), max_length=20):
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
        image = self.image_transform(image)

        # Convert the image to a PyTorch tensor
        image = transforms.ToTensor()(image)

        # Apply text augmentation (not needed for test)
        augment_method = random.choice(self.text_augmentation_methods)
        text = augment_text(text, method=augment_method)

        # Tokenize the text using BERT tokenizer
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_attention_mask=True, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return image, input_ids, attention_mask, text, label

class MultimodalModel(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(text_dim + image_dim, hidden_dim)  # Concatenate text and image features
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, text_features, image_features):
        # Fuse features by concatenating
        fused_features = torch.cat((text_features, image_features), dim=1)

        # Pass through the hidden layers
        x = self.fc1(fused_features)
        x = self.relu(x)
        x = self.fc2(x)  # Final output (logits for classification)
        return x

# Image dataset
def train_and_evaluate_model(train_dataloader, valid_dataloader, test_dataloader):

    # Create the multimodal model
    hidden_dim = 512
    text_dim = 768  # BERT output size
    image_dim = 512  # ResNet18 feature size
    num_classes = 3  # Number of classes
    model = MultimodalModel(text_dim, image_dim, hidden_dim, num_classes)
    model.to(device)

    # Initialize models
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    resnet_model = models.resnet18(pretrained=True)
    resnet_model.fc = nn.Identity()  # Remove the final classification layer

    bert_model.to(device)
    resnet_model.to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            image, input_ids, attention_mask, text, lbls = batch
            image = image.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            lbls = lbls.to(device)

            # Get BERT text embeddings
            with torch.no_grad():
                text_features = bert_model(input_ids=input_ids,
                                        attention_mask=attention_mask).last_hidden_state.mean(dim=1)

            # Get image embeddings from ResNet
            with torch.no_grad():
                image_features = resnet_model(image)

            # Forward pass through the multimodal model
            optimizer.zero_grad()
            outputs = model(text_features, image_features)

            # Compute loss
            loss = criterion(outputs, lbls)
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader)}")


        # Validation phase

        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for image, input_ids, attention_mask, text, lbls in tqdm(valid_dataloader, desc="Validation"):
                # Move data to device
                image = image.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                lbls = lbls.to(device)

                # Get BERT text embeddings
                text_features = bert_model(input_ids=input_ids,
                                           attention_mask=attention_mask).last_hidden_state.mean(dim=1)

                # Get image embeddings from ResNet
                image_features = resnet_model(image)

                # Forward pass through the multimodal model
                outputs = model(text_features, image_features)

                # Compute loss
                loss = criterion(outputs, lbls)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == lbls).sum().item()
                total_preds += lbls.size(0)

        val_accuracy = 100 * correct_preds / total_preds
        avg_val_loss = val_loss / len(valid_dataloader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


    # Test Phase
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    all_texts = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for image, input_ids, attention_mask, text, lbls in tqdm(test_dataloader, desc="Test phase"):
            image = image.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            lbls = lbls.to(device)

            # Get BERT text embeddings
            text_features = bert_model(input_ids=input_ids,
                                       attention_mask=attention_mask).last_hidden_state.mean(dim=1)

            # Get image embeddings from ResNet
            image_features = resnet_model(image)

            # Forward pass through the multimodal model
            outputs = model(text_features, image_features)

            loss = criterion(outputs, lbls)
            test_loss += loss.item()

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
    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

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

    output_csv = 'intermediate_fusion_results.csv'

    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == '__main__':

    train_df = pd.read_pickle('train_df_10.pkl')
    test_df = pd.read_pickle('test_df_10.pkl')
    validate_df = pd.read_pickle('validate_df_10.pkl')

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

    # Instantiate the train dataset
    train_dataset = CombinedDataset(images_df=train_image_data, text_df=train_text_data, labels_df=train_labels)
    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Instantiate the test dataset
    test_dataset = CombinedDataset(images_df=test_image_data, text_df=test_text_data, labels_df=test_labels)
    # Create DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Instantiate the valid dataset
    valid_dataset = CombinedDataset(images_df=valid_image_data, text_df=valid_text_data, labels_df=valid_labels)
    # Create DataLoader
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    # Train and evaluate model
    train_and_evaluate_model(train_dataloader, valid_dataloader, test_dataloader)