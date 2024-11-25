import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel, pipeline
from torch.optim import Adam
from PIL import Image
import numpy as np
from tqdm import tqdm  # For progress bar
import pandas as pd
import torchvision.transforms as transforms
from nlpaug.augmenter.word import SynonymAug, RandomWordAug
import random
from sklearn.metrics import classification_report, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#initialize augmenters
synonym_aug = SynonymAug(aug_src='wordnet', aug_p=0.3) # Synonym replacement
random_word_aug = RandomWordAug(action="insert", aug_p=0.3) # Random word insertion

def augment_text(text, method="synonym"):

    if method == "paraphrase":
        paraphraser = pipeline("text2text-generation", model="t5-small", framework="pt")
        return paraphraser(text, max_length=50, num_beams=5, do_sample=False)[0]['generated_text']
    else:
        return text

# Define the Multimodal Dataset (Image-Text pairs)
class MultimodalDataset(Dataset):
    def __init__(self, image_df, text_df, labels, processor):
        self.images = image_df
        self.texts = text_df
        self.labels = labels
        self.processor = processor

        self.image_transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(15),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                transforms.RandomAdjustSharpness(2, p=0.5)
                                ])
        self.text_augmentation_methods = ["synonym", "random_insert", "paraphrase"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images.iloc[idx]
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

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

        # Apply image transformation
        image = self.image_transform(image)

        # Apply text augmentation
        augment_method = random.choice(self.text_augmentation_methods)
        text = augment_text(text, method=augment_method)

        # Process image and text with CLIP's processor
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding="max_length", truncation=True, max_length=77)

        return inputs, label

# Define the Model using CLIP embeddings with Early Fusion
class EarlyFusionCLIPModel(nn.Module):
    def __init__(self, num_classes):
        super(EarlyFusionCLIPModel, self).__init__()

        # Load the pre-trained CLIP model and processor from HuggingFace
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

        # Get the dimension of the image and text embeddings from the model's output
        # Using dummy input to get the correct embedding size
        with torch.no_grad():
            dummy_image_input = torch.ones(1, 3, 224, 224)  # A dummy image with correct shape
            dummy_text_input = torch.ones(1, 77).to(torch.long)  # A dummy text input with correct shape
            image_embeddings = self.clip_model.get_image_features(pixel_values=dummy_image_input)
            text_embeddings = self.clip_model.get_text_features(input_ids=dummy_text_input)

        # Now we can safely use the embedding size from the model's output
        self.image_embedding_dim = image_embeddings.shape[1]  # Embedding size for images
        self.text_embedding_dim = text_embeddings.shape[1]  # Embedding size for text

        # Final classifier layer
        self.fc1 = nn.Linear(self.image_embedding_dim + self.text_embedding_dim, 100)  # Concatenating image and text features
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, image_input, text_input):
        # Extract image and text embeddings using the CLIP model
        image_embeddings = self.clip_model.get_image_features(**image_input)
        text_embeddings = self.clip_model.get_text_features(**text_input)

        # Normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        # Early Fusion: Concatenate the image and text embeddings
        fused_embeddings = torch.cat([image_embeddings, text_embeddings], dim=-1)

        # Pass through the classifier layer
        logits = self.fc2(self.fc1(fused_embeddings))

        return logits


# Define the training loop
def train_and_evaluate_model(model, train_dataloader, valid_dataloader, test_dataloader, criterion, optimizer):
    num_epochs = 5
    num_classes = 3

    # Training phase
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            inputs, labels = batch
            image_input = {key: value.squeeze(1).to(device) for key, value in inputs.items() if key in ['pixel_values']}
            text_input = {key: value.squeeze(1).to(device) for key, value in inputs.items() if key in ['input_ids', 'attention_mask']}
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(image_input, text_input)

            # Compute loss (assuming classification task)
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader)}")

        # Validation phase

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in valid_dataloader:
                image_input = {key: value.squeeze(1).to(device) for key, value in inputs.items() if
                               key in ['pixel_values']}
                text_input = {key: value.squeeze(1).to(device) for key, value in inputs.items() if
                              key in ['input_ids', 'attention_mask']}
                labels = labels.to(device)
                outputs = model(image_input, text_input)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
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
        for inputs, labels in test_dataloader:
            image_input = {key: value.squeeze(1).to(device) for key, value in inputs.items() if
                            key in ['pixel_values']}
            text_input = {key: value.squeeze(1).to(device) for key, value in inputs.items() if
                            key in ['input_ids', 'attention_mask']}
            labels = labels.to(device)
            outputs = model(image_input, text_input)
            predictions = torch.argmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect data for csv
            all_texts.extend(text_input['input_ids'].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
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

    output_csv = 'early_fusion_results.csv'

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

    # Initialize the CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Create dataset and dataloader
    train_dataset = MultimodalDataset(train_image_data, train_text_data, train_labels, processor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    valid_dataset = MultimodalDataset(valid_image_data, valid_text_data, valid_labels, processor)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

    test_dataset = MultimodalDataset(test_image_data, test_text_data, test_labels, processor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    model = EarlyFusionCLIPModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-5)

    train_and_evaluate_model(model, train_loader, valid_loader, test_loader, criterion, optimizer)
