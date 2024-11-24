import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from torch.optim import Adam
from PIL import Image
import numpy as np
from tqdm import tqdm  # For progress bar
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Multimodal Dataset (Image-Text pairs)
class MultimodalDataset(Dataset):
    def __init__(self, image_df, text_df, labels, processor):
        self.images = image_df
        self.texts = text_df
        self.labels = labels
        self.processor = processor

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
        self.fc = nn.Linear(self.image_embedding_dim + self.text_embedding_dim, num_classes)  # Concatenating image and text features

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
        logits = self.fc(fused_embeddings)

        return logits


# Define the training loop
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    num_epochs = 2

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
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

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        print("avg_loss in epoch %d is %f", epoch, avg_loss)
        print("accuracy in epoch %d is %f", epoch, accuracy)


if __name__ == '__main__':

    train_df = pd.read_pickle('train_df.pkl')
    #test_df = pd.read_pickle('test_df.pkl')
    #validate_df = pd.read_pickle('validate_df.pkl')

    df = train_df
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

    # Initialize the CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Create dataset and dataloader
    train_dataset = MultimodalDataset(image_data, text_data, labels, processor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = EarlyFusionCLIPModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-5)

    train_model(model, train_loader, criterion, optimizer)
