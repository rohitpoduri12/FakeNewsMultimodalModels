import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
import os

# Function to load image and convert to NumPy array
def image_to_array_old(image_path):
    img = Image.open(image_path)  # Open the image using PIL
    return np.array(img)  # Convert to NumPy array

# Function to load the image and convert it into a NumPy array
def image_to_array(image_path):
    try:
        with Image.open(image_path) as img:
            downsample_img = img.resize((img.width // 2, img.height // 2))
            img_array = np.array(downsample_img)
        return img_array
    except (UnidentifiedImageError, IOError):
        return None

def dataframe_to_pkl(tsv_dir, pkl_file_name):

    image_dir = "../public_images/public_image_set"
    df = pd.read_csv(tsv_dir, sep='\t')

    # Select only 10% of frame
    #df = df.head(10000)
    df = df.sample(frac=0.1)
    df = df.reset_index(drop=True)

    print("before")
    print(df.index)
    print(df.columns)
    print(df.shape)
    print(df['id'][0])

    # rename id as image_path, and remove unnecessary columns
    df.rename(columns={'id': 'image_path'}, inplace=True)
    columns_to_keep = ['clean_title', 'image_path', '2_way_label']
    df = df[columns_to_keep]

    df['image_path'] = image_dir + '/' + df['image_path'] + '.jpg'

    print("after")
    print(df.index)
    print(df.columns)
    print(df.shape)
    print(df['image_path'][0])

    # Function to check if the image exists
    df['image_exists'] = df['image_path'].apply(os.path.exists)

    # Filter the DataFrame to keep only rows where the file exists
    df = df[df['image_exists']]

    # Drop the 'image_exists' column, if not needed
    df = df.drop(columns=['image_exists'])

    # Create the image numpy array entry
    df['Image_Data'] = df['image_path'].apply(image_to_array)

    # Remove rows where the numpy array is not None
    df = df[df['Image_Data'].notna()]

    print("after2")
    print(df.index)
    print(df.columns)
    print(df.shape)
    print(df['image_path'][0])

    df.to_pickle(pkl_file_name)

    #img = Image.fromarray(df['Image_Data'][0])
    # Display the image
    #img.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    print(torch.cuda.is_available())

    train_tsv_dir = "../Fakeddit_datasetv2.0/multimodal_only_samples/multimodal_train.tsv"
    test_tsv_dir = "../Fakeddit_datasetv2.0/multimodal_only_samples/multimodal_test_public.tsv"
    validate_tsv_dir = "../Fakeddit_datasetv2.0/multimodal_only_samples/multimodal_validate.tsv"

    train_pkl_file = 'train_df.pkl'
    test_pkl_file = 'test_df.pkl'
    validate_pkl_file = 'validate_df.pkl'

    #dataframe_to_pkl(train_tsv_dir, train_pkl_file)
    #dataframe_to_pkl(test_tsv_dir, test_pkl_file)
    dataframe_to_pkl(validate_tsv_dir, validate_pkl_file)