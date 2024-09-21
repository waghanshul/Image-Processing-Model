import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('C:/Users/dell/OneDrive/Desktop/test1.csv')

# Function to download an image from a URL
def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# Function to plot images with their metadata
def plot_images_with_metadata(df, num_images=64):
    plt.figure(figsize=(10, 10))
    
    for i in range(num_images):
        plt.subplot(8, 8, i + 1)
        img = download_image(df.loc[i, 'image_link'])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.xlabel(df.loc[i, 'entity_name'])
    
    plt.show()

# Plot a batch of 64 images
plot_images_with_metadata(df, num_images=64)
