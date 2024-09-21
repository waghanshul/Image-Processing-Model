import os
import pandas as pd
from PIL import Image
import numpy as np
from skimage import color
import requests
from io import BytesIO
import aiohttp
import asyncio
import pytesseract
import nest_asyncio

# Apply nest_asyncio to allow running asyncio within Jupyter/Colab
nest_asyncio.apply()

# Path to the CSV file
csv_file_path = '/content/sample_test.csv'

# Load the CSV file into a pandas DataFrame
if not os.path.exists(csv_file_path):
    print(f"Error: The file {csv_file_path} does not exist. Please check the path.")
else:
    data = pd.read_csv(csv_file_path)

    # Display the first few rows to inspect the data
    print("CSV Data:")
    print(data.head())

    # Define the directory to save grayscale images and OCR text
    grayscale_directory = '/content/grayscale_images'
    text_directory = '/content/ocr_texts'
    if not os.path.exists(grayscale_directory):
        os.makedirs(grayscale_directory)
    if not os.path.exists(text_directory):
        os.makedirs(text_directory)

    # Function to preprocess and convert images to grayscale and perform OCR
    async def preprocess_and_convert_to_grayscale(session, image_url, image_save_path, text_save_path):
        try:
            async with session.get(image_url) as response:
                if response.status == 200:
                    img_bytes = await response.read()
                    img = Image.open(BytesIO(img_bytes))
                    
                    # Check if the image has 3 channels (RGB)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img_array = np.array(img)
                    
                    # Ensure the image is not empty
                    if img_array.size == 0:
                        print(f"Image at {image_url} is empty.")
                        return
                    
                    # Convert to grayscale
                    grayscale_img = color.rgb2gray(img_array)
                    grayscale_img_pil = Image.fromarray((grayscale_img * 255).astype(np.uint8))
                    grayscale_img_pil.save(image_save_path)
                    print(f"Grayscale image saved at {image_save_path}")
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(grayscale_img_pil)
                    with open(text_save_path, 'w') as text_file:
                        text_file.write(text)
                    print(f"OCR text saved at {text_save_path}")

                else:
                    print(f"Failed to retrieve image from {image_url}. HTTP Status Code: {response.status}")
        except Exception as e:
            print(f"Failed to process {image_url}: {e}")

    async def main():
        async with aiohttp.ClientSession() as session:
            tasks = []
            for idx, row in data.iterrows():
                image_url = row['image_link']
                grayscale_save_path = os.path.join(grayscale_directory, f"grayscale_{idx}.jpg")
                text_save_path = os.path.join(text_directory, f"text_{idx}.txt")
                task = preprocess_and_convert_to_grayscale(session, image_url, grayscale_save_path, text_save_path)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
