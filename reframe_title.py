import pandas as pd
import openai
from tqdm import tqdm
import os
import requests
from PIL import Image
from io import BytesIO
import csv

# Set the OpenAI API key
openai.api_key = 'your APIkey'

# Load the dataset
input_file = 'reframed_title_txt.csv'
output_file = 'complete_reframed.csv'
df = pd.read_csv(input_file, delimiter=';', on_bad_lines='skip')

# Drop rows with missing or empty 'title'
df = df.dropna(subset=['title'])
df = df[df['title'].str.strip() != '']

# Create an image output folder
IMAGE_FOLDER = os.path.join(os.getcwd(), 'img_reframe')
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Define a function to reframe the title
def reframe_title_chat(title_text, emotion):
    """
    Reframes the input title text to a specific emotion using OpenAI GPT.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Please rewrite the following headline to express a {emotion} sentiment. Keep it concise and engaging.",
                },
                {"role": "user", "content": title_text},
            ],
            max_tokens=50,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Failed to reframe: {str(e)}"

# Define a function to generate an image prompt
def generate_image_prompt(title, emotion):
    """
    Generates a text prompt for the image generation model based on the title and emotion.
    """
    sanitized_title = title.replace("'", "").replace('"', "")  # Remove problematic characters
    return f"A visually compelling representation of an {emotion} sentiment based on the headline: {sanitized_title}."

# Define a function to generate and save an image locally
def generate_image_and_save(prompt, article_id, emotion):
    """
    Generates an image using OpenAI's DALL-E API and saves it locally.
    Links the image path to the article ID and emotion.
    """
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"
        )
        image_url = response['data'][0]['url']
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # Save the image locally
            img = Image.open(BytesIO(image_response.content))
            file_name = f"{article_id}_{emotion}.png"
            file_path = os.path.join(IMAGE_FOLDER, file_name)
            img.save(file_path)
            return file_path
        else:
            return f"Failed to download image: HTTP {image_response.status_code}"
    except Exception as e:
        return f"Failed to generate image: {str(e)}"

# Reframe titles and generate images
emotions = ['angry', 'hopeful', 'fearful']

for emotion in emotions:
    print(f"Processing '{emotion}' sentiment...")
    
    # Reframe the titles
    df[f'{emotion}_title'] = [
        reframe_title_chat(title_text, emotion) for title_text in tqdm(df['title'], desc=f"Reframing titles to {emotion}")
    ]
    
    # Generate and save the images locally
    df[f'{emotion}_img'] = [
        generate_image_and_save(
            generate_image_prompt(title, emotion),
            article_id,
            emotion
        )
        for article_id, title in tqdm(zip(df['id'], df[f'{emotion}_title']), total=len(df), desc=f"Generating {emotion} images")
    ]
    print(f"Processing for '{emotion}' completed.\n")

# Ensure the output file is cleared
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"Existing file '{output_file}' has been removed.")

# Save the updated dataset
df.to_csv(output_file, index=False, sep=';', quoting=csv.QUOTE_MINIMAL)
print(f"Reframed titles and generated images have been saved to '{output_file}' using ';' as the delimiter.")
