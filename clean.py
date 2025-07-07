import pandas as pd
import openai
from tqdm import tqdm  # Install tqdm if not already installed: pip install tqdm
import csv
import os

# Set the OpenAI API key
openai.api_key = 'your APIkey'

# Load the entire dataset
input_file = 'cleaned_master_corpus.csv'  # Ensure this file is cleaned and preprocessed
output_file = 'reframed_articles_full.csv'

# Load the cleaned dataset
df = pd.read_csv(input_file, delimiter=';', on_bad_lines='skip')

# Ensure there are no missing or empty 'text' values
df = df.dropna(subset=['text'])  # Drop rows with missing 'text'
df = df[df['text'].str.strip() != '']  # Drop rows where 'text' is empty

def reframe_article_chat(article_text, emotion):
    """
    Reframes the input article text to a specific emotion using OpenAI GPT.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Please reframe the following text to express a {emotion} sentiment. Retain the original meaning and structure while incorporating the emotional tone.",
                },
                {"role": "user", "content": article_text},
            ],
            max_tokens=500,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Failed to reframe: {str(e)}"

# Process the dataset and add reframed text for each emotion
emotions = ['angry', 'hopeful', 'fearful']

# Add columns for each emotion
for emotion in emotions:
    print(f"Reframing for '{emotion}' sentiment...")
    df[f'text_{emotion}'] = [
        reframe_article_chat(article_text, emotion) for article_text in tqdm(df['text'], desc=f"Processing {emotion}")
    ]
    print(f"Reframing for '{emotion}' completed.\n")

# Save the reframed dataset to a new CSV file
if os.path.exists(output_file):
    os.remove(output_file)  # Remove existing file to avoid appending
    print(f"Existing file '{output_file}' has been removed.")

df.to_csv(output_file, index=False, sep=';', quoting=csv.QUOTE_MINIMAL)
print(f"Reframed articles have been saved to '{output_file}'.")