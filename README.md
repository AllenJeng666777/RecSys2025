# Overview:
The project investigates how emotionally reframing news content with large language models (LLMs) influences user preferences and engagement in news recommender systems. By applying GPT-4 to rewrite titles and articles into distinct emotional frames, the study analyzes how emotional tone affects click behavior, time spent reading, and article selection.

# Files:

clean.py: the Python script used for reframing the articles. The text column of the dataset is sent through the API key to GPT, which responds with a new CSV file containing new columns with the reframed texts.

reframe_title.py: Reframes article titles in the same way as clean.py. The title column is sent to GPT, which reframes the titles into different emotional frames.

complete_reframed.csv, reframed_articles_full.csv, reframed_title_txt.csv: The different datasets we use when performing the data analysis.
