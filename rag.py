import feedparser
import warnings
import logging
import sys
import re
import json
import shutil
import pathlib
import torch
import transformers
import openai
import pandas as pd
import fitz  # PyMuPDF
import glob
from tqdm.notebook import tqdm
import requests
from urllib.parse import urlencode
from sklearn.neighbors import NearestNeighbors
from IPython.display import Markdown, display
from xml.etree import ElementTree
from bs4 import BeautifulSoup
import feedparser
import numpy as np
from collections import OrderedDict
import subprocess
from collections import defaultdict
import os
import pandas as pd
import openai
import itertools
import time

embedding_model_id = "BAAI/bge-small-en-v1.5"

file_path = "/project_data/shared/general_lab_resources/GPT/openai_api_key.txt"
try:
    with open(file_path, 'r') as file:
        lines = file.readline().strip()
    # print(lines)  # This will print a list of lines from the file
    os.environ['OPENAI_API_KEY'] = lines
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    openai.api_key = lines
except FileNotFoundError:
    print(f"The file at {file_path} was not found.")
except IOError:
    print(f"An error occurred while reading the file at {file_path}.")

def compute_embeddings(df, column_to_embed, api=False, embedding_model_id=embedding_model_id):

    texts = df[column_to_embed].tolist()
    all_responses = []

    for i, text in enumerate(texts):

        # print(f"Processing text {i+1}/{len(texts)}")
        text = text.replace('\n', '')
        text = text[:550] # embed just the first 600 characters, otherwise it can exceed models limits
        
        if api:
            api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{embedding_model_id}"
            headers = {"Authorization": f"Bearer {hf_token}"}
            response = requests.post(api_url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}})
            response_json = response.json()
            all_responses.append(response_json)
        else:
            url = "127.0.0.1:8080/embed"
            headers = "Content-Type: application/json"      
            
            data_json = json.dumps({"inputs": [text]})
            command = ["curl", url, "-X", "POST", "-d", data_json, "-H", "Content-Type: application/json"]
            response = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            try:
                batch_embedding = json.loads(response.stdout)
                if not isinstance(batch_embedding, list):
                    print(f"Warning: Response for text {i} is not a list. Response: {batch_embedding}")
                    batch_embedding = [batch_embedding]  # Ensuring it's a list

                if len(batch_embedding) != 1:
                    print(f"Warning: Response for text {i} contains {len(batch_embedding)} items.")

                all_responses.extend(batch_embedding)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON response for text {i}: {e}")
                print(f"Response: {response.stdout}")

    embeddings = pd.concat([
        df,
        pd.DataFrame(all_responses, index=df.index)],
        axis=1
    )

    return embeddings.reset_index(drop=True)

def retrieve_rank_knn_chunks2(query, literature_sentences_embedded, n_knn, n_sentence_hyde):
    
    ## CREATE AND EMBED FOUR SENTENCE ANSWER TO QUERY
    # four_sentence_answer = four_sentence_completion_gpt(query)
    n_sentence_answer = n_sentence_completion_gpt(query, n = n_sentence_hyde)
    subqueries_embedded = compute_embeddings(n_sentence_answer, column_to_embed = "subquery")

    ## RUN NEAREST NEIGHBORS
    literature_features = literature_sentences_embedded.iloc[:, 2:]
    nn = NearestNeighbors(n_neighbors=n_knn)  # Set the number of neighbors you want
    for column in literature_features.columns:
        literature_features[column] = pd.to_numeric(literature_features[column], errors='coerce')
    literature_features.dropna(inplace=True)
    nn.fit(literature_features)
    subquery_features = subqueries_embedded.iloc[:, 1:]
    nearest_neighbors_list = []
    for index, row in subquery_features.iterrows():
        distances, indices = nn.kneighbors([row])
        nearest_neighbors = literature_features.iloc[indices[0]].copy()  # Changed from df to literature_features
        nearest_neighbors['query_index'] = index  # Add a column to identify which query row this belongs to
        nearest_neighbors_list.append(nearest_neighbors)

    ## GET PARENT CHUNKS FROM NEAREST NEIGHBOR SENTENCES
    all_nearest_neighbors = pd.concat(nearest_neighbors_list)
    file_paths = literature_sentences_embedded.iloc[all_nearest_neighbors.index.to_list()]['parent_chunk_path']
    all_texts = []
    for file_path in file_paths:

        with open(file_path, 'r') as file:
            doi = file_path.split('/')[-1].replace('.txt', '').replace('-', '/', 1)
            doi = re.sub(r'\.\d+$', '', doi)
            content = file.read()
            formatted_content = f'\n"""\nThe doi for the following information is: {doi}\n{content}\n"""\n'
            all_texts.append(formatted_content)

    output = n_sentence_answer.loc[np.repeat(n_sentence_answer.index, n_knn)].reset_index(drop=True)
    output['hits'] = all_texts
    output = output.drop_duplicates(subset = 'hits')
    output['query'] = query  # This line ensures every row in 'output' has the 'query' column filled with the query string
    
    return output

def n_sentence_completion_gpt(query, n):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Content-Type': 'application/json','Authorization': f'Bearer {OPENAI_API_KEY}'}
    data = {
        # 'model': 'gpt-3.5-turbo',
        'model': 'gpt-4-1106-preview',
        'messages': [
            {'role': 'system', 'content': f'Your job is to create a hypothetical answer that is supposed to resemble the answer to a question that would be found in an academic article for a plant compound species association. Create the Hypothetical answer in {n} sentences. The hypothetical answer should not be a mere yes or no but rather should resemble what information would be found in an academic article to answer the compound~species association. There may be different names for species and compounds than the ones given so try to accommodate for that as well. If I request an answer with more than one sentence, the sentences should be delimited with "~~"'},
            {'role': 'user', 'content': f'Question: {query}'}
        ],'temperature': 0
    }
    response = requests.post(url, json=data, headers=headers).json()
    response = pd.DataFrame({'subquery': response['choices'][0]['message']['content'].split("~~")})
    return response

def generate_system_prompt(combination_keys):
    """Concatenate the selected prompt phrases into a single system prompt."""
    return " ".join([prompt_phrases[key] for key in combination_keys])

import os
import requests
import time
from tqdm import tqdm

def process_dataframe(
    dataframe, 
    prompt_combinations, 
    model="gpt-4o-mini",
    sleep_time=1,
    api_key=None,
    content_columns=None,  # New argument for specifying content columns
    store_content_used=False  # New argument to control storing content
):
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key must be provided either as an argument or set in environment variable 'OPENAI_API_KEY'.")

    if content_columns is None or len(content_columns) != len(prompt_combinations):
        raise ValueError("Content columns must be provided and must match the length of prompt combinations.")

    total_combinations = len(prompt_combinations)
    total_rows = len(dataframe)
    
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    
    for combo_index, (combo, content_column) in enumerate(zip(prompt_combinations, content_columns), start=1):
        combo_name = "_".join(combo)
        column_name = f"gpt_response_{combo_name}~{content_column}"  # Updated to include content column
        
        if store_content_used:
            content_column_name = f"content_used_{combo_name}~{content_column}"  # Updated to include content column
        
        system_prompt = generate_system_prompt(combo)
        print(f"Processing combination {combo_index}/{total_combinations}: {combo_name} with content column '{content_column}'")
        
        responses = []
        content_used = [] if store_content_used else None  # Conditionally initialize list
        
        for i, row in enumerate(tqdm(dataframe.itertuples(index=False), desc=f"Processing Rows for {combo_name}", total=total_rows, leave=False), start=1):
            # Fetch the content for the specified column
            content = getattr(row, content_column)
            
            user_content = {
                "role": "user",
                "content": f"Question: {row.association}\n\nContent: {content}"
            }
            messages = [{"role": "system", "content": system_prompt}, user_content]
            
            data = {
                'model': model,
                'messages': messages,
                'temperature': 0
            }
            
            try:
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    response_json = response.json()
                    reply = response_json['choices'][0]['message']['content'].strip()
                else:
                    print(f"Error processing row {i}: {response.status_code} {response.text}")
                    reply = "Error in generating response."
            except Exception as e:
                print(f"Exception processing row {i}: {e}")
                reply = "Error in generating response."
            
            responses.append(reply)
            if store_content_used:
                content_used.append(content)  # Store the used content if required
            
            time.sleep(sleep_time)
        
        # Add responses to DataFrame
        dataframe[column_name] = responses
        if store_content_used:
            dataframe[content_column_name] = content_used  # Store content for verification if required

    return dataframe


## Only about 10 dollars per 250 records

######## READ IN DATA
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
warnings.simplefilter(action='ignore', category=FutureWarning)
chunk_directory = "/project_data/shared/text_mining/papers_chunked"
csv_files = glob.glob(os.path.join(chunk_directory, '**/*_embedded_sentences.csv'), recursive=True)
dataframes = [pd.read_csv(file) for file in csv_files]
literature_sentences_embedded = pd.concat(dataframes, ignore_index=True)

records = pd.read_csv("/project_data/shared/text_mining/filtered_lit_3000.csv")

i = 1  # Starting index (inclusive)
x = 250  # Ending index (inclusive)
total_records = x - i + 1

results = []

# Iterate over the specified range of records
for index, record in tqdm(records.iloc[i-1:x].iterrows(), total=total_records, desc="Processing Records"):
    species = record['species']
    compound_name = record['compound_name']
    overall_question = f"Is {compound_name} Found in {species}"
    
    # Retrieve subquestion results using your custom function
    subquestion_results = retrieve_rank_knn_chunks2(
        query=overall_question,
        literature_sentences_embedded=literature_sentences_embedded,
        n_knn=3,
        n_sentence_hyde=1
    )
    
    # Initialize lists to collect aggregated data
    aggregated_contents = []
    aggregated_dois = set()  # Set avoids duplicate DOIs
    aggregated_questions = []
    
    # Iterate over each row in the subquestion results
    for _, row in subquestion_results.iterrows():
        content = row['hits']
        aggregated_contents.append(content)
        
        # Extract DOIs using regex
        doi_regex = r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b'
        found_dois = re.findall(doi_regex, content, flags=re.IGNORECASE)
        
        # Normalize and add DOIs to the set
        for doi in found_dois:
            normalized_doi = doi.replace('/', '-')
            aggregated_dois.add(normalized_doi)
        
        # Collect the subquery used
        aggregated_questions.append(row['subquery'])
    
    # Combine all aggregated data into a single dictionary entry
    results.append({
        'association': overall_question,
        'contents': aggregated_contents,  # List of all contents
        'dois': list(aggregated_dois),    # List of unique normalized DOIs
        'compound': record['compound_name'],
        'title': record['title'],
        'abstract': record['abstract'],
        'abstract_doi': record['doi'],
        'manual_evaluation': record['T/F/UC'],
        'questions_used_for_search': aggregated_questions  # List of subqueries
    })

# Optional: Convert results to a DataFrame for easier manipulation
rag_output = pd.DataFrame(results)
rag_output = rag_output.drop_duplicates(subset=['association']).reset_index(drop=True)
rag_output.to_csv("/project_data/shared/ai_in_phytochemistry/vingette_2_RAG/rag_output.csv")

# Define prompt phrases
# Costs about $10 to run over 200 articles

assistant = ("You are a helpful assistant.")

say_true_or_false = (
    "Based on the provided information, determine if the compound was found in the specified species and output 'true' if it was, or 'false' if it wasn't. "
    "Remember, your task is to identify the presence or absence of a species-compound association. "
    "Make sure to consider all the relevant information provided in the study before making your classification. "
    "Please respond ONLY with 'true' or 'false'. If there isnt enough information to determine the answer, say 'false'."
)

you_are_an_ai_classifier = (
    "You are an AI classifier tasked with determining the existence of a species-compound association. "
    "Your goal is to analyze the provided information and correctly identify if a specific compound was found in a given species. "
    "You will be presented with various scientific studies and their results. "
    "You should read the description of the study carefully, paying attention to the details about the compound and the species mentioned."
)

beware_of_synonyms = (
    "Additionally, different names for plant species and chemicals may be used as well, so be aware of that."
)
use_background_knowledge = (
    "Although the text may not explicitly state the species-compound association, use your background knowledge and best judgment to make an informed decision."
)
youre_going_to_want_to_say_false = (
    "You're going to want to say 'false' which may not be correct. If you know the compound is found in the species, then go ahead and say 'true'."
)
go_ahead_and_say_true = (
    "Go ahead and return 'true' even if there is the slightest indication (a potentially non-obvious) association present."
)

# Define the different system prompt phrases you might want to combine
prompt_phrases = {
    'assistant': assistant,
    'you_are_an_ai_classifier': you_are_an_ai_classifier,
    'say_true_or_false': say_true_or_false,
    'beware_of_synonyms': beware_of_synonyms,
    'use_background_knowledge': use_background_knowledge,
    'youre_going_to_want_to_say_false': youre_going_to_want_to_say_false,
    'go_ahead_and_say_true': go_ahead_and_say_true
}

# Define the combinations you want to test initially
# Each combination is a list of keys from the prompt_phrases dictionary
prompt_combinations = [
    ['assistant', 'say_true_or_false'],
    ['assistant', 'say_true_or_false'],
    ['you_are_an_ai_classifier', 'say_true_or_false'],
    ['you_are_an_ai_classifier', 'say_true_or_false'],
    ['assistant', 'say_true_or_false', 'use_background_knowledge'],
    ['you_are_an_ai_classifier', 'use_background_knowledge', 'say_true_or_false'],
    ['you_are_an_ai_classifier', 'say_true_or_false', 'youre_going_to_want_to_say_false'],
    ['you_are_an_ai_classifier', 'say_true_or_false', 'go_ahead_and_say_true'],
    ['you_are_an_ai_classifier', 'say_true_or_false', 'go_ahead_and_say_true', 'youre_going_to_want_to_say_false']
]

# Define the content columns for each combination
content_columns = [
    'abstract',
    'contents',
    'abstract',
    'contents',
    'contents',
    'contents',
    'contents',
    'contents',
    'contents'
]

# Load your dataframe
dataframe = pd.read_csv("/project_data/shared/ai_in_phytochemistry/vingette_2_RAG/rag_output.csv")

# Process the dataframe
rag_output_processed = process_dataframe(
    dataframe=dataframe, 
    prompt_combinations=prompt_combinations,
    model="gpt-4o",
    content_columns=content_columns,
    store_content_used = True
)

# Save the processed dataframe to CSV
rag_output_processed.to_csv("/project_data/shared/ai_in_phytochemistry/vingette_2_RAG/rag_output_processed_rep3.csv")

# Define prompt phrases
assistant = ("You are a helpful assistant.")

say_true_or_false = (
    "Based on the provided information, determine if the compound was found in the specified species and output 'true' if it was, or 'false' if it wasn't. "
    "Remember, your task is to identify the presence or absence of a species-compound association. "
    "Make sure to consider all the relevant information provided in the study before making your classification. "
    "Please respond ONLY with 'true' or 'false'. If there isn't enough information to determine the answer, say 'false'."
)

you_are_an_ai_classifier = (
    "You are an AI classifier tasked with determining the existence of a species-compound association. "
    "Your goal is to analyze the provided information and correctly identify if a specific compound was found in a given species. "
    "You will be presented with various scientific studies and their results. "
    "You should read the description of the study carefully, paying attention to the details about the compound and the species mentioned."
)

beware_of_synonyms = (
    "Additionally, different names for plant species and chemicals may be used as well, so be aware of that."
)
use_background_knowledge = (
    "Although the text may not explicitly state the species-compound association, use your background knowledge and best judgment to make an informed decision."
)
youre_going_to_want_to_say_false = (
    "You're going to want to say 'false' which may not be correct. If you know the compound is found in the species, then go ahead and say 'true'."
)
go_ahead_and_say_true = (
    "Go ahead and return 'true' even if there is the slightest indication (a potentially non-obvious) association present."
)

# Define the different system prompt phrases you might want to combine
prompt_phrases = {
    'assistant': assistant,
    'you_are_an_ai_classifier': you_are_an_ai_classifier,
    'say_true_or_false': say_true_or_false,
    'beware_of_synonyms': beware_of_synonyms,
    'use_background_knowledge': use_background_knowledge,
    'youre_going_to_want_to_say_false': youre_going_to_want_to_say_false,
    'go_ahead_and_say_true': go_ahead_and_say_true
}

# Define the combinations you want to test initially
prompt_combinations = [
    ['assistant', 'say_true_or_false'],
    ['assistant', 'say_true_or_false'],
    ['you_are_an_ai_classifier', 'say_true_or_false'],
    ['you_are_an_ai_classifier', 'say_true_or_false'],
    ['assistant', 'say_true_or_false', 'use_background_knowledge'],
    ['you_are_an_ai_classifier', 'use_background_knowledge', 'say_true_or_false'],
    ['you_are_an_ai_classifier', 'say_true_or_false', 'youre_going_to_want_to_say_false'],
    ['you_are_an_ai_classifier', 'say_true_or_false', 'go_ahead_and_say_true'],
    ['you_are_an_ai_classifier', 'say_true_or_false', 'go_ahead_and_say_true', 'youre_going_to_want_to_say_false']
]

# Print each full prompt combination
for combination in prompt_combinations:
    full_prompt = " ".join([prompt_phrases[key] for key in combination])
    print("Prompt Combination:", combination)
    print(full_prompt)
    print("\n" + "="*80 + "\n")
