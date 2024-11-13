####################################################
##################### MODULES ######################
####################################################

import os
import base64
import requests
import pandas as pd
import warnings
import openai
import json
client = OpenAI(api_key = '')

######################################################
##################### FUNCTIONS ######################
######################################################

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def check_llm(table_data, api_key):
    check_prompt = '''
You are a highly accurate and detail-oriented language model.

Your task is to analyze the following table data and determine if it contains both:

1. **Plant species names**: These could be in either the rows or the columns.
2. **Chemical compound names**: These could also be in either the rows or the columns.

Respond with "yes" if the table contains both plant species names and chemical compound names, regardless of their orientation (rows or columns). Respond with "no" if either is missing. Do not provide any additional information or explanations.
    '''

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {"role": "system", "content": check_prompt},
            {"role": "user", "content": f"Table data: {table_data}"}
        ],
        "max_tokens": 10
    }

    check_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    check_response_json = check_response.json()
    return check_response_json['choices'][0]['message']['content'].strip().lower()

def determine_orientation(table_data, api_key):
    orientation_prompt = '''
You are a highly accurate and detail-oriented language model.

Your task is to analyze the structure of the provided table data and label it according to the following categories:

1. **rows_are_compounds**: If the table lists chemical compounds (e.g., phenolic acids) in the rows and corresponding values (e.g., concentrations or quantities) in the columns.
2. **rows_are_species**: If the table lists plant species (e.g., wheat, corn) in the rows and corresponding values (e.g., concentrations or quantities of various compounds) in the columns.

Provide only the label ("rows_are_compounds" or "rows_are_species") based on the table structure, without any additional comments or explanations.
    '''

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {"role": "system", "content": orientation_prompt},
            {"role": "user", "content": f"Table data: {table_data}"}
        ],
        "max_tokens": 10
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()

    # Extract the LLM response content which is the label
    orientation_label = response_json['choices'][0]['message']['content'].strip()
    
    return orientation_label

def extraction_function(file_list, folder_path):
    
    extraction_prompt = '''
You are an expert in parsing tables of scientific data.

Your task is to transcribe the table from the provided PNG image into a structured JSON format. Follow these instructions:

1. **Extract the table data**: Accurately extract all the data from the table, including headers and row labels.
2. **Represent the data in JSON**: Output the table data in JSON format as an array of rows, where each row is an array of cell values.
3. **Maintain the structure**: Ensure that the headers and data align correctly, preserving the original table structure.
5. **Handle Empty Cells**: Represent empty cells with null values.
6. **Remove Extraneous Information**: Omit captions, footnotes, or other non-data text.
7. **Expand Abbreviations**: Expand any chemical name abbreviations to their full names. Here are some examples:
   - p-HB = p-hydroxybenzoic acid
   - VAN = vanillic acid
   - CAF = caffeic acid
   - SYR = syringic acid
   - p-COU = p-coumaric acid
   - FER = ferulic acid
   - o-COU = o-coumaric acid
   Please note that this list is not exhaustive. Expand other similar abbreviations you encounter to their full names using the context provided by the table or surrounding text.
8. **Complete Output**: Output the entire table without truncating. Ensure the full table is included.

Provide the table data in the specified JSON format without any additional comments or explanations.
    '''
    
    for filename in file_list: # filename = file_list[0]
        image_path = os.path.join(folder_path, filename)
        
        base64_image = encode_image(image_path)
    
        is_valid_table = False
        retry_count = 0
        max_retries = 3
    
        while not is_valid_table and retry_count < max_retries:
            retry_count += 1

            ### EXTRACTION
            print(f"Extracting {filename}, attempt {retry_count}...")
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": extraction_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Here is the table image:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                        ]}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "table_response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "table_data": {
                                        "type": "array",
                                        "items": {
                                            "type": "array",
                                            "items": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "null"}
                                                ]
                                            }
                                        }
                                    }
                                },
                                "required": ["table_data"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }
                    }
                )
            except Exception as e:
                warnings.warn(f"API request failed for {filename}: {e}. Retrying...")
                continue  # Retry the loop
    
            # Extract the parsed response
            try:
                parsed_data = json.loads(response.choices[0].message.content)
                table_data = parsed_data['table_data']
            except Exception as e:
                warnings.warn(f"Error accessing parsed data for {filename}: {e}. Retrying...")
                continue  # Retry the loop
    
            # Convert the table data to a pandas DataFrame
            try:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
    
                # Perform the check LLM twice for validation
                table_json_str = df.to_json(orient='split')
                check_result_1 = check_llm(table_json_str, api_key)
                check_result_2 = check_llm(table_json_str, api_key)
    
                if check_result_1 == "yes" and check_result_2 == "yes":
                    # If both checks pass, proceed to save the output
                    is_valid_table = True
                    print(f"Valid table with both plant species and compound names detected for {filename}. Saving output.")
    
                    # Write the DataFrame to a CSV file
                    output_file_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_extracted.csv")
                    df.to_csv(output_file_path, index=False)
                else:
                    # If either check fails, retry
                    warnings.warn(f"File {filename} failed the check LLM validation. Retrying...")
    
            except Exception as e:
                warnings.warn(f"Error processing table data for {filename}: {e}. Retrying...")


        if is_valid_table:
            orientation_label = determine_orientation(table_json_str, api_key)
    
            if orientation_label == "rows_are_species":
                df = df.transpose().reset_index()
                df.columns = df.iloc[0]
                df = df.drop(0)
    
            # Save the final DataFrame
            df.to_csv(output_file_path, index=False)
            print(f"Extracted and saved output for {filename}\n\n")

def processing_function(file_list, folder_path):
    
    processing_prompt = '''
You are an expert in parsing tables of scientific data.

Your task is to process a previously transcribed table according to the following instructions:

1. **Output Format**: Output the table data in JSON format as an array of rows, where each row is an array of cell values.
2. **Headers and Row Names**: Include all headers and row names as they appear in the original table, omitting special characters and units (e.g. "%", "mg/g", etc.).
3. **Handle Empty Cells**: Represent empty cells with null values.
4. **Means Only**: If a table entry includes a mean with precision (e.g., X ± Y), include only the mean (X) and omit any other symbols in the entries like question marks, letters, etc.
5. **Consistent Formatting**: Ensure consistent formatting across all rows and columns, and avoid any unnecessary symbols or formatting that doesn't align with a simple table.
6. **Complete Output**: Output the entire table without truncating. Ensure the full table is included.

Provide the table in the specified format without any additional comments or explanations.
'''

    # 5. **Ranges**: If a cell in a table reports a range (e.g. 24-28), replace the range with the central value of the range.
    for filename in file_list: # filename = file_list[0]
        table_content = pd.read_csv(os.path.join(folder_path, filename)).to_json()
        is_valid_table = False
        retry_count = 0
        max_retries = 10
    
        while not is_valid_table and retry_count < max_retries:
            retry_count += 1

            ### PROCESSING
            print(f"Processing {filename}, attempt {retry_count}...")
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=[
                        {"role": "system", "content": processing_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": table_content},
                        ]}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "table_response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "table_data": {
                                        "type": "array",
                                        "items": {
                                            "type": "array",
                                            "items": {
                                                "anyOf": [
                                                    {"type": "string"},
                                                    {"type": "null"}
                                                ]
                                            }
                                        }
                                    }
                                },
                                "required": ["table_data"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }
                    }
                )
            except Exception as e:
                warnings.warn(f"API request failed for {filename}: {e}. Retrying...")
                continue  # Retry the loop
    
            # Extract the parsed response
            try:
                parsed_data = json.loads(response.choices[0].message.content)
                table_data = parsed_data['table_data']
            except Exception as e:
                warnings.warn(f"Error accessing parsed data for {filename}: {e}. Retrying...")
                continue  # Retry the loop
    
            # Convert the table data to a pandas DataFrame
            try:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
    
                # Perform the check LLM twice for validation
                table_json_str = df.to_json(orient='split')
                check_result_1 = check_llm(table_json_str, api_key)
                check_result_2 = check_llm(table_json_str, api_key)
    
                if check_result_1 == "yes" and check_result_2 == "yes":
                    # If both checks pass, proceed to save the output
                    is_valid_table = True
                    print(f"Valid table with both plant species and compound names detected for {filename}. Saving output.")
    
                    # Write the DataFrame to a CSV file
                    output_file_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_processed.csv")
                    df.to_csv(output_file_path, index=False)
                else:
                    # If either check fails, retry
                    warnings.warn(f"File {filename} failed the check LLM validation. Retrying...")
    
            except Exception as e:
                warnings.warn(f"Error processing table data for {filename}: {e}. Retrying...")


        if is_valid_table:
            orientation_label = determine_orientation(table_json_str, api_key)
    
            if orientation_label == "rows_are_species":
                df = df.transpose().reset_index()
                df.columns = df.iloc[0]
                df = df.drop(0)
    
            # Save the final DataFrame
            df.to_csv(output_file_path, index=False)
            print(f"Processed and saved output for {filename}\n\n")

# Define the import_check function
def import_check(folder_path):
    # Dictionary to store DataFrames with the filename as the key
    dataframes = {}

    # Loop over all _processed.csv files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith("_processed.csv"):
            csv_file_path = os.path.join(folder_path, filename)
            
            # Read the processed .csv file into a DataFrame
            df = pd.read_csv(csv_file_path)
            
            # Rename the first column header to "compound_name"
            df.rename(columns={df.columns[0]: "compound_name"}, inplace=True)
            
            # Convert all entries in the "compound_name" column to lowercase and replace spaces with underscores
            df['compound_name'] = df['compound_name'].str.lower().str.replace(' ', '_')
            
            # Store the DataFrame in the dictionary with the filename (without extension) as the key
            dataframes[os.path.splitext(filename)[0]] = df

    # List of filenames (keys)
    filenames = list(dataframes.keys())

    # Find DataFrames with no common compound names
    isolated_files = []

    for i, file1 in enumerate(filenames):
        compounds1 = set(dataframes[file1]['compound_name'].tolist())
        has_common = False
        
        for j, file2 in enumerate(filenames):
            if i != j:
                compounds2 = set(dataframes[file2]['compound_name'].tolist())
                if not compounds1.isdisjoint(compounds2):
                    has_common = True
                    break
        
        if not has_common:
            isolated_files.append(file1)

    # Return the isolated files
    return isolated_files

def main_processing(folder_path, api_key):

    # Get list of all PNG files that don't have a corresponding *_processed.csv file, run import_function
    file_list = [
        file for file in os.listdir(folder_path) 
        if file.endswith(".png") and not os.path.exists(os.path.join(folder_path, file.replace(".png", "_extracted.csv")))
    ]    
    extraction_function(file_list, folder_path)

    # Get list of all *_processed.png files, run processing_function
    file_list = [
        file for file in os.listdir(folder_path) 
        if file.endswith("_extracted.csv") and not os.path.exists(os.path.join(folder_path, file.replace("_extracted.csv", "_extracted_processed.csv")))
    ]
    processing_function(file_list, folder_path)
    
    # Initialize attempt counter
    attempts = 0
    max_attempts = 3

    # Run import_check to find isolated files
    isolated_files = import_check(folder_path)
    
    # Try up to 3 times to reprocess isolated files
    while isolated_files and attempts < max_attempts:
        attempts += 1
        print(f"Reprocessing isolated files (Attempt {attempts})...\n")
        files_to_reprocess = [f"{file.replace('_processed', '')}.png" for file in isolated_files]
        extraction_function(files_to_reprocess, folder_path)
        processing_function(files_to_reprocess, folder_path)
        # Run the check again after reprocessing
        isolated_files = import_check(folder_path)
        
    if isolated_files:
        print("These files still have no common compound_name entries with any other DataFrame after reprocessing:\n")
        for file in isolated_files:
            print(file)
    else:
        print("All DataFrames share at least one compound_name entry with another DataFrame after reprocessing.")

################################################
##################### RUN ######################
################################################

main_processing(
    folder_path = "/project_data/shared/ai_in_phytochemistry/vingette_3_visionAI/articles/phytosterols/png_annot/",
    api_key = ''
)