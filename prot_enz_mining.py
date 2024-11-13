import pandas as pd
from Bio import Entrez
import re
import openai

# Provide your email address for NCBI Entrez usage
Entrez.email = "your.email@example.com"

# Directly set the OpenAI API key
openai.api_key = ''  # Replace with your actual API key

# Function to fetch protein or nucleotide entries by protein ID or accession number from NCBI
def fetch_protein_entries(protein_ids):
    records = []
    for protein_id in protein_ids:
        try:
            # Try fetching from the protein database first
            handle = Entrez.efetch(db="protein", id=protein_id, rettype="gb", retmode="text")
            record = handle.read()
            handle.close()
            records.append(record)
        except Exception as e:
            print(f"Error fetching protein {protein_id} from protein database: {e}")
            try:
                # If fetching from protein fails, try the nucleotide database
                handle = Entrez.efetch(db="nucleotide", id=protein_id, rettype="gb", retmode="text")
                record = handle.read()
                handle.close()
                records.append(record)
            except Exception as e:
                print(f"Error fetching protein {protein_id} from nucleotide database: {e}")
                records.append(None)  # Append None if both attempts fail
    return records

# Function to extract PubMed IDs from a protein record
def extract_pubmed_ids(record):
    # Check if the record is not None or empty
    if record:
        pubmed_ids = re.findall(r'(?:PubMed\sID\s*[:=]\s*|PMID\s*[:=]\s*|PubMed\s*[:=]\s*|PUBMED\s*[:=]\s*|PUBMED\s+|PubMed\s+)(\d+)', record)
        if pubmed_ids:
            return pubmed_ids  # Return found PubMed IDs
    return ["No PubMed ID Found"]  # Return this message if no IDs are found

# Function to fetch abstracts from PubMed using PubMed IDs
def fetch_pubmed_abstracts(pmid_list):
    abstracts = []
    for pmid in pmid_list:
        if pmid:
            try:
                handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
                abstract = handle.read()
                handle.close()
                abstracts.append(abstract if abstract else "Abstract is missing")
            except Exception as e:
                abstracts.append(f"Error fetching abstract for PMID {pmid}: {e}")
    return "\n\n".join(abstracts) if abstracts else "No matching article found"
# gpt-3.5-turbo
# gpt-4o-mini
# gpt-4-0125-preview
# gpt-4o-2024-08-06
# GPT query function
def query_gpt(df, max_rows=144):
    responses = []
    for i, row in df.head(max_rows).iterrows():
        product = row.get('Product', 'Unknown')
        specific_record = row.get('Specific_Record', 'No Record')
        query = f"Does the enzyme described in the following record make {product}?\n\n{specific_record}"
        
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": """
                        You are an expert at analyzing scientific literature.
                        I am going to give you a specific record from NCBI and an abstract that contains information about an enzyme and its product.
                        Your job is to look at the specific record and abstract that I provide and determine if the enzyme mentioned makes, produces, synthesizes, or catalyzes the product I ask about.
                        Think through the problem step by step before answering the question.
                        Here are some parameters on how you should answer the question:
                        The first word of your response MUST be yes or no.
                        After the first word of your response, you should include a brief explanation as to why you chose yes or no.
                        If you are not entirely sure that the enzyme I mention makes the product I ask about, your answer should be no, and you should indicate that manual verification is needed.
                    """},
                    {"role": "user", "content": query}
                ]
            )
            responses.append(response['choices'][0]['message']['content'])
        except Exception as e:
            print(f"Error querying GPT: {e}")
            responses.append("Error")

    df['GPT_Response'] = responses
    return df

# Load the new Excel file (Protein_IDs_All) with all protein IDs
df = pd.read_excel("/project_data/shared/ai_in_phytochemistry/Protein_IDs_All (3).xlsx")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces

# Fetch specific records from NCBI
specific_records = fetch_protein_entries(df['Protein_ID'].tolist())

# Creating DataFrame with relevant data, properly inserting fetched records
df_proteins = pd.DataFrame({
    'Protein_ID': df['Protein_ID'],
    'Specific_Record': specific_records,  # Now properly using the fetched specific records
    'Product': df['Product']
})

# Extract PubMed IDs and fetch abstracts
df_proteins['PubMed_IDs'] = df_proteins['Specific_Record'].apply(extract_pubmed_ids)
df_proteins['Abstracts'] = df_proteins['PubMed_IDs'].apply(fetch_pubmed_abstracts)

# Prepare GPT queries
gpt_queries = df_proteins[['Protein_ID', 'Specific_Record', 'Product']]
gpt_responses = query_gpt(gpt_queries)

# Merge results into final DataFrame
df_final = df_proteins.merge(gpt_responses, on=['Protein_ID', 'Specific_Record', 'Product'], how='left')

# Set the GPT response correctly to avoid SettingWithCopyWarning
df_final.loc[:, 'GPT_Response'] = gpt_responses  # This is the line where the warning is addressed

# Save to Excel
df_final.to_excel("GPT3.5Turbo_CoT1_3.xlsx", index=False)

