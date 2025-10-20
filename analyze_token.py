import pandas as pd
from transformers import AutoTokenizer
import os

# --- Configuration ---
INPUT_FILE = "all_prompts.csv"
OUTPUT_FILE = "tokenized_summary.csv"
# Define the model whose tokenizer should be used.
# Change this to match the specific model you are preparing data for (e.g., 'gpt2', 'roberta-base', 'llama-v2').
MODEL_NAME = "openai/gpt-oss-20b" 

# --- Setup ---
print(f"Loading tokenizer for model: {MODEL_NAME}...")
try:
    # Attempt to load the tokenizer. Exit if model name is invalid or network error occurs.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Please check the model name or your internet connection.")
    exit()

def count_tokens(text):
    """
    Tokenizes the input text and returns the number of tokens.
    It handles NaN/None/non-string inputs gracefully.
    """
    if pd.isna(text) or not isinstance(text, str):
        return 0
    
    # Setting add_special_tokens=False to count the content tokens only.
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

# --- Execution ---

try:
    # 1. Load the CSV file. This will raise FileNotFoundError if INPUT_FILE is missing.
    print(f"\nReading data from '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)

    # Ensure required columns are present
    if "task_id" not in df.columns or "prompt" not in df.columns:
        raise ValueError(f"Input file must contain 'task_id' and 'prompt' columns. Found: {df.columns.tolist()}")

    # 2. Calculate token count for each prompt
    print("Calculating token counts...")
    df['token_count'] = df['prompt'].apply(count_tokens)

    # 3. Create the new DataFrame with only task_id and token_count
    result_df = df[['task_id', 'token_count']].copy()

    # 4. Sort the results from minimum token count to maximum
    print("Sorting results by token count (min to max)...")
    sorted_df = result_df.sort_values(by='token_count', ascending=True).reset_index(drop=True)

    # 5. Save the sorted data to a new CSV file
    sorted_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Token summary saved to '{OUTPUT_FILE}'.")
    print("\nFirst 5 rows of the output:")
    print(sorted_df.head())

except FileNotFoundError:
    print(f"\n--- ERROR ---")
    print(f"Input file '{INPUT_FILE}' was not found.")
    print("Please ensure the CSV file is in the same directory and named exactly 'input.csv'.")
    print("-------------")
except ValueError as e:
    print(f"\nError: Data validation failed. {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred during processing: {e}")
