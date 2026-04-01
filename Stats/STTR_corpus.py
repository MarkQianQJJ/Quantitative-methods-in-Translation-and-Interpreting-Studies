import os
import pandas as pd
import string

# Function to read the text from file and clean it
def read_and_clean_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Lowercasing for uniformity
        # Remove punctuation and split into words
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
    return words

# Function to calculate STTR for a given text
def calculate_sttr(words, chunk_size=500):
    sttr_values = []
    
    # Process the text in chunks of `chunk_size` words
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i+chunk_size]
        if len(chunk) < chunk_size:
            # If chunk is smaller than the chunk_size, add 0 as the STTR value
            sttr_values.append(0)
        else:
            # Calculate Type/Token Ratio (TTR)
            types = len(set(chunk))  # Number of unique words (types)
            tokens = len(chunk)      # Total words (tokens)
            ttr = types / tokens if tokens > 0 else 0
            sttr_values.append(ttr)
    
    # Return the average STTR
    return sum(sttr_values) / len(sttr_values) if sttr_values else 0

# Main function to calculate STTR for both EO and HT files and output to CSV
def calculate_sttr_for_files(eo_dir, ht_dir, output_file='sttr_results.csv'):
    results = []
    
    # Process English Original (EO) files
    for filename in os.listdir(eo_dir):
        if filename.endswith('.txt') and filename.startswith('EOR'):
            file_path = os.path.join(eo_dir, filename)
            words = read_and_clean_text(file_path)
            sttr = calculate_sttr(words)
            results.append([filename, 'EO', sttr])
    
    # Process Human Translation (HT) files
    for filename in os.listdir(ht_dir):
        if filename.endswith('.txt') and filename.startswith('HTREN'):
            file_path = os.path.join(ht_dir, filename)
            words = read_and_clean_text(file_path)
            sttr = calculate_sttr(words)
            results.append([filename, 'HT', sttr])
    
    # Create a DataFrame and save as CSV
    df = pd.DataFrame(results, columns=['TextID', 'TextType', 'STTR'])
    df.to_csv(output_file, index=False)

# Define the directories for EO and HT
eo_directory = r'D:\vscode\corpus data\STTR\EU\Regulation_EU'
ht_directory = r'D:\vscode\corpus data\STTR\PKU\Regulation_PKU\Regulation_EN'

# Call the main function to process the files and generate the CSV
calculate_sttr_for_files(eo_directory, ht_directory)

print("STTR calculation complete. Results saved to 'sttr_results.csv'.")