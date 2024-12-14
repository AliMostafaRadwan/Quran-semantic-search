import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

def remove_arabic(text):
    """
    Remove Arabic text and clean the remaining English text.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove text between parentheses that contains Arabic
    text = re.sub(r'\([^)]*[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF][^)]*\)', '', text)
    
    # Remove standalone Arabic text
    text = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', '', text)
    
    # Clean up extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def process_tafsir_files():
    """
    Process all tafsir files and create a cleaned version with only English text.
    """
    tafsir_dir = Path("tafsir/en-tafisr-ibn-kathir")
    processed_data = []
    
    # Find all JSON files recursively
    json_files = list(tafsir_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            print(f"\nProcessing file: {json_file}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Check if the file contains the expected structure
                if isinstance(data, dict) and 'text' in data and 'surah' in data and 'ayah' in data:
                    print(f"Found valid tafsir entry for Surah {data['surah']}, Ayah {data['ayah']}")
                    
                    # Clean the explanation text
                    cleaned_text = remove_arabic(data['text'])
                    
                    if cleaned_text.strip():  # Only add if there's text after cleaning
                        processed_data.append({
                            'surah': data['surah'],
                            'verse_number': data['ayah'],
                            'explanation': cleaned_text
                        })
                        print(f"Successfully processed Surah {data['surah']}, Verse {data['ayah']}")
                        print(f"Sample of cleaned text: {cleaned_text[:100]}...")
                    else:
                        print(f"Warning: No valid text after cleaning for Surah {data['surah']}, Verse {data['ayah']}")
                else:
                    print(f"Warning: Unexpected JSON structure in {json_file}")
                    print(f"Keys found: {list(data.keys())}")
        
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            continue
    
    # Sort by surah and verse number
    processed_data.sort(key=lambda x: (x['surah'], x['verse_number']))
    print(f"\nTotal processed verses: {len(processed_data)}")
    
    return processed_data

def create_embeddings(processed_data, output_dir="embeddings"):
    """
    Create embeddings from the processed data.
    """
    if not processed_data:
        print("No data to process!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get all explanations
    texts = [item['explanation'] for item in processed_data]
    
    # Process in batches
    batch_size = 32
    embeddings_list = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings_list.append(batch_embeddings)
    
    # Combine all embeddings
    embeddings = np.vstack(embeddings_list)
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save the results
    index_path = os.path.join(output_dir, "tafsir_semantic_index.index")
    data_path = os.path.join(output_dir, "tafsir_verses.json")
    
    print(f"Saving index to {index_path}")
    faiss.write_index(index, index_path)
    
    print(f"Saving processed data to {data_path}")
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print("Processing completed successfully!")

def main():
    print("Starting tafsir processing...")
    
    # First, process and clean all files
    processed_data = process_tafsir_files()
    
    if processed_data:
        print("\nSample of processed data:")
        print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))
        
        # Create embeddings from processed data
        create_embeddings(processed_data)
    else:
        print("No data was processed. Please check the input files.")

if __name__ == "__main__":
    main() 