import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import requests
from io import BytesIO
import base64
from pathlib import Path

def get_verse(surah_num: int, verse_num: int) -> dict:
    """
    Retrieve verse data from Quran JSON files.
    
    Args:
        surah_num (int): The surah number (1-114)
        verse_num (int): The verse number
        
    Returns:
        dict: A dictionary containing the verse text and translation
    """
    try:
        # Load the Arabic text
        with open(f'quranjson/source/surah/surah_{surah_num}.json', 'r', encoding='utf-8') as f:
            surah_data = json.load(f)
            
        # Load the English translation
        with open(f'quranjson/source/translation/en/en_translation_{surah_num}.json', 'r', encoding='utf-8') as f:
            translation_data = json.load(f)
            
        # Get the specific verse
        verse_key = f"verse_{verse_num}"
        verse_text = surah_data['verse'][verse_key]
        verse_translation = translation_data['verse'][verse_key]
        
        return {
            'surah': surah_num,
            'verse_number': verse_num,
            'arabic_text': verse_text,
            'translation': verse_translation
        }
        
    except FileNotFoundError:
        return {'error': 'Surah file not found'}
    except KeyError:
        return {'error': 'Verse not found'}
    except json.JSONDecodeError:
        return {'error': 'Invalid JSON file'}


def load_verses() -> list:
    """
    Load all verses and their translations into a list for embedding.
    
    Returns:
        list: A list of dictionaries containing surah, verse_number, and translation.
    """
    all_verses = []
    for surah_num in range(1, 115):  # 114 Surahs
        try:
            # Load Surah and Translation files
            with open(f'quranjson/source/surah/surah_{surah_num}.json', 'r', encoding='utf-8') as f:
                surah_data = json.load(f)
            with open(f'quranjson/source/translation/en/en_translation_{surah_num}.json', 'r', encoding='utf-8') as f:
                translation_data = json.load(f)

            # Combine the verses and translations
            for verse_key, translation in translation_data['verse'].items():
                verse_num = int(verse_key.split("_")[1])
                all_verses.append({
                    'surah': surah_num,
                    'verse_number': verse_num,
                    'translation': translation
                })
        except FileNotFoundError:
            continue
    return all_verses


def create_index(verses: list):
    """
    Create a FAISS index from the Quranic translations.
    
    Args:
        verses (list): List of verses with translations.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    translations = [verse['translation'] for verse in verses]
    embeddings = model.encode(translations, convert_to_numpy=True)

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index and verses data
    faiss.write_index(index, "quran_semantic_index.index")
    with open("quran_verses.json", "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=4)


def semantic_search(query: str, top_k=3) -> list:
    """
    Perform semantic search on the tafsir explanations.
    """
    # Load precomputed index and explanations from embeddings directory
    index_path = "embeddings/tafsir_semantic_index.index"
    explanations_path = "embeddings/tafsir_verses.json"
    
    if not os.path.exists(index_path) or not os.path.exists(explanations_path):
        raise FileNotFoundError("Embeddings files not found. Please run generate_embeddings.py first.")
    
    index = faiss.read_index(index_path)
    with open(explanations_path, "r", encoding="utf-8") as f:
        explanations = json.load(f)

    # Encode the query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Perform search
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve results
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        verse = explanations[idx]
        verse['similarity_score'] = 1 - dist
        results.append(verse)
    return results


def get_audio_url(surah_num: int, verse_num: int) -> str:
    """
    Generate the audio URL for a specific verse from Everyayah.com
    """
    # Using Mishary Rashid Alafasy recitation
    base_url = "https://everyayah.com/data/Alafasy_128kbps"
    # Format surah and verse numbers with leading zeros
    surah_str = str(surah_num).zfill(3)
    verse_str = str(verse_num).zfill(3)
    return f"{base_url}/{surah_str}{verse_str}.mp3"

def main():
    st.title("Quran Explorer with Tafsir Ibn Kathir")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Verse Lookup", "Tafsir Search"])
    
    with tab1:
        st.header("Verse Lookup")
        col1, col2 = st.columns(2)
        
        with col1:
            surah_num = st.number_input("Surah Number", min_value=1, max_value=114, value=1)
        with col2:
            verse_num = st.number_input("Verse Number", min_value=1, value=1)
        
        if st.button("Get Verse"):
            verse_data = get_verse(surah_num, verse_num)
            
            if 'error' not in verse_data:
                st.subheader(f"Surah {verse_data['surah']}, Verse {verse_data['verse_number']}")
                
                # Display Arabic text in larger font
                st.markdown(f"<h2 style='text-align: right; direction: rtl;'>{verse_data['arabic_text']}</h2>", 
                          unsafe_allow_html=True)
                
                st.markdown("**Translation:**")
                st.write(verse_data['translation'])
                
                # Audio player
                audio_url = get_audio_url(surah_num, verse_num)
                st.audio(audio_url)
            else:
                st.error(verse_data['error'])
    
    with tab2:
        st.header("Search in Tafsir Ibn Kathir")
        
        # Check if embeddings exist
        if not os.path.exists("embeddings/tafsir_semantic_index.index"):
            st.error("""
                Embeddings not found. Please run generate_embeddings.py first to create the search index.
                Command: python generate_embeddings.py
            """)
            return
        
        query = st.text_input("Enter your question or topic:")
        num_results = st.slider("Number of results", min_value=1, max_value=10, value=3)
        
        if st.button("Search"):
            if query:
                try:
                    with st.spinner("Searching..."):
                        results = semantic_search(query, top_k=num_results)
                    
                    for i, result in enumerate(results, 1):
                        st.markdown(f"### Result {i}")
                        st.markdown(f"**Surah {result['surah']}, Verse {result['verse_number']}**")
                        
                        # Get full verse data including Arabic text
                        full_verse = get_verse(result['surah'], result['verse_number'])
                        
                        if 'error' not in full_verse:
                            # Display Arabic text
                            st.markdown(f"<h3 style='text-align: right; direction: rtl;'>{full_verse['arabic_text']}</h3>", 
                                      unsafe_allow_html=True)
                            
                            # Display translation
                            st.markdown("**Translation:**")
                            st.write(full_verse['translation'])
                            
                            # Display tafsir explanation
                            st.markdown("**Tafsir Ibn Kathir:**")
                            st.write(result['explanation'])
                            
                            st.markdown(f"Similarity Score: {result['similarity_score']:.2f}")
                            
                            # Audio player
                            audio_url = get_audio_url(result['surah'], result['verse_number'])
                            st.audio(audio_url)
                            
                            st.markdown("---")
                        else:
                            st.error(f"Error loading verse: {full_verse['error']}")
                except Exception as e:
                    st.error(f"Error performing search: {str(e)}")
            else:
                st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()
