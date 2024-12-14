# Quran Tafsir Search Engine

A semantic search engine for Quran verses and their interpretations (Tafsir Ibn Kathir), with a user-friendly Streamlit interface.
i used Tafsir Ibn Kathir since its the largest of them all
![Tafsir Analysis](tafsir_analysis.png)


# screenshots
![image](https://github.com/user-attachments/assets/a042ad5f-a3cd-40fe-806c-97e00b408608)
![image](https://github.com/user-attachments/assets/9876e455-1b72-4237-abbb-2119dd3a2795)

## Features

- 🔍 Semantic search through Ibn Kathir's Tafsir
- 🎯 Accurate verse matching using FAISS similarity search
- 🔊 Audio recitation for each verse
- 📱 User-friendly Streamlit interface
- 🌐 Support for both Arabic text and English translations

## Installation

1. Clone the repository: 
2. Clone https://github.com/spa5k/tafsir_api.git to get the tafsir folder
3. Clone https://github.com/semarketir/quranjson to get the verses and the translation 
4. run ```pip install -r requirements.txt```
5. run the ```generate_embeddings.py```
6. run ```streamlit run main.py```
