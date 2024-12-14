import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def analyze_tafsir_lengths():
    """
    Analyze and compare the lengths of different tafsirs from JSON files.
    Returns a dictionary with statistics for each tafsir.
    """
    tafsir_dir = Path("tafsir")
    tafsir_stats = {}
    
    # Get all tafsir directories
    tafsir_paths = [d for d in tafsir_dir.iterdir() if d.is_dir()]
    
    for tafsir_path in tafsir_paths:
        tafsir_name = tafsir_path.name
        total_length = 0
        verse_count = 0
        surah_lengths = {}
        
        # Process only JSON files in the root of each tafsir directory
        json_files = [f for f in tafsir_path.glob("*.json")]
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Calculate length of text content
                    if isinstance(data, dict):
                        text_content = json.dumps(data, ensure_ascii=False)
                        length = len(text_content)
                        surah_num = json_file.stem  # Get surah number from filename
                        surah_lengths[surah_num] = length
                        total_length += length
                        verse_count += len(data)
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                continue
        
        if surah_lengths:
            tafsir_stats[tafsir_name] = {
                'total_length': total_length,
                'average_length': total_length / len(surah_lengths) if len(surah_lengths) > 0 else 0,
                'surah_lengths': surah_lengths,
                'verse_count': verse_count
            }
    
    return tafsir_stats

def visualize_results(stats):
    """
    Create visualizations for the tafsir analysis results.
    """
    # Prepare data for plotting
    tafsirs = list(stats.keys())
    total_lengths = [stats[t]['total_length'] for t in tafsirs]
    avg_lengths = [stats[t]['average_length'] for t in tafsirs]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot total lengths
    bars1 = ax1.bar(tafsirs, total_lengths)
    ax1.set_title('Total Length of Each Tafsir')
    ax1.set_xlabel('Tafsir')
    ax1.set_ylabel('Total Length (characters)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot average lengths
    bars2 = ax2.bar(tafsirs, avg_lengths)
    ax2.set_title('Average Length per Surah')
    ax2.set_xlabel('Tafsir')
    ax2.set_ylabel('Average Length (characters)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def main():
    # Analyze tafsir lengths
    stats = analyze_tafsir_lengths()
    
    # Print detailed statistics
    print("\nTafsir Analysis Results:")
    print("=" * 50)
    
    # Convert to pandas DataFrame for better display
    results_df = pd.DataFrame({
        'Total Length': [stats[t]['total_length'] for t in stats],
        'Average Length': [stats[t]['average_length'] for t in stats],
        'Verse Count': [stats[t]['verse_count'] for t in stats]
    }, index=stats.keys())
    
    # Sort by total length
    results_df = results_df.sort_values('Total Length', ascending=False)
    
    # Print results
    print("\nOverall Statistics:")
    print(results_df)
    
    # Print detailed comparison
    print("\nDetailed Comparison:")
    for tafsir_name, data in sorted(stats.items(), 
                                  key=lambda x: x[1]['total_length'], 
                                  reverse=True):
        print(f"\n{tafsir_name}:")
        print(f"Total Length: {data['total_length']:,} characters")
        print(f"Average Length per Surah: {data['average_length']:,.2f} characters")
        print(f"Number of verses covered: {data['verse_count']}")
    
    # Create visualizations
    fig = visualize_results(stats)
    
    # Save the plot
    fig.savefig('tafsir_analysis.png')
    print("\nVisualization saved as 'tafsir_analysis.png'")

if __name__ == "__main__":
    main() 