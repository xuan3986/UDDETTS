import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from collections import defaultdict
import argparse

def plot_histogram_with_bins(data, bins, title):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=29, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black', 
             label='Histogram')
    
    for boundary in bins:
        plt.axvline(x=boundary, color='red', linestyle='--', 
                    linewidth=1.5, alpha=0.8)
    
    kde = gaussian_kde(data)
    x = np.linspace(min(data)-0.1, max(data)+0.1, 1000)
    plt.plot(x, kde(x), color='darkorange', linewidth=2, 
             label='KDE')
    
    stats_text = f"""
    Mean: {np.mean(data):.2f}
    Std: {np.std(data):.2f}
    Min: {np.min(data):.2f}
    Max: {np.max(data):.2f}
    Bins: {len(bins)-1}
    """
    plt.text(0.98, 0.7, stats_text, transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=20,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel(f"{title} with Bins", fontsize=28)
    plt.ylabel('Density', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{args.pic_path}/Total_{title}.png')
    plt.cla()


def get_cluster_bins(data_col, n_bins):
    """Generate non-uniform box boundaries using K-means"""
    X = data_col.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
    kmeans.fit(X)
    
    centers = np.sort(kmeans.cluster_centers_.flatten())
    boundaries = []
    boundaries.append(np.min(data_col))
    for i in range(len(centers)-1):
        boundaries.append((centers[i] + centers[i+1]) / 2)
    boundaries.append(np.max(data_col))
    
    return np.array(boundaries)


def main():
    data_path = args.data_path
    arousal_dict, dominance_dict, valence_dict, emotion_dict = {}, {}, {}, {}

    for audio_base_folder in ["train"]:
        utt2ADV_file = os.path.join(data_path, audio_base_folder, "utt2ADV")
        utt2emo_flie = os.path.join(data_path, audio_base_folder, "utt2emo")
        with open(utt2ADV_file, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            for line in lines:
                # 'utt [arousal, dominance, valence] for each lines'
                parts = line.strip().split(' ', 1)
                utt = parts[0]
                ADV_values = eval(parts[1])  # [arousal, dominance, valence]
                arousal = float(ADV_values[0])  # arousal
                dominance = float(ADV_values[1])  # dominance
                valence = float(ADV_values[2])  # valence
                if arousal == 0 and dominance == 0 and valence == 0:
                    continue
                if arousal > 7.0 or dominance > 7.0 or valence > 7.0:
                    print(f"{utt} ADV value is more than 7.0")
                    continue
                if arousal <1.0 or dominance <1.0 or valence <1.0:
                    print(f"{utt} ADV value is less than 1.0")
                    continue
                arousal_dict[utt] = arousal
                dominance_dict[utt] = dominance
                valence_dict[utt] = valence
        with open(utt2emo_flie, 'r', encoding='utf-8') as f2:
            for line in f2:
                parts = line.strip().split(' ')
                utt = parts[0]
                emotion_label = parts[1]
                emotion_dict[utt] = emotion_label

    assert arousal_dict.keys() == dominance_dict.keys() == valence_dict.keys()
    utts = list(arousal_dict.keys())
    data = np.array([[arousal_dict[utt], dominance_dict[utt], valence_dict[utt]] for utt in utts])



    # Calculate the ADV center values for each emotion
    emotion_groups = defaultdict(list)
    for utt in utts:
        emo = emotion_dict[utt]
        point = [arousal_dict[utt], dominance_dict[utt], valence_dict[utt]]
        emotion_groups[emo].append(point)

    emotion_centers = {}
    for emo, points in emotion_groups.items():
        points_array = np.array(points)
        # Calculate the mean center in three-dimensional space
        center = np.mean(points_array, axis=0)
        emotion_centers[emo] = {
            'center': center,
            'count': len(points)
        }
    for emo, info in emotion_centers.items():
        print(f"Emotion {emo} (Samples Count:{info['count']}):")
        print(f"  ADV: {info['center']}")
        
    # Calculate global extremum
    dimension_ranges = {
        'arousal': (np.min(data[:, 0]), np.max(data[:, 0])),
        'dominance': (np.min(data[:, 1]), np.max(data[:, 1])),
        'valence': (np.min(data[:, 2]), np.max(data[:, 2]))
    }
    print(f"Arousal: {dimension_ranges['arousal']}")
    print(f"Dominance: {dimension_ranges['dominance']}")
    print(f"Valence: {dimension_ranges['valence']}")

    # Generate uniform bins
    # arousal_bin_boundaries = np.histogram_bin_edges(data[:, 0], bins=14)
    # dominance_bin_boundaries = np.histogram_bin_edges(data[:, 1], bins=14)
    # valence_bin_boundaries = np.histogram_bin_edges(data[:, 2], bins=14)

    # Generate non-uniform bins, n_bins=14 is calculated by the central limit theorem
    arousal_bin_boundaries = get_cluster_bins(data[:, 0], n_bins=14)
    dominance_bin_boundaries = get_cluster_bins(data[:, 1], n_bins=14)
    valence_bin_boundaries = get_cluster_bins(data[:, 2], n_bins=14)

    print("Arousal bin boundaries:", arousal_bin_boundaries)
    print("Dominance bin boundaries:", dominance_bin_boundaries)
    print("Valence bin boundaries:", valence_bin_boundaries)

    plot_histogram_with_bins(data[:, 0], arousal_bin_boundaries, 'Arousal Distribution')
    plot_histogram_with_bins(data[:, 1], dominance_bin_boundaries, 'Dominance Distribution')
    plot_histogram_with_bins(data[:, 2], valence_bin_boundaries, 'Valence Distribution')

    array_dict = {'arousal_bin':arousal_bin_boundaries, 
                'dominance_bin':dominance_bin_boundaries, 
                'valence_bin':valence_bin_boundaries}
    np.savez('bins.npz', **array_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--pic_path', type=str)
    args = parser.parse_args()
    main()