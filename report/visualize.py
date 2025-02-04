import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
import umap as umap  # ✅ Fix UMAP Import
import os
output_path="/path/iemocap/salmon/input/"
# Load the filtered dataset
with open("path/librisqa_sumarized_data.pkl", "rb") as f:
    filtered_data = pickle.load(f)
os.makedirs(output_path, exist_ok=True)
# Store correlation results across dataset
layer_correlation_matrix = []

# Process each pair in the dataset
for UID, data in filtered_data.items():
    print(f"Processing UID: {UID}")

    # Load stored embeddings
    audio_embedding = data["audio_embedding"]  # Shape: (num_layers, 5120)
    text_embedding = data["text_embedding"]  # Shape: (num_layers, 5120)

    # **1️⃣ Compute Pearson Correlation Between Layers**
    num_layers = audio_embedding.shape[0]
    correlation_matrix = np.zeros((num_layers, num_layers))

    for i in range(num_layers):  # Audio layers
        for j in range(num_layers):  # Text layers
            corr, _ = pearsonr(audio_embedding[i], text_embedding[j])
            correlation_matrix[i, j] = corr

    layer_correlation_matrix.append(correlation_matrix)

# **2️⃣ Compute Global Mean Correlation Matrix Across Dataset**
global_correlation_matrix = np.mean(layer_correlation_matrix, axis=0)  # Shape: (num_layers, num_layers)


# **3️⃣ Scatter Plot of Pairwise Correlations**
audio_layer_indices, text_layer_indices = np.meshgrid(range( num_layers), range(num_layers))
pairwise_correlations = global_correlation_matrix.flatten()
plt.figure(figsize=(8, 6))
plt.scatter(audio_layer_indices.flatten(), text_layer_indices.flatten(),
            c=pairwise_correlations, cmap="coolwarm", s=50)
plt.colorbar(label="Correlation")
plt.title("Scatter Plot of Pairwise Correlations", fontsize=14)
plt.xlabel("Audio Layer", fontsize=12)
plt.ylabel("Text Layer", fontsize=12)
plt.grid(True)
plt.savefig(output_path+"pairwise_correlation_scatter.png", dpi=300)
plt.close()

# **3️⃣ Plot & Save Heatmap of Layer Correlations**
plt.figure(figsize=(10, 8))
sns.heatmap(global_correlation_matrix, annot=False, cmap="coolwarm", xticklabels=[f"Layer {i}" for i in range(num_layers)], yticklabels=[f"Layer {i}" for i in range(num_layers)])
plt.title("Pearson Correlation Between Audio & Text Layers")
plt.xlabel("Text Layer")
plt.ylabel("Audio Layer")
plt.savefig(output_path+"heatmap_audio_text_correlation.png")  # 
plt.close()

# **1️⃣ Layer-Wise Similarity Trends**
average_correlations = np.diag(global_correlation_matrix)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(average_correlations) + 1), average_correlations, marker='o', label="Layer-Wise Correlation")
plt.title("Layer-Wise Average Correlation Between Audio & Text Layers", fontsize=14)
plt.xlabel("Layer Index", fontsize=12)
plt.ylabel("Average Correlation", fontsize=12)
plt.grid(True)
plt.xticks(range(1, len(average_correlations) + 1))  # Ensure proper layer indexing from 1
plt.legend()
plt.savefig(output_path+"layer_wise_correlation_trend.png", dpi=300)
plt.close()



# **4️⃣ Save Hierarchical Clustering of Layers**
plt.figure(figsize=(12, 6))
linkage_matrix = linkage(global_correlation_matrix, method='ward')
dendrogram(linkage_matrix, labels=[f"Layer {i}" for i in range(num_layers)], leaf_rotation=90)
plt.title("Hierarchical Clustering of Audio & Text Layers")
plt.xlabel("Layer Index")
plt.ylabel("Distance")
plt.savefig(output_path+"hierarchical_clustering_audio_text.png")  # ✅ Save instead of show
plt.close()

# **5️⃣ PCA Projection (5120 → 2D) & Save**
pca = PCA(n_components=2)
all_layers = np.vstack((audio_embedding, text_embedding))  # Combine audio & text layers
pca_projection = pca.fit_transform(all_layers)

# Split into audio & text for visualization
audio_pca = pca_projection[:num_layers]
text_pca = pca_projection[num_layers:]

# Save PCA Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(audio_pca[:, 0], audio_pca[:, 1], label="Audio Layers", alpha=0.7, color="blue")
plt.scatter(text_pca[:, 0], text_pca[:, 1], label="Text Layers", alpha=0.7, color="red")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Projection of Audio & Text Layers")
plt.legend()
plt.grid()
plt.savefig(output_path+"pca_audio_text_layers.png")  # ✅ Save instead of show
plt.close()

# **6️⃣ UMAP Projection (5120 → 2D) & Save**
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_projection = umap_model.fit_transform(all_layers)

# Split into audio & text for visualization
audio_umap = umap_projection[:num_layers]
text_umap = umap_projection[num_layers:]

# Save UMAP Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(audio_umap[:, 0], audio_umap[:, 1], label="Audio Layers", alpha=0.7, color="blue")
plt.scatter(text_umap[:, 0], text_umap[:, 1], label="Text Layers", alpha=0.7, color="red")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.title("UMAP Projection of Audio & Text Layers")
plt.legend()
plt.grid()
plt.savefig(output_path+"umap_audio_text_layers.png")  # ✅ Save instead of show
plt.close()

print("✅ All visualizations saved successfully!")
