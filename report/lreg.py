from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
output_path="path/iemocap/qwen/input/"

# Load the filtered dataset
with open("pathiemocap/qwen/iemocap_sumarized_data.pkl", "rb") as f:
    filtered_data = pickle.load(f)

# Aggregate embeddings across all samples
aggregated_text_embeddings = []
aggregated_audio_embeddings = []
num_layers = None

for UID, data in filtered_data.items():
    text_embedding = data["input_text_embedding"]  # Shape: (num_layers, 5120)
    audio_embedding = data["input_audio_embedding"]  # Shape: (num_layers, 5120)
    
    if num_layers is None:
        num_layers = text_embedding.shape[0]
    
    aggregated_text_embeddings.append(text_embedding)
    aggregated_audio_embeddings.append(audio_embedding)

aggregated_text_embeddings = np.stack(aggregated_text_embeddings, axis=1)  # Shape: (num_layers, N_samples, 5120)
aggregated_audio_embeddings = np.stack(aggregated_audio_embeddings, axis=1)  # Shape: (num_layers, N_samples, 5120)

# Split data into training and validation sets once for all layers
N_samples = aggregated_text_embeddings.shape[1]
sample_indices = np.arange(N_samples)
train_indices, val_indices = train_test_split(sample_indices, test_size=0.1, random_state=42)

t2a_results = []
a2t_results = []

models_text_to_audio = {}  # Store trained models for Text-to-Audio with layer indices
models_audio_to_text = {}  # Store trained models for Audio-to-Text with layer indices

for layer_idx in range(num_layers):
    print(f"Processing layer {layer_idx + 1}...")

    # Select embeddings for the current layer
    X_text = aggregated_text_embeddings[layer_idx]  # Shape: (N_samples, 5120)
    Y_audio = aggregated_audio_embeddings[layer_idx]  # Shape: (N_samples, 5120)

    # Split using pre-determined indices
    X_train_text = X_text[train_indices]
    X_val_text = X_text[val_indices]
    Y_train_audio = Y_audio[train_indices]
    Y_val_audio = Y_audio[val_indices]

    # Text-to-Audio Regression
    reg_t2a = LinearRegression()
    reg_t2a.fit(X_train_text, Y_train_audio)
    Y_pred_audio = reg_t2a.predict(X_val_text)
    mse_loss_t2a = mean_squared_error(Y_val_audio, Y_pred_audio)
    t2a_results.append({
        "layer_idx": layer_idx + 1,
        "mse_loss": mse_loss_t2a
    })
    models_text_to_audio[f"layer_{layer_idx+1}"] = reg_t2a

    # Audio-to-Text Regression
    X_audio = aggregated_audio_embeddings[layer_idx]  # Shape: (N_samples, 5120)
    Y_text = aggregated_text_embeddings[layer_idx]  # Shape: (N_samples, 5120)

    X_train_audio = X_audio[train_indices]
    X_val_audio = X_audio[val_indices]
    Y_train_text = Y_text[train_indices]
    Y_val_text = Y_text[val_indices]

    reg_a2t = LinearRegression()
    reg_a2t.fit(X_train_audio, Y_train_text)
    Y_pred_text = reg_a2t.predict(X_val_audio)
    mse_loss_a2t = mean_squared_error(Y_val_text, Y_pred_text)
    a2t_results.append({
        "layer_idx": layer_idx + 1,
        "mse_loss": mse_loss_a2t
    })
    models_audio_to_text[f"layer_{layer_idx+1}"] = reg_a2t

# Save models
# os.makedirs(output_path+"models", exist_ok=True)
# with open(output_path+"models/text_to_audio_models.pkl", "wb") as f:
#     pickle.dump(models_text_to_audio, f)
# with open(output_path+"models/audio_to_text_models.pkl", "wb") as f:
#     pickle.dump(models_audio_to_text, f)

print("✅ Models saved successfully!")

# Visualization of MSE Loss
t2a_mse = [result["mse_loss"] for result in t2a_results]
a2t_mse = [result["mse_loss"] for result in a2t_results]
layer_indices = [result["layer_idx"] for result in t2a_results]

plt.figure(figsize=(8, 6))
plt.plot(layer_indices, t2a_mse, marker='o', label="Text-to-Audio MSE (Validation)")
plt.plot(layer_indices, a2t_mse, marker='o', linestyle='--', label="Audio-to-Text MSE (Validation)")
plt.title("MSE Loss Across Layers", fontsize=14)
plt.xlabel("Layer Index", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.grid(True)
plt.xticks(layer_indices)
plt.legend()
plt.savefig(output_path+"mse_loss_trends_across_layers.png", dpi=300)
plt.close()

# # Visualization of Coefficients Heatmap
# coefficients_t2a = np.array([model.coef_.flatten() for model in models_text_to_audio.values()])
# coefficients_a2t = np.array([model.coef_.flatten() for model in models_audio_to_text.values()])

# plt.figure(figsize=(10, 8))
# sns.heatmap(coefficients_t2a, cmap="coolwarm", xticklabels=False, yticklabels=layer_indices)
# plt.title("Text-to-Audio Regression Coefficients", fontsize=14)
# plt.xlabel("Feature Index", fontsize=12)
# plt.ylabel("Layer Index", fontsize=12)
# plt.savefig(output_path+"regression_coefficients_text_to_audio.png", dpi=300)
# plt.close()

# plt.figure(figsize=(10, 8))
# sns.heatmap(coefficients_a2t, cmap="coolwarm", xticklabels=False, yticklabels=layer_indices)
# plt.title("Audio-to-Text Regression Coefficients", fontsize=14)
# plt.xlabel("Feature Index", fontsize=12)
# plt.ylabel("Layer Index", fontsize=12)
# plt.savefig(output_path+"regression_coefficients_audio_to_text.png", dpi=300)
# plt.close()

# print("✅ Visualizations for MSE trends and regression coefficients saved successfully!")
