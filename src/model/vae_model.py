import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# === CONFIGURATION ===
USE_OUTPUT_FILES = False # Set to True to use output files, False to use MIMIC files
INPUT_FILE = (
    "../../data/vae_data/vae_input_hosp_sample.csv"
    if USE_OUTPUT_FILES else
    "../../data/vae_data/vae_input_hosp.csv"
)

ANOMALY_OUTPUT_FOLDER = "anomalies"
ANOMALY_OUTPUT_FILE = "vae_anomalies_subject_ids.csv"
os.makedirs(ANOMALY_OUTPUT_FOLDER, exist_ok=True)

# === Step 1: Load and Preprocess Data ===
df = pd.read_csv(INPUT_FILE)
subject_ids = df["subject_id"].values
features_df = df.select_dtypes(include=[np.number]).drop(columns=["subject_id"], errors="ignore")

print("Applying signed log1p transform...")
log_transform = lambda x: np.sign(x) * np.log1p(np.abs(x))
features_df = features_df.apply(log_transform)

print("Standardizing features...")
X_scaled = np.empty_like(features_df.values, dtype=np.float32)
scaler = StandardScaler()
for i, col in enumerate(tqdm(features_df.columns, desc="scaling")):
    std = features_df[col].std()
    if std == 0 or np.isnan(std):
        X_scaled[:, i] = 0.0
    else:
        X_scaled[:, i] = scaler.fit_transform(features_df[[col]]).flatten()
X = X_scaled

if not np.isfinite(X).all():
    raise ValueError("Scaled data contains NaN or Inf.")

X_train, X_test, ids_train, ids_test = train_test_split(X, subject_ids, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=256, shuffle=True)

# === Step 2: VAE Definition ===
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc_dec1 = nn.Linear(latent_dim, 128)
        self.fc_out = nn.Linear(128, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc_dec1(z))
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# === Step 3: Training Loop ===
def vae_loss(x_recon, x, mu, logvar, kl_weight=1.0):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_div

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-4)

EPOCHS = 50
vae.train()
for epoch in range(EPOCHS):
    total_loss = 0
    kl_weight = min(1.0, epoch / 10)
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for batch in loop:
        x = batch[0].to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(x)
        loss = vae_loss(x_recon, x, mu, logvar, kl_weight)
        if torch.isnan(loss):
            print("NaN detected in loss!")
            exit(1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# === Step 4: Anomaly Detection ===
vae.eval()
X_test_tensor = torch.tensor(X_test).to(device)

mse_list = []
with torch.no_grad():
    for i in tqdm(range(0, len(X_test_tensor), 256), desc="Computing MSE"):
        batch = X_test_tensor[i:i+256]
        recon, _, _ = vae(batch)
        batch_mse = ((batch - recon) ** 2).mean(dim=1).cpu().numpy()
        mse_list.append(batch_mse)
mse = np.concatenate(mse_list)

# Compute Z-scores and apply threshold
mean_mse = np.mean(mse)
std_mse = np.std(mse)
z_scores = (mse - mean_mse) / std_mse
z_threshold = 3.0
anomalies = z_scores > z_threshold

print(f"Mean MSE: {mean_mse:.4f}, Std MSE: {std_mse:.4f}")
print(f"Z-score threshold: {z_threshold}")
print(f"Detected {anomalies.sum()} anomalies out of {len(mse)} test samples.")

# Save anomaly subject IDs
anomaly_ids = ids_test[anomalies]
anomaly_df = pd.DataFrame({"subject_id": anomaly_ids})
output_path = os.path.join(ANOMALY_OUTPUT_FOLDER, ANOMALY_OUTPUT_FILE)
anomaly_df.to_csv(output_path, index=False)
print(f"Subject IDs of anomalies saved to {output_path}")

# === Visualization: MSE Histogram (log y-axis, wide bins) ===
mse_hist_path = os.path.join(ANOMALY_OUTPUT_FOLDER, "mse_reconstruction_error_histogram.png")
plt.figure(figsize=(10, 6))
plt.hist(mse[~np.isnan(mse)], bins=100)
plt.yscale("log")
plt.axvline(mean_mse + z_threshold * std_mse, color='red', linestyle='dashed', label=f'Z = {z_threshold}')
plt.title("Reconstruction Error (MSE) Histogram")
plt.xlabel("MSE")
plt.ylabel("Log Count")
plt.legend()
plt.tight_layout()
plt.savefig(mse_hist_path)
plt.close()

# === Visualization: Z-score Histogram ===
zscore_hist_path = os.path.join(ANOMALY_OUTPUT_FOLDER, "zscore_histogram.png")
plt.figure(figsize=(10, 6))
plt.hist(z_scores[~np.isnan(z_scores)], bins=100)
plt.axvline(z_threshold, color='red', linestyle='dashed', label=f'Z = {z_threshold}')
plt.title("Z-score of Reconstruction Errors")
plt.xlabel("Z-score")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(zscore_hist_path)
plt.close()
