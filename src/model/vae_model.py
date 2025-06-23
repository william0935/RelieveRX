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
USE_OUTPUT_FILES = True  # True = use sample file, False = use full dataset

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

# Preserve subject_id for tracking
subject_ids = df["subject_id"].values

# Extract numeric features (excluding subject_id)
features_df = df.select_dtypes(include=[np.number]).drop(columns=["subject_id"], errors="ignore")
features_df.fillna(features_df.mean(), inplace=True)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(features_df.astype(np.float32))

# Train-test split (track subject_ids for test set)
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
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

EPOCHS = 50
vae.train()
for epoch in range(EPOCHS):
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for batch in loop:
        x = batch[0].to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(x)
        loss = vae_loss(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item() / len(x))
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(X_train):.4f}")

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

# Thresholding (top 5% highest MSE = anomalies)
threshold = np.percentile(mse, 95)
anomalies = mse > threshold

# Extract subject_ids of anomalies
anomaly_ids = ids_test[anomalies]
anomaly_df = pd.DataFrame({"subject_id": anomaly_ids})

# Save subject IDs to anomalies folder
output_path = os.path.join(ANOMALY_OUTPUT_FOLDER, ANOMALY_OUTPUT_FILE)
anomaly_df.to_csv(output_path, index=False)

# === Optional: Visualization ===
plt.hist(mse, bins=50)
plt.axvline(threshold, color='red', linestyle='dashed', label='Anomaly Threshold')
plt.title("Reconstruction Error Histogram")
plt.xlabel("MSE")
plt.ylabel("Count")
plt.legend()
plt.show()

print(f"Detected {anomalies.sum()} anomalies out of {len(anomalies)} test samples.")
print(f"Subject IDs of anomalies saved to {output_path}")
