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
import json
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# 1. globals
INPUT_FILE = "../../data/model_data/feature_matrix.csv"
CATEGORY_MAPPING_FILE = "../../data/model_data/category_mappings.json"
ANOMALY_OUTPUT_FOLDER = "anomalies"
ANOMALY_OUTPUT_FILE = "vae_anomalies_subject_ids.csv"
os.makedirs(ANOMALY_OUTPUT_FOLDER, exist_ok=True)

# 2. load and preprocess data
ROWS_PER_PATIENT_PER_FILE = 100

print("loading data...")
df = pd.read_csv(INPUT_FILE)
subject_ids = df["subject_id"].values
features_df = df.select_dtypes(include=[np.number]).drop(columns=["subject_id"], errors="ignore")
print("loaded data!")

print("applying signed log1p transform...")
log_transform = lambda x: np.sign(x) * np.log1p(np.abs(x))
features_df = features_df.apply(log_transform)

print("standardizing features...")
X_scaled = np.empty_like(features_df.values, dtype=np.float32)
scaler = StandardScaler()
for i, col in enumerate(features_df.columns):
    std = features_df[col].std()
    if std == 0 or np.isnan(std):
        X_scaled[:, i] = 0.0
    else:
        X_scaled[:, i] = scaler.fit_transform(features_df[[col]]).flatten()
X = X_scaled

if not np.isfinite(X).all():
    raise ValueError("scaled data contains NaN or Inf.")

# splitting into training, validation, test
X_trainval, X_test, ids_trainval, ids_test = train_test_split(
    X, subject_ids, test_size=0.2, random_state=42
)
X_train, X_val, ids_train, ids_val = train_test_split(
    X_trainval, ids_trainval, test_size=0.25, random_state=42
)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=256, shuffle=False)
print("loaded training and validation!")

# 3. define vae
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

def vae_loss(x_recon, x, mu, logvar, kl_weight=1.0):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_div

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-4)

# 4. training loop (no early stopping, always 50 epochs, save best model)
EPOCHS = 50
best_val_loss = float("inf")

print("starting training...")
for epoch in range(EPOCHS):
    vae.train()
    total_loss = 0
    kl_weight = min(1.0, epoch / 10)

    for batch in train_loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(x)
        loss = vae_loss(x_recon, x, mu, logvar, kl_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # validation
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            x_recon, mu, logvar = vae(x)
            loss = vae_loss(x_recon, x, mu, logvar, kl_weight)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # save best model only
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(vae.state_dict(), "vae_best_model.pt")

# load best model before test evaluation
vae.load_state_dict(torch.load("vae_best_model.pt"))
print("finished training")

# 5. anomaly detection
vae.eval()
X_test_tensor = torch.tensor(X_test).to(device)

mse_list = []
with torch.no_grad():
    for i in range(0, len(X_test_tensor), 256):
        batch = X_test_tensor[i:i+256]
        recon, _, _ = vae(batch)
        batch_mse = ((batch - recon) ** 2).mean(dim=1).cpu().numpy()
        mse_list.append(batch_mse)
mse = np.concatenate(mse_list)

mean_mse = np.mean(mse)
std_mse = np.std(mse)
z_scores = (mse - mean_mse) / std_mse
z_threshold = 2.5
anomalies = z_scores > z_threshold

print(f"Mean MSE: {mean_mse:.4f}, Std MSE: {std_mse:.4f}")
print(f"Z-score threshold: {z_threshold}")
print(f"Detected {anomalies.sum()} anomalies out of {len(mse)} test samples.")

# 6. save anomaly subject IDs and full data rows
anomaly_ids = ids_test[anomalies]
anomaly_data = df[df["subject_id"].isin(anomaly_ids)].copy()

# 7. decode categorical columns
if os.path.exists(CATEGORY_MAPPING_FILE):
    with open(CATEGORY_MAPPING_FILE, "r") as f:
        mappings = json.load(f)
    for key, decode_map in mappings.items():
        for i in range(ROWS_PER_PATIENT_PER_FILE):
            col = f"{key}_{i}"
            if col in anomaly_data.columns:
                decode_map = {int(k): v for k, v in decode_map.items()}
                anomaly_data.loc[:, col] = anomaly_data[col].map(decode_map).fillna(anomaly_data[col])

output_path = os.path.join(ANOMALY_OUTPUT_FOLDER, ANOMALY_OUTPUT_FILE)
anomaly_data.to_csv(output_path, index=False)
print(f"Full anomaly data saved to {output_path}")

# 8. MSE Histogram
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

# Z-score Histogram
zscore_hist_path = os.path.join(ANOMALY_OUTPUT_FOLDER, "zscore_histogram.png")
plt.figure(figsize=(10, 6))
plt.hist(z_scores[~np.isnan(z_scores)], bins=100, range=(-4, 4))
plt.yscale("log")
plt.axvline(z_threshold, color='red', linestyle='dashed', label=f'Z = {z_threshold}')
plt.title("Z-score of Reconstruction Errors")
plt.xlabel("Z-score")
plt.ylabel("Log Count")
plt.legend()
plt.tight_layout()
plt.savefig(zscore_hist_path)
plt.close()

# 9. load labels from separate CSV
LABELS_FILE = "feature_matrix_labeled.csv"
labels_df = pd.read_csv(LABELS_FILE)  # contains columns: subject_id, label (0/1)

# align labels with test set
labels_test_df = labels_df.set_index("subject_id").loc[ids_test].reset_index()
labels_test = labels_test_df["at_risk"].values

# binary predictions from VAE (anomaly = 1, normal = 0)
preds = anomalies.astype(int)

# calculate metrics
precision = precision_score(labels_test, preds)
recall = recall_score(labels_test, preds)
cm = confusion_matrix(labels_test, preds)

print("\n=== VAE Anomaly Detection vs Labels ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("Confusion Matrix (rows=true, cols=pred):")
print(cm)
