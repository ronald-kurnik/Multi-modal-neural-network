"""
Demo 1 (Simple Forward Version)
Multi-modal neural network that learns a material fingerprint
from plasma sensor data (spectral + electrical + thermal + process).

No domain-adversarial training – just a clean forward network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Synthetic multi-modal plasma data
# ------------------------------------------------------------
def generate_plasma_data(n_samples=2000, n_materials=4, seed=42):
    rng = np.random.default_rng(seed)
    domain = rng.integers(0, n_materials, size=n_samples)

    power = rng.uniform(20, 80, n_samples)
    gas_flow = rng.uniform(1, 10, n_samples)
    speed = rng.uniform(5, 30, n_samples)

    # True properties
    thickness = 0.5 + 0.3 * domain + 0.01 * power + rng.normal(0, 0.05, n_samples)
    density   = 1.2 + 0.15 * domain - 0.005 * gas_flow + rng.normal(0, 0.03, n_samples)
    resistivity = 10 + 2 * domain + 0.1 * speed + rng.normal(0, 0.5, n_samples)

    # Optical Emission Spectrum (64 bins)
    wavelengths = np.linspace(300, 900, 64)
    spectra = np.zeros((n_samples, 64))
    for i in range(n_samples):
        base = np.exp(-0.5 * ((wavelengths - 450 - 20*domain[i]) / 30)**2)
        base += 0.6 * np.exp(-0.5 * ((wavelengths - 650 + 10*domain[i]) / 40)**2)
        base *= (0.5 + 0.01 * power[i])
        spectra[i] = base + rng.normal(0, 0.02, 64)

    # Electrical features
    elec = np.column_stack([
        0.8 + 0.1*domain + 0.005*power + rng.normal(0, 0.03, n_samples),
        30 + 5*domain - 0.1*gas_flow + rng.normal(0, 1, n_samples),
        0.2 + 0.05*domain + rng.normal(0, 0.02, n_samples)
    ])

    # Thermal features
    thermal = np.column_stack([
        40 + 0.2*power + 2*domain + rng.normal(0, 1, n_samples),
        5 + 0.5*domain + rng.normal(0, 0.3, n_samples)
    ])

    process = np.column_stack([power, gas_flow, speed])

    X = {
        "spectral": spectra.astype(np.float32),
        "electrical": elec.astype(np.float32),
        "thermal": thermal.astype(np.float32),
        "process": process.astype(np.float32)
    }
    y = np.column_stack([thickness, density, resistivity]).astype(np.float32)
    domains = domain.astype(np.int64)
    return X, y, domains, wavelengths

X, y, domains, wavelengths = generate_plasma_data()
print("Data shapes:", {k: v.shape for k, v in X.items()})

# ------------------------------------------------------------
# 2. Simple Forward Multi-Modal Network
# ------------------------------------------------------------
class SimpleMultiModalNet(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Modality-specific encoders
        self.spec_enc = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32)
        )
        self.elec_enc = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16)
        )
        self.therm_enc = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 8)
        )
        self.proc_enc = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 8)
        )

        # Fusion → compact fingerprint
        self.fusion = nn.Sequential(
            nn.Linear(32 + 16 + 8 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Property prediction head
        self.prop_head = nn.Linear(latent_dim, 3)

    def forward(self, spec, elec, therm, proc):
        h = torch.cat([
            self.spec_enc(spec),
            self.elec_enc(elec),
            self.therm_enc(therm),
            self.proc_enc(proc)
        ], dim=1)
        z = self.fusion(h)          # 32-D material fingerprint
        props = self.prop_head(z)   # predict thickness, density, resistivity
        return z, props

# ------------------------------------------------------------
# 3. Training
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMultiModalNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Use materials 0,1,2 for training – hold out material 3
train_mask = domains < 3
test_mask  = domains == 3

def to_tensor(x):
    return torch.from_numpy(x).to(device)

train_loader = DataLoader(
    TensorDataset(
        to_tensor(X["spectral"][train_mask]),
        to_tensor(X["electrical"][train_mask]),
        to_tensor(X["thermal"][train_mask]),
        to_tensor(X["process"][train_mask]),
        to_tensor(y[train_mask])
    ),
    batch_size=64, shuffle=True
)

print("\nTraining simple forward multi-modal network...")
for epoch in range(40):
    model.train()
    total_loss = 0.0
    for spec, elec, therm, proc, props_true in train_loader:
        z, props_pred = model(spec, elec, therm, proc)
        loss = F.mse_loss(props_pred, props_true)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}  loss = {total_loss / len(train_loader):.4f}")

# ------------------------------------------------------------
# 4. Evaluation on held-out material
# ------------------------------------------------------------
model.eval()
with torch.no_grad():
    z_test, props_pred = model(
        to_tensor(X["spectral"][test_mask]),
        to_tensor(X["electrical"][test_mask]),
        to_tensor(X["thermal"][test_mask]),
        to_tensor(X["process"][test_mask])
    )
    props_pred = props_pred.cpu().numpy()
    props_true = y[test_mask]
    z_test = z_test.cpu().numpy()

print("\n=== Generalization to held-out material (domain 3) ===")
for i, name in enumerate(["Thickness", "Density", "Resistivity"]):
    r2 = r2_score(props_true[:, i], props_pred[:, i])
    print(f"{name:12s}  R² = {r2:.3f}")

# ------------------------------------------------------------
# 5. Plots
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Example spectra
for d in range(4):
    idx = np.where(domains == d)[0][0]
    axes[0, 0].plot(wavelengths, X["spectral"][idx], label=f"Material {d}")
axes[0, 0].set_title("Example Optical Emission Spectra")
axes[0, 0].set_xlabel("Wavelength (nm)")
axes[0, 0].set_ylabel("Intensity")
axes[0, 0].legend()

# 2. Predicted vs True
for i, name in enumerate(["Thickness", "Density", "Resistivity"]):
    axes[0, 1].scatter(props_true[:, i], props_pred[:, i], s=15, alpha=0.6, label=name)
axes[0, 1].plot([props_true.min(), props_true.max()],
                [props_true.min(), props_true.max()], "k--", lw=1)
axes[0, 1].set_title("Held-out Material: Predicted vs True")
axes[0, 1].set_xlabel("True")
axes[0, 1].set_ylabel("Predicted")
axes[0, 1].legend()

# 3. Latent fingerprint colored by thickness
sc = axes[1, 0].scatter(z_test[:, 0], z_test[:, 1], c=props_true[:, 0],
                        cmap="viridis", s=20)
axes[1, 0].set_title("Latent Fingerprint (colored by Thickness)")
axes[1, 0].set_xlabel("Latent dim 1")
axes[1, 0].set_ylabel("Latent dim 2")
plt.colorbar(sc, ax=axes[1, 0], fraction=0.046)

# 4. Residuals
residuals = props_true - props_pred
axes[1, 1].hist(residuals[:, 0], bins=30, alpha=0.7, label="Thickness")
axes[1, 1].hist(residuals[:, 1], bins=30, alpha=0.7, label="Density")
axes[1, 1].set_title("Prediction Residuals (held-out)")
axes[1, 1].set_xlabel("Residual")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("demo1_simple_forward.png", dpi=140, bbox_inches="tight")
print("\nSaved: demo1_simple_forward.png")
plt.show()