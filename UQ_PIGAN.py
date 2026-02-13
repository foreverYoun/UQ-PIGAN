# UQ-PIGAN.py
"""
Complete PID-GAN training + generation pipeline for uncertainty quantification
based on training-set error.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from config import *
import random
from data_preprocessing import DataPreprocessor, load_data
from physics_model import hybrid_physics_model

random.seed(2025)


def calculate_physics_uncertainty(train_features, train_labels, preprocessor):
    """Estimate physics-model uncertainty from training data."""
    velocity_norm = train_features[:, 3]
    min_v = preprocessor.min_values['侵彻速度']
    max_v = preprocessor.max_values['侵彻速度']
    velocity_real = velocity_norm * (max_v - min_v) + min_v

    real_features = np.zeros_like(train_features)
    for i, fname in enumerate(IMPACT_FEATURES):
        min_val = preprocessor.min_values[fname]
        max_val = preprocessor.max_values[fname]
        real_features[:, i] = train_features[:, i] * (max_val - min_val) + min_val

    m, d, l, v, fc, psi = [real_features[:, i] for i in range(6)]

    h_pred = np.array([
        hybrid_physics_model(m[i], d[i], l[i], v[i], fc[i], psi[i])
        if v[i] < V_JONES_MAX else None
        for i in range(len(train_features))
    ])

    h_real = train_labels.squeeze()

    mask_forrestal = velocity_real < V_FORRESTAL_MAX
    if mask_forrestal.sum() > 0:
        err_f = np.abs(h_real[mask_forrestal] - h_pred[mask_forrestal]) / (np.abs(h_real[mask_forrestal]) + 1e-6)
        uncertainty_forrestal = float(np.mean(err_f))
    else:
        uncertainty_forrestal = 0.10

    mask_jones = (velocity_real >= V_FORRESTAL_MAX) & (velocity_real < V_JONES_MAX)
    if mask_jones.sum() > 0:
        err_j = np.abs(h_real[mask_jones] - h_pred[mask_jones]) / (np.abs(h_real[mask_jones]) + 1e-6)
        uncertainty_jones = float(np.mean(err_j))
    else:
        uncertainty_jones = 0.20

    return uncertainty_forrestal, uncertainty_jones


def compute_physics_consistency(data, label, preprocessor, uncertainty_forrestal, uncertainty_jones, kappa=KAPPA_PHYSICS):
    """Compute physics consistency score using the Forrestal-Jones model."""
    device = data.device
    batch_size = data.shape[0]

    data_np = data.detach().cpu().numpy()
    real_features = np.zeros_like(data_np)
    for i, fname in enumerate(IMPACT_FEATURES):
        min_val = preprocessor.min_values[fname]
        max_val = preprocessor.max_values[fname]
        real_features[:, i] = data_np[:, i] * (max_val - min_val) + min_val

    m, d, l, v, fc, psi = [real_features[:, i] for i in range(6)]

    h_pred = np.array([
        hybrid_physics_model(m[i], d[i], l[i], v[i], fc[i], psi[i])
        for i in range(batch_size)
    ])

    h_real = label.detach().cpu().numpy().squeeze()
    h_real_safe = np.where(np.abs(h_real) < 1e-3, 1e-3, h_real)
    relative_error = np.abs(h_real - h_pred) / np.abs(h_real_safe)

    sigma = np.zeros(batch_size)
    mask_forrestal = v < V_FORRESTAL_MAX
    mask_jones = (v >= V_FORRESTAL_MAX) & (v < V_JONES_MAX)
    mask_invalid = v >= V_JONES_MAX

    sigma[mask_forrestal] = uncertainty_forrestal
    sigma[mask_jones] = uncertainty_jones
    sigma[mask_invalid] = UNCERTAINTY_INVALID

    physics_score = np.exp(-0.5 * ((relative_error - sigma) / (kappa * sigma + 1e-6)) ** 2)
    physics_score[mask_invalid] = 0.0
    physics_score[np.isnan(physics_score)] = 0.5

    return torch.FloatTensor(physics_score).to(device)


class PenetrationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features) if not isinstance(features, torch.Tensor) else features
        self.labels = torch.FloatTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        if self.labels.dim() == 1:
            self.labels = self.labels.unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_normalize_data():
    print("=" * 50 + "\nLoading data...\n" + "=" * 50)
    train_df, test_df = load_data(TRAIN_CSV_PATH, TEST_CSV_PATH)
    preprocessor = DataPreprocessor(min_max_path=MIN_MAX_PATH)
    train_features, train_labels, test_features, test_labels = preprocessor.fit(train_df, test_df)
    return train_features.numpy(), train_labels.numpy(), test_features.numpy(), test_labels.numpy(), preprocessor


class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dims, output_scale=1.0):
        super(Generator, self).__init__()
        self.output_scale = output_scale
        layers = []
        input_dim = latent_dim + condition_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        return self.model(x) * self.output_scale


class Discriminator(nn.Module):
    def __init__(self, condition_dim, hidden_dims):
        super(Discriminator, self).__init__()
        layers = []
        input_dim = condition_dim + 1 + 1
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.3)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, condition, label, physics_score):
        if label.dim() == 1:
            label = label.unsqueeze(1)
        if physics_score.dim() == 1:
            physics_score = physics_score.unsqueeze(1)
        x = torch.cat([condition, label, physics_score], dim=1)
        return self.model(x)


def train_pidgan(train_features, train_labels, preprocessor, uncertainty_forrestal, uncertainty_jones):
    """Train PID-GAN and save checkpoints and logs."""
    print("=" * 50 + "\nInitializing model...\n" + "=" * 50)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    dataset = PenetrationDataset(train_features, train_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    label_mean, label_std = train_labels.mean(), train_labels.std()
    output_scale = label_std * 3

    G = Generator(LATENT_DIM, CONDITION_DIM, GENERATOR_HIDDEN_DIMS, output_scale=output_scale).to(device)
    D = Discriminator(CONDITION_DIM, DISCRIMINATOR_HIDDEN_DIMS).to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=2000, gamma=0.8)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=2000, gamma=0.9)
    criterion = nn.BCELoss()

    target_mean = torch.tensor(label_mean, device=device)
    target_std = torch.tensor(label_std, device=device)

    training_history = []

    print("=" * 50 + "\nStart training...\n" + "=" * 50)
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        d_losses, g_losses, d_real_accs, d_fake_accs = [], [], [], []
        avg_real_physics, avg_fake_physics, avg_invalid_ratios = [], [], []

        for real_conditions, real_labels in dataloader:
            batch_size = real_conditions.size(0)
            real_conditions, real_labels = real_conditions.to(device), real_labels.to(device)

            real_label_smooth = torch.FloatTensor(batch_size, 1).uniform_(0.9, 1.0).to(device)
            fake_label_smooth = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.1).to(device)

            real_physics_scores = compute_physics_consistency(
                real_conditions, real_labels, preprocessor, uncertainty_forrestal, uncertainty_jones
            )

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_labels = G(noise, real_conditions)

            fake_physics_scores = compute_physics_consistency(
                real_conditions, fake_labels, preprocessor, uncertainty_forrestal, uncertainty_jones
            )

            optimizer_D.zero_grad()
            d_real_loss = criterion(D(real_conditions, real_labels, real_physics_scores), real_label_smooth)
            d_fake_loss = criterion(
                D(real_conditions, fake_labels.detach(), fake_physics_scores.detach()),
                fake_label_smooth
            )
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            d_fake_output = D(real_conditions, fake_labels, fake_physics_scores)
            g_adv_loss = criterion(d_fake_output, real_label_smooth)
            g_dist_loss = torch.abs(fake_labels.mean() - target_mean) + torch.abs(fake_labels.std() - target_std)
            g_loss = g_adv_loss + 0.3 * g_dist_loss
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
            optimizer_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            d_real_accs.append((D(real_conditions, real_labels, real_physics_scores) > 0.5).float().mean().item())
            d_fake_accs.append((d_fake_output < 0.5).float().mean().item())
            avg_real_physics.append(real_physics_scores.mean().item())
            avg_fake_physics.append(fake_physics_scores.mean().item())
            avg_invalid_ratios.append((fake_physics_scores == 0.0).float().mean().item())

        scheduler_G.step()
        scheduler_D.step()

        epoch_metrics = {
            'epoch': epoch,
            'discriminator_loss': np.mean(d_losses),
            'generator_loss': np.mean(g_losses),
            'd_real_accuracy': np.mean(d_real_accs),
            'd_fake_accuracy': np.mean(d_fake_accs),
            'real_physics_score': np.mean(avg_real_physics),
            'fake_physics_score': np.mean(avg_fake_physics),
            'invalid_sample_ratio': np.mean(avg_invalid_ratios),
            'lr_g': scheduler_G.get_last_lr()[0],
            'lr_d': scheduler_D.get_last_lr()[0],
            'epoch_time_s': time.time() - epoch_start_time
        }
        training_history.append(epoch_metrics)

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] | "
                f"D Loss: {epoch_metrics['discriminator_loss']:.4f} | "
                f"G Loss: {epoch_metrics['generator_loss']:.4f} | "
                f"D(real): {epoch_metrics['d_real_accuracy']:.3f} | "
                f"D(fake): {epoch_metrics['d_fake_accuracy']:.3f} | "
                f"eta_real: {epoch_metrics['real_physics_score']:.3f} | "
                f"eta_fake: {epoch_metrics['fake_physics_score']:.3f} | "
                f"Overspeed ratio: {epoch_metrics['invalid_sample_ratio']:.2%}"
            )

    total_time = time.time() - start_time
    print("=" * 50 + f"\nTraining finished! Total time: {total_time / 60:.2f} minutes\n" + "=" * 50)

    log_df = pd.DataFrame(training_history)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, 'uq_pigan_training_log.csv')
    log_df.to_csv(log_path, index=False)
    print(f"Training log saved to: {log_path}")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    g_path = os.path.join(MODEL_OUTPUT_DIR, 'uq_pigan_generator.pth')
    d_path = os.path.join(MODEL_OUTPUT_DIR, 'uq_pigan_discriminator.pth')
    torch.save(G.state_dict(), g_path)
    torch.save(D.state_dict(), d_path)
    print(f"Models saved to: {MODEL_OUTPUT_DIR}/")

    return G, D


def generate_data(G, num_samples, preprocessor, uncertainty_forrestal, uncertainty_jones):
    print("=" * 50 + f"\nGenerating {num_samples} samples...\n" + "=" * 50)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    G.eval()

    train_df, _ = load_data(TRAIN_CSV_PATH, TEST_CSV_PATH)
    train_df_filtered = train_df[train_df['侵彻速度'] < 1500]

    sampled_conditions = train_df_filtered[IMPACT_FEATURES].sample(n=num_samples, replace=True).values
    normalized_conditions = np.zeros_like(sampled_conditions)
    for i, fname in enumerate(IMPACT_FEATURES):
        min_val, max_val = preprocessor.min_values[fname], preprocessor.max_values[fname]
        normalized_conditions[:, i] = (sampled_conditions[:, i] - min_val) / (max_val - min_val)

    generated_data = []
    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch_conditions = torch.FloatTensor(normalized_conditions[i:i + BATCH_SIZE]).to(device)
            noise = torch.randn(batch_conditions.size(0), LATENT_DIM, device=device)
            fake_labels = G(noise, batch_conditions).cpu().numpy()
            batch_data = np.hstack([sampled_conditions[i:i + batch_conditions.size(0)], fake_labels])
            generated_data.append(batch_data)

    generated_data = np.vstack(generated_data)
    df = pd.DataFrame(generated_data, columns=IMPACT_FEATURES + [LABEL_FEATURE])
    df = df[df['侵彻速度'] < 1500]

    output_path = os.path.join(OUTPUT_DIR, 'uqpigan_generated_data.csv')
    df.to_csv(output_path, index=False, encoding='gbk')
    print(f"Generated data saved to: {output_path}")
    return df


def main():
    print("Step 1: Load data")
    train_features, train_labels, _, _, preprocessor = load_and_normalize_data()
    uncertainty_forrestal, uncertainty_jones = calculate_physics_uncertainty(train_features, train_labels, preprocessor)

    global UNCERTAINTY_FORRESTAL, UNCERTAINTY_JONES
    UNCERTAINTY_FORRESTAL = uncertainty_forrestal
    UNCERTAINTY_JONES = uncertainty_jones

    print("\nStep 2: Train PID-GAN model")
    G, D = train_pidgan(train_features, train_labels, preprocessor, uncertainty_forrestal, uncertainty_jones)

    print("\nStep 3: Generate synthetic data")
    generate_data(G, NUM_GENERATE_SAMPLES, preprocessor, uncertainty_forrestal, uncertainty_jones)

    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()

