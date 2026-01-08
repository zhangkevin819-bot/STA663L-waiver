#!/usr/bin/env python3
"""
Simple runner script - minimal dependencies
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print("=" * 60)
print("Advanced ML Pipeline - Simple Demo")
print("=" * 60)

# Check PyTorch
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device: {device}\n")

# Generate Sample Data
print("📊 Step 1: Generating sample data...")
np.random.seed(42)
n_samples = 1000
n_features = 10

X = np.random.randn(n_samples, n_features)
y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)

print(f"   Dataset: {n_samples} samples, {n_features} features")
print(f"   Classes: {np.unique(y, return_counts=True)}")

# Create Simple Model
print("\n🧠 Step 2: Creating neural network...")

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        return self.net(x)

model = SimpleNet(n_features).to(device)
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Prepare Data
print("\n📦 Step 3: Preparing data loaders...")

train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(y_train)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.LongTensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

print(f"   Train samples: {len(train_dataset)}")
print(f"   Test samples: {len(test_dataset)}")

# Training
print("\n🎯 Step 4: Training model...")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    
    accuracy = 100 * correct / total
    avg_loss = train_loss / len(train_loader)
    
    print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Evaluation
print("\n📈 Step 5: Evaluating on test set...")

model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)
        test_correct += (predictions == targets).sum().item()
        test_total += targets.size(0)

test_accuracy = 100 * test_correct / test_total
print(f"   Test Accuracy: {test_accuracy:.2f}%")

# Demo Prediction
print("\n🔮 Step 6: Demo predictions...")

model.eval()
sample_inputs = torch.FloatTensor(X_test[:5]).to(device)
with torch.no_grad():
    outputs = model(sample_inputs)
    probabilities = torch.softmax(outputs, dim=-1)
    predictions = outputs.argmax(dim=-1)

print("\n   Sample Predictions:")
for i in range(5):
    pred = predictions[i].item()
    prob = probabilities[i][pred].item()
    actual = y_test[i]
    status = "✓" if pred == actual else "✗"
    print(f"   {status} Sample {i+1}: Predicted={pred}, Actual={actual}, Confidence={prob:.2%}")

print("\n" + "=" * 60)
print("✅ Demo completed successfully!")
print("=" * 60)
