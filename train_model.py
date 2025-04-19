# train_model.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib

# Paths
DATA_PATH = 'data/DDI_data.csv'
SAVED_MODELS_PATH = 'saved_models/'

# Create directory if not exists
os.makedirs(SAVED_MODELS_PATH, exist_ok=True)

# Step 1: Data Preprocessing
print("Loading and preprocessing data...")
ddi_data = pd.read_csv(DATA_PATH)

ddi_data = ddi_data.dropna(subset=['drug1_name', 'drug2_name', 'interaction_type'])

# Encode drug names
drug_encoder = LabelEncoder()
all_drugs = pd.concat([ddi_data['drug1_name'], ddi_data['drug2_name']]).unique()
drug_encoder.fit(all_drugs)

ddi_data['drug1_id'] = drug_encoder.transform(ddi_data['drug1_name'])
ddi_data['drug2_id'] = drug_encoder.transform(ddi_data['drug2_name'])

# Encode interaction types
interaction_encoder = OneHotEncoder(sparse=False)
y = interaction_encoder.fit_transform(ddi_data[['interaction_type']])

# Node features
num_drugs = len(drug_encoder.classes_)
x = torch.eye(num_drugs)

# Edge list
edge_index = torch.tensor(ddi_data[['drug1_id', 'drug2_id']].values.T, dtype=torch.long)

# Labels
labels = torch.tensor(y, dtype=torch.float32)

# Train/test split
train_idx, test_idx = train_test_split(range(len(ddi_data)), test_size=0.2, random_state=42)
drug1_idx = torch.tensor(ddi_data['drug1_id'].values)
drug2_idx = torch.tensor(ddi_data['drug2_id'].values)

# Step 2: Model Definition
print("Building model...")

class DDIPredictor(nn.Module):
    def __init__(self, num_drug_features, num_interaction_types):
        super(DDIPredictor, self).__init__()
        self.conv1 = GCNConv(num_drug_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = nn.Linear(64*2, num_interaction_types)
        
    def forward(self, x, edge_index, drug1_idx, drug2_idx):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        drug1_feat = x[drug1_idx]
        drug2_feat = x[drug2_idx]
        combined = torch.cat([drug1_feat, drug2_feat], dim=1)
        return torch.sigmoid(self.fc(combined))

model = DDIPredictor(num_drug_features=num_drugs, num_interaction_types=y.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# Step 3: Training
print("Training model...")
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index, drug1_idx[train_idx], drug2_idx[train_idx])
    loss = criterion(output, labels[train_idx])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# Step 4: Save the model and encoders
print("Saving model and encoders...")
torch.save(model.state_dict(), os.path.join(SAVED_MODELS_PATH, 'ddi_model.pth'))
joblib.dump(drug_encoder, os.path.join(SAVED_MODELS_PATH, 'drug_encoder.pkl'))
joblib.dump(interaction_encoder, os.path.join(SAVED_MODELS_PATH, 'interaction_encoder.pkl'))

print("✅ Training complete and models saved!")
