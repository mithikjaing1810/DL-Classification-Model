# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model

<img width="797" height="748" alt="image" src="https://github.com/user-attachments/assets/3314324a-5c26-4447-a89e-ee90b99c5a94" />


### DESIGN STEPS

STEP 1: Load the dataset and required libraries.

STEP 2: Preprocess the data (clean, encode, and normalize).

STEP 3: Split the data into training and testing sets.

STEP 4: Build the neural network model.

STEP 5: Train the model using Adam optimizer and CrossEntropy Loss.

STEP 6: Evaluate the model using accuracy, confusion matrix, and classification report.





## PROGRAM

### Name : G.Mithik jain

### Register Number:212224240087

```

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gdown
import re

# ---------------------------------------------------------
# Name: Athul Krishna A V
# Register No: 212225240017
# ---------------------------------------------------------

# 1. Download CSV directly from Google Drive using gdown
# New link provided by user:
new_google_drive_link = '/content/customers (1).csv'

# Extract file ID from the new link
match = re.search(r'/d/([a-zA-Z0-9_-]+)', new_google_drive_link)
if match:
    file_id = match.group(1)
else:
    # Fallback to the original file_id if extraction fails
    file_id = '1gys5ahmudCTWV3SdSRGm0_jt1p8p9sqC'

output_path = 'customers.csv'

# Download the file
gdown.download(id=file_id, output=output_path, quiet=False)

# Load dataset locally
data = pd.read_csv(output_path)

# Display initial dataset info
print(data.head())
print("Columns:", data.columns)

# 2. Data Preprocessing
# Drop ID column as it's not useful for classification
data = data.drop(columns=['ID'])

# Handle missing values in Work_Experience and Family_Size
data.fillna(
    {'Work_Experience': 0, 'Family_Size': data['Family_Size'].median()},
    inplace=True,
)

# Encode categorical feature variables
categorical_columns = [
    'Gender',
    'Ever_Married',
    'Graduated',
    'Profession',
    'Spending_Score',
    'Var_1',
]
for col in categorical_columns:
  data[col] = LabelEncoder().fit_transform(data[col])

# Encode target variable (Segmentation: A, B, C, D -> 0, 1, 2, 3)
label_encoder = LabelEncoder()
data['Segmentation'] = label_encoder.fit_transform(data['Segmentation'])

# Split features and target
X = data.drop(columns=['Segmentation'])
y = data['Segmentation'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


# 3. Define Neural Network Model
class PeopleClassifier(nn.Module):

  def __init__(self, input_size):
    super(PeopleClassifier, self).__init__()
    self.fc1 = nn.Linear(input_size, 32)
    self.fc2 = nn.Linear(32, 16)
    self.fc3 = nn.Linear(16, 8)
    self.fc4 = nn.Linear(8, 4)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x


# 4. Training Function
def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# Initialize model, loss function, and optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=100)

# 5. Model Evaluation
model.eval()
predictions, actuals = [], []

with torch.no_grad():
  for X_batch, y_batch in test_loader:
    outputs = model(X_batch)
    _, predicted = torch.max(outputs, 1)
    predictions.extend(predicted.numpy())
    actuals.extend(y_batch.numpy())

# Compute evaluation metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(
    actuals,
    predictions,
    target_names=[str(i) for i in label_encoder.classes_],
)

print("\nName: Athul Krishna A V")
print("Register No: 212225240017")
print(f'Test Accuracy: {accuracy * 100:.2f}%')
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Plot Confusion Matrix
sns.heatmap(
    conf_matrix,
    annot=True,
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    fmt='g',
)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 6. Sample Prediction
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)

with torch.no_grad():
  output = model(sample_input)
  predicted_class_index = torch.argmax(output[0]).item()
  predicted_class_label = label_encoder.inverse_transform(
      [predicted_class_index]
  )[0]

actual_label = label_encoder.inverse_transform([y_test[12].item()])[0]

print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {actual_label}')

```





### Dataset Information
<img width="948" height="727" alt="image" src="https://github.com/user-attachments/assets/0279b193-edad-4d46-a2c9-ef2f0c85c094" />


### OUTPUT


## Classification Report
<img width="680" height="270" alt="image" src="https://github.com/user-attachments/assets/1b829408-3548-4965-8330-a57eda4b6a8b" />


## Confusion Matrix

<img width="725" height="657" alt="image" src="https://github.com/user-attachments/assets/ee0ff605-5db8-429b-838a-db6396c2e3a3" />




## RESULT

Thus, the neural network classification model was successfully developed, trained, and evaluated using PyTorch, and it classified the customer segments effectively.
