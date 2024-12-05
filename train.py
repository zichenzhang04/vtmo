import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from __future__ import print_function
import os

# Initialize model, optimizer, and InfoNCE loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiWayTransformerWithInfoNCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
info_nce_loss = InfoNCELoss(temperature=0.07)

# Path to save the best model
best_model_path = '/content/drive/MyDrive/best_vtmo_model.pth'

# Load the best model if it exists
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Loaded the best model from memory.")

# Training loop with model saving
num_epochs = 15
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    for concatenated_img, _ in train_loader:
        # Split the concatenated images into separate image and touch sensor parts
        img, touch_img = torch.chunk(concatenated_img, 2, dim=1)
        img, touch_img = img.to(device), touch_img.to(device)

        # Forward pass
        img_out, touch_out = model(img, touch_img)

        # Compute InfoNCE loss
        loss = info_nce_loss(img_out, touch_out)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Validation loop with model saving
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for concatenated_img, _ in val_loader:
            img, touch_img = torch.chunk(concatenated_img, 2, dim=1)
            img, touch_img = img.to(device), touch_img.to(device)

            img_out, touch_out = model(img, touch_img)
            val_loss += info_nce_loss(img_out, touch_out).item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save the model if validation loss is the best we've seen so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

# Path to the saved best model
best_model_path = '/content/drive/MyDrive/best_vtmo_model.pth'

# Initialize the model and load the best model weights if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiWayTransformerWithInfoNCELoss().to(device)

if torch.cuda.is_available():
    model.load_state_dict(torch.load(best_model_path))
else:
    model.load_state_dict(torch.load(best_model_path, map_location=device))

print("Loaded the best validation model for testing.")

# Set the model to evaluation mode
model.eval()

# Initialize counters for accuracy
correct_matches = 0
total_samples = 0
temperature = 0.07  # Use the same temperature as in training

with torch.no_grad():
    for concatenated_img, _ in test_loader:  # Assuming `test_loader` is defined
        # Split the concatenated images into separate image and touch sensor parts
        img, touch_img = torch.chunk(concatenated_img, 2, dim=1)
        img, touch_img = img.to(device), touch_img.to(device)

        # Forward pass
        img_out, touch_out = model(img, touch_img)

        # Normalize outputs to calculate cosine similarity
        img_out = F.normalize(img_out, dim=-1)
        touch_out = F.normalize(touch_out, dim=-1)

        # Calculate cosine similarity matrix (batch_size x batch_size)
        similarity_matrix = torch.matmul(img_out, touch_out.T) / temperature

        # For each image, check if the highest similarity is with the correct tactile pair
        batch_size = similarity_matrix.size(0)
        total_samples += batch_size

        # Find the indices of the maximum values along each row (prediction)
        predicted_indices = similarity_matrix.argmax(dim=1)

        # Check if each prediction is the correct index
        correct_matches += (predicted_indices ==
                            torch.arange(batch_size, device=device)).sum().item()

# Calculate accuracy
accuracy = correct_matches / total_samples
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# !pip install ptflops

# Define a wrapper for compatibility with ptflops


class MultiWayTransformerWrapper(nn.Module):
    def __init__(self, model):
        super(MultiWayTransformerWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Split the input tensor into `img` and `touch_img`
        img, touch_img = torch.chunk(x, 2, dim=1)
        return self.model(img, touch_img)


# Initialize the original model and the wrapper
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiWayTransformerWithInfoNCELoss().to(device)
model_wrapper = MultiWayTransformerWrapper(model).to(device)

# Define the input shape: concatenating two inputs of shape (3, 224, 224) along the channel dimension
input_shape = (6, 224, 224)  # Concatenate img and touch_img channels

# Calculate FLOPs and Params using the wrapper
with torch.cuda.device(0):
    flops, params = get_model_complexity_info(
        model_wrapper, input_shape, as_strings=True, print_per_layer_stat=True)

print(f"FLOPs: {flops}")
print(f"Parameters: {params}")

# Now use the same methods and loss to train a model that consists of two encoders (initialized from Beit-base).

# Define the model with two Beit-based encoders


class DualEncoderModel(nn.Module):
    def __init__(self):
        super(DualEncoderModel, self).__init__()

        # Initialize two Beit-base models as encoders
        self.img_encoder = BeitModel.from_pretrained(
            "microsoft/beit-base-patch16-224")
        self.touch_encoder = BeitModel.from_pretrained(
            "microsoft/beit-base-patch16-224")

    def forward(self, img, touch_img):
        # Get CLS token embeddings from both encoders
        img_outputs = self.img_encoder(img)
        touch_outputs = self.touch_encoder(touch_img)

        # Extract the CLS token as the final representation
        # CLS token for visual image
        img_out = img_outputs.last_hidden_state[:, 0]
        # CLS token for tactile image
        touch_out = touch_outputs.last_hidden_state[:, 0]

        return img_out, touch_out


# Initialize model, optimizer, and InfoNCE loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DualEncoderModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
info_nce_loss = InfoNCELoss(temperature=0.07)

# Path to save the best model
best_model_path = '/content/drive/MyDrive/best_dual_encoder_model.pth'

# Training loop with model saving
num_epochs = 15
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    for concatenated_img, _ in train_loader:  # Assuming train_loader is defined
        # Split the concatenated images into separate image and touch sensor parts
        img, touch_img = torch.chunk(concatenated_img, 2, dim=1)
        img, touch_img = img.to(device), touch_img.to(device)

        # Forward pass
        img_out, touch_out = model(img, touch_img)

        # Compute InfoNCE loss
        loss = info_nce_loss(img_out, touch_out)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Validation loop with model saving
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for concatenated_img, _ in val_loader:  # Assuming val_loader is defined
            img, touch_img = torch.chunk(concatenated_img, 2, dim=1)
            img, touch_img = img.to(device), touch_img.to(device)

            img_out, touch_out = model(img, touch_img)
            val_loss += info_nce_loss(img_out, touch_out).item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save the model if validation loss is the best we've seen so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")


# Initialize the model and load the best model weights if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    model.load_state_dict(torch.load(best_model_path))
else:
    model.load_state_dict(torch.load(best_model_path, map_location=device))

print("Loaded the best validation model for testing.")

# Set the model to evaluation mode
model.eval()

# Initialize counters for accuracy
correct_matches = 0
total_samples = 0
temperature = 0.07  # Use the same temperature as in training

with torch.no_grad():
    for concatenated_img, _ in test_loader:  # Assuming `test_loader` is defined
        # Split the concatenated images into separate image and touch sensor parts
        img, touch_img = torch.chunk(concatenated_img, 2, dim=1)
        img, touch_img = img.to(device), touch_img.to(device)

        # Forward pass
        img_out, touch_out = model(img, touch_img)

        # Normalize outputs to calculate cosine similarity
        img_out = F.normalize(img_out, dim=-1)
        touch_out = F.normalize(touch_out, dim=-1)

        # Calculate cosine similarity matrix (batch_size x batch_size)
        similarity_matrix = torch.matmul(img_out, touch_out.T) / temperature

        # For each image, check if the highest similarity is with the correct tactile pair
        batch_size = similarity_matrix.size(0)
        total_samples += batch_size

        # Find the indices of the maximum values along each row (prediction)
        predicted_indices = similarity_matrix.argmax(dim=1)

        # Check if each prediction is the correct index
        correct_matches += (predicted_indices ==
                            torch.arange(batch_size, device=device)).sum().item()

# Calculate accuracy
accuracy = correct_matches / total_samples
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Based on the model in baseline
# Initialize the DualEncoderModel and the wrapper
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DualEncoderModel().to(device)
model_wrapper = DualEncoderWrapper(model).to(device)

# Define the input shape: concatenating two inputs of shape (3, 224, 224) along the channel dimension
input_shape = (6, 224, 224)  # Concatenate img and touch_img channels

# Calculate FLOPs and Params using the wrapper
with torch.cuda.device(0):
    flops, params = get_model_complexity_info(
        model_wrapper, input_shape, as_strings=True, print_per_layer_stat=True)

print(f"FLOPs: {flops}")
