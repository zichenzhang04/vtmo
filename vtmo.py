from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from __future__ import print_function
import numpy as np
from skimage import color
from PIL import Image
import torchvision.datasets as datasets
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split
from transformers import BeitModel
from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))

        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_vlffn=False,
        layer_scale_init_values=0.1,
        max_text_len=40,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_touch = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_touch = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_imag = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_vt = None
        if with_vlffn:
            self.mlp_vt = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.norm2_vt = norm_layer(dim)

        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        self.max_text_len = max_text_len

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x),
                    mask=mask, relative_position_bias=relative_position_bias))

        if modality_type == "image":
            x = x + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x)))
        elif modality_type == "touch":
            x = x + self.drop_path(self.gamma_2 * self.mlp_touch(self.norm2_touch(x)))
        else:
            if self.mlp_vl is None:
                x_touch = x[:, : self.max_text_len]
                x_imag = x[:, self.max_text_len :]
                x_touch = x_touch + self.drop_path(self.gamma_2 * self.mlp_touch(self.norm2_touch(x_touch)))
                x_imag = x_imag + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag)))
                x = torch.cat([x_touch, x_imag], dim=1)
            else:
                x = x + self.drop_path(self.gamma_2 * self.mlp_vt(self.norm2_vt(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        return x


class MultiWayTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        need_relative_position_embed=True,
        use_abs_pos_emb=False,
        layer_scale_init_values=0.1,
        vlffn_start_layer_index=10,
        config=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            need_relative_position_embed (bool): enable relative position bias on self-attention
            use_abs_pos_emb (bool): enable abs pos emb
            layer_scale_init_values (float or None): layer scale init values, set None to disable
            vlffn_start_layer_index (int): vl-ffn start index
            config: (dict): other hyper from pytorch-lighting
        """
        super().__init__()
        self.use_abs_pos_emb = use_abs_pos_emb
        self.need_relative_position_embed = need_relative_position_embed

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.vlffn_start_layer_index = vlffn_start_layer_index
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if self.use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_vlffn=(i >= self.vlffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def visual_embed(self, _x):
        x = self.patch_embed(_x)
        x = x.flatten(2).transpose(1, 2)
        B, L, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        x_mask = torch.ones(x.shape[0], x.shape[1])

        return x, x_mask

class TouchFolderContrastive(Dataset):
    """Dataset for contrastive learning, pairing video and touch sensor frames."""

    def __init__(self, root, transform=None, split='train'):
        self.dataroot = root
        self.transform = transform
        self.split = split
        self.pairs = []

        # Collect all pairs from the six main folders
        for folder in os.listdir(root):
            video_folder = os.path.join(root, folder, 'video_frame')
            gelsight_folder = os.path.join(root, folder, 'gelsight_frame')

            if os.path.exists(video_folder) and os.path.exists(gelsight_folder):
                for file_name in os.listdir(video_folder):
                    video_path = os.path.join(video_folder, file_name)
                    gel_path = os.path.join(gelsight_folder, file_name)
                    if os.path.exists(gel_path):
                        self.pairs.append((video_path, gel_path))

        # Split dataset into train, validation, and test
        train_size = int(0.7 * len(self.pairs))
        val_size = int(0.15 * len(self.pairs))
        test_size = len(self.pairs) - train_size - val_size

        self.train_data, self.val_data, self.test_data = random_split(
            self.pairs, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Select appropriate data split
        if self.split == 'train':
            self.data = self.train_data
        elif self.split == 'val':
            self.data = self.val_data
        elif self.split == 'test':
            self.data = self.test_data
        else:
            raise ValueError(f"Invalid split name: {split}")

    def __getitem__(self, index):
        """Returns a contrastive pair."""
        video_path, gel_path = self.data[index]

        video_img = Image.open(video_path).convert('RGB')
        gel_img = Image.open(gel_path).convert('RGB')

        if self.transform is not None:
            video_img = self.transform(video_img)
            gel_img = self.transform(gel_img)

        # Concatenate the video and gel images as positive pair
        out = torch.cat((video_img, gel_img), dim=0)

        return out, index  # index for tracking in contrastive loss

    def __len__(self):
        return len(self.data)

def get_contrastive_loader(batch_size=32):
    data_folder = '/content/drive/MyDrive/touch_and_go_copy/'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = TouchFolderContrastive(data_folder, transform=transform, split='train')
    val_dataset = TouchFolderContrastive(data_folder, transform=transform, split='val')
    test_dataset = TouchFolderContrastive(data_folder, transform=transform, split='test')

    print(f'Number of train samples: {len(train_dataset)}')
    print(f'Number of val samples: {len(val_dataset)}')
    print(f'Number of test samples: {len(test_dataset)}')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=4, prefetch_factor=2)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=4, prefetch_factor=2)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=4, prefetch_factor=2)

    return train_loader, val_loader, test_loader

# Get the train and validation loaders
train_loader, val_loader, test_loader = get_contrastive_loader()
# Define InfoNCE Loss
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, img_out, touch_out):
        # Normalize outputs to calculate cosine similarity
        img_out = F.normalize(img_out, dim=-1)
        touch_out = F.normalize(touch_out, dim=-1)

        # Cosine similarity matrix
        logits = torch.matmul(img_out, touch_out.T) / self.temperature

        # Diagonal elements are positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Apply cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss

# Define the model with InfoNCE loss support
class MultiWayTransformerWithInfoNCELoss(MultiWayTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load Beit-base model
        beit_base = BeitModel.from_pretrained("microsoft/beit-base-patch16-224")
        beit_state_dict = beit_base.state_dict()

        # Copy weights for the fully connected networks in each block
        for layer_idx, block in enumerate(self.blocks):
            # Access corresponding FC weights from Beit-base
            beit_fc1_weight = beit_state_dict[f"encoder.layer.{layer_idx}.intermediate.dense.weight"]
            beit_fc1_bias = beit_state_dict[f"encoder.layer.{layer_idx}.intermediate.dense.bias"]
            beit_fc2_weight = beit_state_dict[f"encoder.layer.{layer_idx}.output.dense.weight"]
            beit_fc2_bias = beit_state_dict[f"encoder.layer.{layer_idx}.output.dense.bias"]

            # Assign the same weights to the image, touch, and (if exists) vl FC layers in your model
            for fc_network in [block.mlp_imag, block.mlp_touch, block.mlp_vt if block.mlp_vt else block.mlp_imag]:
                fc_network.fc1.weight.data.copy_(beit_fc1_weight)
                fc_network.fc1.bias.data.copy_(beit_fc1_bias)
                fc_network.fc2.weight.data.copy_(beit_fc2_weight)
                fc_network.fc2.bias.data.copy_(beit_fc2_bias)

        # # Freeze the attention layer and mlp_imag in each block
        # for block in self.blocks:
        #     # Freeze attention layer parameters
        #     for param in block.attn.parameters():
        #         param.requires_grad = False

        #     # Freeze mlp_imag parameters
        #     for param in block.mlp_imag.parameters():
        #         param.requires_grad = False

    def forward(self, img, touch_img):
        img_emb, _ = self.visual_embed(img)
        touch_emb, _ = self.visual_embed(touch_img)

        for block in self.blocks:
            img_emb = block(img_emb, modality_type="image")
            touch_emb = block(touch_emb, modality_type="touch")

        img_out = self.norm(img_emb[:, 0])
        touch_out = self.norm(touch_emb[:, 0])

        return img_out, touch_out

# Initialize model, optimizer, and InfoNCE loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiWayTransformerWithInfoNCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
info_nce_loss = InfoNCELoss(temperature=0.07)

# Path to save the best model
best_model_path = '/content/drive/MyDrive/best_vtmo_model.pth'

# Initialize model, optimizer, and InfoNCE loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiWayTransformerWithInfoNCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
info_nce_loss = InfoNCELoss(temperature=0.07)

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
        correct_matches += (predicted_indices == torch.arange(batch_size, device=device)).sum().item()

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
    flops, params = get_model_complexity_info(model_wrapper, input_shape, as_strings=True, print_per_layer_stat=True)

print(f"FLOPs: {flops}")
print(f"Parameters: {params}")

# Now use the same methods and loss to train a model that consists of two encoders (initialized from Beit-base).

# Define the model with two Beit-based encoders
class DualEncoderModel(nn.Module):
    def __init__(self):
        super(DualEncoderModel, self).__init__()

        # Initialize two Beit-base models as encoders
        self.img_encoder = BeitModel.from_pretrained("microsoft/beit-base-patch16-224")
        self.touch_encoder = BeitModel.from_pretrained("microsoft/beit-base-patch16-224")

    def forward(self, img, touch_img):
        # Get CLS token embeddings from both encoders
        img_outputs = self.img_encoder(img)
        touch_outputs = self.touch_encoder(touch_img)

        # Extract the CLS token as the final representation
        img_out = img_outputs.last_hidden_state[:, 0]  # CLS token for visual image
        touch_out = touch_outputs.last_hidden_state[:, 0]  # CLS token for tactile image

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
        correct_matches += (predicted_indices == torch.arange(batch_size, device=device)).sum().item()

# Calculate accuracy
accuracy = correct_matches / total_samples
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Define a wrapper for compatibility with ptflops
class DualEncoderWrapper(nn.Module):
    def __init__(self, model):
        super(DualEncoderWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Split the input tensor into `img` and `touch_img`
        img, touch_img = torch.chunk(x, 2, dim=1)
        return self.model(img, touch_img)

# Initialize the DualEncoderModel and the wrapper
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DualEncoderModel().to(device)
model_wrapper = DualEncoderWrapper(model).to(device)

# Define the input shape: concatenating two inputs of shape (3, 224, 224) along the channel dimension
input_shape = (6, 224, 224)  # Concatenate img and touch_img channels

# Calculate FLOPs and Params using the wrapper
with torch.cuda.device(0):
    flops, params = get_model_complexity_info(model_wrapper, input_shape, as_strings=True, print_per_layer_stat=True)

print(f"FLOPs: {flops}")