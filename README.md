# IMPROVING WASTE IMAGE CLASSIFICATION PERFORMANCE OF EFFICIENT DL MODELS

## Dataset
[Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)
- Collection of 15,000 images (each 256x256 pixels) depicting various recyclable materials, general waste, and household items across 30 distinct categories.
- 500 images per category and 250 images per subcategory.

## Initial Baseline Implementation & Accuracy
1. EfficientNetV2 : **80%**
2. PyramidNet + ASAM : 49%
3. MobileViT : **84%**
4. Swin Transformer : 78%

## Main Proposed Method 01 - EfficientNetV2-S
### Changes
- Replaced SE(Squeeze and Excitation) module in MBConv block with ECA (Efficient Channel Attention) module.
- Added ECA module in Fused-MBConv block.
- Replaced SiLU activation with Leaky ReLU.

### Training Techniques
- Data Augmentation (for real-world scenario) : Random Resized Crop, Random Horizontal Flip

### Result 
- Overfitting Alleviation : Validation loss **1.15 → 0.59**
- Accuracy Increment : **80.5% → 88.4%**
- Faster Inference Speed : **15.86 → 6.29(ms)** / NVIDIA T4 GPU

## Main Proposed Method 02 - MobileViT
### Changes
- Replace Self-Attention(SA) layer with Inverted Residuals(IR) block (stacked MV2 blocks)
- Linear Bottleneck layers (w/ ReLU6, & exclude at final layer)

### Training Techniques
- Data Augmentation : Random Resized Crop, Random Horizontal Flip
- Weight Decay (L2 Regularization)
- Learning Rate Scheduler (to find better local optima)
- Weight Initialization (to prevent vanishing/exploding gradients)
  - CNN layers : He Init.
  - Linear layers : Normal Init.

### Result
- Accuracy Increment : **84.4% → 90.2%**
- Faster Inference Speed : **4.97 → 4.27(ms)** / NVIDIA T4 GPU
- Overfitting alleviation : Shortened gap between train/val loss

## Other Proposed Methods
### EfficientNetV2-S
- Replaced SE module with GLU (Gated Linear Unit) module.
  - Overfitting Alleviation, 2 Times Faster Inference Speed at CPU and GPU.
- Used ImageNet-1K pre-trained weights and added Attention module after feature extractor.
  - 5 Times Faster Inference Speed at GPU.

### MobileViT
- Added 5x5 Conv layer in Local Representation of MobileViT Block.
  - 2.4 Times Faster Inference Speed at CPU.
