#!/usr/bin/env python3
"""
GradCAM Implementation for Medical Image Explainability
=========================================================
Implements Gradient-weighted Class Activation Mapping (GradCAM) for 
explaining deep learning predictions on medical images (chest X-rays).

GradCAM shows which regions of an image the model focuses on when making predictions.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import requests
from io import BytesIO
from PIL import Image

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input

print("=" * 80)
print("🔬 GradCAM for Medical Image Explainability")
print("=" * 80)
print("\nGradient-weighted Class Activation Mapping (GradCAM)")
print("Shows which parts of an X-ray image the AI model looks at")
print()

# ============================================================================
# PART 1: GradCAM Implementation
# ============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM)
    
    Produces visual explanations for CNN predictions by highlighting
    important regions in the input image.
    
    Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (2017)
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize GradCAM explainer.
        
        Args:
            model: Keras/TensorFlow model
            layer_name: Name of the convolutional layer to visualize
                       (defaults to last conv layer)
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
        # Create gradient model
        self.grad_model = keras.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        raise ValueError("No convolutional layer found in model")
    
    def compute_heatmap(self, img_array, class_idx=None, eps=1e-8):
        """
        Compute GradCAM heatmap for a given image.
        
        Args:
            img_array: Preprocessed image array
            class_idx: Index of class to visualize (if None, uses predicted class)
            eps: Small epsilon for numerical stability
            
        Returns:
            heatmap: GradCAM heatmap (2D array)
            prediction: Model prediction
        """
        # Compute gradients
        with tf.GradientTape() as tape:
            # Get convolutional outputs and predictions
            conv_outputs, predictions = self.grad_model(img_array)
            
            # Get the class index if not provided
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the prediction score for the target class
            class_channel = predictions[:, class_idx]
        
        # Compute gradients of the class score with respect to the conv output
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv output channels by importance
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap by averaging weighted channels
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= (np.max(heatmap) + eps)  # Normalize to [0, 1]
        
        return heatmap, predictions.numpy()
    
    def overlay_heatmap(self, heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay GradCAM heatmap on original image.
        
        Args:
            heatmap: GradCAM heatmap
            original_img: Original image (PIL Image or numpy array)
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap for heatmap
            
        Returns:
            Superimposed image with heatmap overlay
        """
        # Convert original image to numpy array if needed
        if isinstance(original_img, Image.Image):
            original_img = np.array(original_img)
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Convert heatmap to RGB using colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is in correct format
        if len(original_img.shape) == 2:  # Grayscale
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        
        # Normalize original image to [0, 255]
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(
            original_img.astype(np.uint8),
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        return superimposed

# ============================================================================
# PART 2: Load or Create Medical Imaging Model
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: Loading Medical Imaging Model")
print("=" * 80)

print("\n🔧 Setting up DenseNet121 (pre-trained on ImageNet)")
print("   Note: For production, use a model trained on chest X-rays")

# Create a simple model (DenseNet121 - commonly used for medical imaging)
base_model = DenseNet121(
    weights='imagenet',
    include_top=True,
    input_shape=(224, 224, 3)
)

print(f"✅ Model loaded: {base_model.name}")
print(f"   Total layers: {len(base_model.layers)}")
print(f"   Input shape: {base_model.input_shape}")
print(f"   Output classes: {base_model.output_shape[-1]}")

# For demonstration, we'll use the pre-trained model
# In practice, you'd fine-tune this on chest X-ray dataset
model = base_model

# ============================================================================
# PART 3: Download Sample Medical Images
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: Preparing Sample Medical Images")
print("=" * 80)

# Create directory for sample images
images_dir = Path("sample_medical_images")
images_dir.mkdir(exist_ok=True)

print(f"\n📁 Sample images directory: {images_dir}")

# For this demo, we'll create a synthetic chest X-ray-like image
# In production, you'd use real MIMIC-CXR images
def create_sample_xray(filename, width=224, height=224):
    """Create a synthetic chest X-ray-like image for demonstration."""
    # Create a grayscale image with lung-like structures
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Add gradient background (chest cavity)
    for i in range(height):
        img[i, :] = int(30 + (i / height) * 60)
    
    # Add lung-like regions (darker areas)
    center_y, center_x = height // 2, width // 2
    
    # Left lung
    cv2.ellipse(img, (center_x - 40, center_y), (60, 80), 0, 0, 360, 80, -1)
    
    # Right lung
    cv2.ellipse(img, (center_x + 40, center_y), (60, 80), 0, 0, 360, 80, -1)
    
    # Add some noise and texture
    noise = np.random.normal(0, 10, (height, width))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Add a potential abnormality (bright spot - simulating infiltrate)
    cv2.circle(img, (center_x + 30, center_y - 20), 15, 200, -1)
    cv2.circle(img, (center_x + 30, center_y - 20), 15, 150, 3)
    
    # Save image
    cv2.imwrite(str(filename), img)
    return img

# Create sample images
print("\n📸 Creating synthetic chest X-ray samples...")
sample_normal = create_sample_xray(images_dir / "chest_xray_normal.png")
print(f"   ✅ Created: chest_xray_normal.png")

# Create one with more prominent abnormality
sample_abnormal = create_sample_xray(images_dir / "chest_xray_pneumonia.png")
# Add additional infiltrate
center_y, center_x = sample_abnormal.shape[0] // 2, sample_abnormal.shape[1] // 2
cv2.circle(sample_abnormal, (center_x - 25, center_y + 30), 20, 180, -1)
cv2.imwrite(str(images_dir / "chest_xray_pneumonia.png"), sample_abnormal)
print(f"   ✅ Created: chest_xray_pneumonia.png")

# ============================================================================
# PART 4: Apply GradCAM to Medical Images
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: Applying GradCAM to Chest X-rays")
print("=" * 80)

# Initialize GradCAM
print("\n🔧 Initializing GradCAM explainer...")
gradcam = GradCAM(model, layer_name='conv5_block16_concat')
print(f"✅ Using layer: {gradcam.layer_name}")

# Process images
sample_files = list(images_dir.glob("*.png"))

for img_path in sample_files:
    print(f"\n{'=' * 80}")
    print(f"📊 Analyzing: {img_path.name}")
    print(f"{'=' * 80}")
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Get predictions
    predictions = model.predict(img_array, verbose=0)
    top_pred_idx = np.argmax(predictions[0])
    top_pred_score = predictions[0][top_pred_idx]
    
    print(f"\n🎯 Model Prediction:")
    print(f"   Class Index: {top_pred_idx}")
    print(f"   Confidence: {top_pred_score:.1%}")
    
    # Compute GradCAM
    print(f"\n🔥 Computing GradCAM heatmap...")
    heatmap, _ = gradcam.compute_heatmap(img_array, class_idx=top_pred_idx)
    
    # Load original image for overlay
    original_img = cv2.imread(str(img_path))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    superimposed = gradcam.overlay_heatmap(heatmap, original_img, alpha=0.5)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Chest X-ray', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('GradCAM Heatmap\n(Model Attention)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(superimposed)
    axes[2].set_title(f'GradCAM Overlay\nConfidence: {top_pred_score:.1%}', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'GradCAM Visualization - {img_path.stem}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save visualization
    output_path = f"gradcam_{img_path.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization: {output_path}")
    
    # Analyze heatmap statistics
    hot_spots = np.percentile(heatmap, 90)
    high_attention_percentage = (heatmap > hot_spots).sum() / heatmap.size * 100
    
    print(f"\n📈 Heatmap Analysis:")
    print(f"   Max attention: {heatmap.max():.3f}")
    print(f"   Mean attention: {heatmap.mean():.3f}")
    print(f"   High attention areas: {high_attention_percentage:.1f}% of image")

# ============================================================================
# PART 5: Multi-scale GradCAM Visualization
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: Advanced Multi-scale GradCAM")
print("=" * 80)

def compare_gradcam_layers(model, img_array, original_img, layers_to_compare):
    """Compare GradCAM visualizations from different layers."""
    n_layers = len(layers_to_compare)
    fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 10))
    
    if n_layers == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, layer_name in enumerate(layers_to_compare):
        try:
            # Create GradCAM for this layer
            gc = GradCAM(model, layer_name=layer_name)
            heatmap, predictions = gc.compute_heatmap(img_array)
            superimposed = gc.overlay_heatmap(heatmap, original_img, alpha=0.5)
            
            # Plot heatmap
            axes[0, idx].imshow(heatmap, cmap='jet')
            axes[0, idx].set_title(f'{layer_name}\n(Heatmap)', fontsize=10)
            axes[0, idx].axis('off')
            
            # Plot overlay
            axes[1, idx].imshow(superimposed)
            axes[1, idx].set_title(f'{layer_name}\n(Overlay)', fontsize=10)
            axes[1, idx].axis('off')
            
        except Exception as e:
            axes[0, idx].text(0.5, 0.5, f'Error:\n{layer_name}', 
                            ha='center', va='center')
            axes[0, idx].axis('off')
            axes[1, idx].axis('off')
    
    plt.suptitle('Multi-Scale GradCAM Analysis\n(Different Network Layers)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# Find multiple conv layers for comparison
print("\n🔍 Finding convolutional layers for multi-scale analysis...")
conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
selected_layers = conv_layers[-3:] if len(conv_layers) >= 3 else conv_layers

print(f"   Selected layers: {selected_layers}")

# Apply to first sample
if sample_files:
    img_path = sample_files[0]
    print(f"\n📊 Applying multi-scale GradCAM to: {img_path.name}")
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    original_img = cv2.imread(str(img_path))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    fig = compare_gradcam_layers(model, img_array, original_img, selected_layers)
    
    output_path = "gradcam_multiscale_analysis.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("📁 GENERATED FILES")
print("=" * 80)

print("\n✅ GradCAM Visualizations:")
for img_path in sample_files:
    print(f"   - gradcam_{img_path.stem}.png")

print(f"\n✅ Multi-scale Analysis:")
print(f"   - gradcam_multiscale_analysis.png")

print("\n✅ Sample Images:")
for img_path in sample_files:
    print(f"   - {img_path}")

print("\n" + "=" * 80)
print("🎓 GRADCAM EXPLANATION")
print("=" * 80)

print("""
What is GradCAM?
----------------
Gradient-weighted Class Activation Mapping (GradCAM) is a technique for
producing visual explanations for decisions from CNN-based models.

How it works:
1. Forward pass: Image → CNN → Prediction
2. Backward pass: Compute gradients of prediction w.r.t. conv layer
3. Weight importance: Average gradients across spatial dimensions
4. Create heatmap: Weighted combination of feature maps
5. Overlay: Superimpose heatmap on original image

What the colors mean:
🔴 RED/HOT     = High importance (model looks here)
🔵 BLUE/COLD   = Low importance (model ignores)
🟡 YELLOW      = Medium importance

Clinical Usage:
- Verify the model looks at relevant anatomical regions
- Identify potential biases (e.g., focusing on artifacts)
- Build trust with clinicians by showing reasoning
- Detect failure cases (wrong region highlighted)

For Medical Imaging:
- Helps radiologists understand AI decisions
- Can reveal when AI uses spurious correlations
- Important for FDA approval and clinical deployment
- Enables collaborative human-AI diagnosis
""")

print("\n" + "=" * 80)
print("🔬 NEXT STEPS FOR MIMIC-CXR")
print("=" * 80)

print("""
To use GradCAM with real MIMIC-CXR chest X-rays:

1. Download MIMIC-CXR dataset from PhysioNet:
   https://physionet.org/content/mimic-cxr/2.0.0/
   
2. Train a disease classification model:
   - Pneumonia detection
   - Cardiomegaly detection
   - Pleural effusion detection
   - Multi-label classification
   
3. Apply GradCAM to explain predictions:
   - Which lung regions show abnormalities
   - Confirm model doesn't use artifacts
   - Generate reports for clinicians

4. Integrate with this system:
   - Combine with SHAP/LIME for tabular data
   - Multi-modal explainability (images + vitals)
   - Complete diagnostic dashboard
""")

print("\n✅ GradCAM demonstration complete!\n")
