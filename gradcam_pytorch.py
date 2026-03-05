#!/usr/bin/env python3
"""
GradCAM Implementation for Medical Image Explainability (PyTorch)
===================================================================
Implements Gradient-weighted Class Activation Mapping (GradCAM) for 
explaining deep learning predictions on medical images using PyTorch.

GradCAM shows which regions of an image the model focuses on when making predictions.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

print("=" * 80)
print("🔬 GradCAM for Medical Image Explainability")
print("=" * 80)
print("\nGradient-weighted Class Activation Mapping (GradCAM)")
print("Shows which parts of an X-ray image the AI model looks at")
print()

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    print(f"✅ PyTorch version: {torch.__version__}")
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available, using simplified visualization")
    PYTORCH_AVAILABLE = False

# ============================================================================
# PART 1: GradCAM Implementation
# ============================================================================

if PYTORCH_AVAILABLE:
    
    class GradCAM:
        """
        Gradient-weighted Class Activation Mapping (GradCAM) for PyTorch models.
        
        Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization" (2017)
        """
        
        def __init__(self, model, target_layer):
            """
            Initialize GradCAM explainer.
            
            Args:
                model: PyTorch model
                target_layer: Target convolutional layer for visualization
            """
            self.model = model
            self.target_layer = target_layer
            
            # Hooks for gradient and activation extraction
            self.gradients = None
            self.activations = None
            
            # Register hooks
            self._register_hooks()
        
        def _register_hooks(self):
            """Register forward and backward hooks."""
            
            def forward_hook(module, input, output):
                self.activations = output
            
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0]
            
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)
        
        def generate_heatmap(self, input_tensor, class_idx=None):
            """
            Generate GradCAM heatmap for input image.
            
            Args:
                input_tensor: Input image tensor
                class_idx: Target class index (if None, uses predicted class)
                
            Returns:
                heatmap: GradCAM heatmap (2D numpy array)
                prediction: Model prediction probabilities
            """
            self.model.eval()
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Get predicted class if not specified
            if class_idx is None:
                class_idx = output.argmax(dim=1)
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass for target class
            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            
            # Get gradients and activations
            gradients = self.gradients.cpu().data.numpy()[0]
            activations = self.activations.cpu().data.numpy()[0]
            
            # Calculate weights (global average pooling of gradients)
            weights = np.mean(gradients, axis=(1, 2))
            
            # Weighted combination of activation maps
            heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                heatmap += w * activations[i]
            
            # Apply ReLU and normalize
            heatmap = np.maximum(heatmap, 0)
            heatmap = heatmap / (heatmap.max() + 1e-8)
            
            return heatmap, output.detach().cpu().numpy()
        
        @staticmethod
        def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
            """
            Overlay heatmap on original image.
            
            Args:
                heatmap: GradCAM heatmap
                original_img: Original image (numpy array or PIL Image)
                alpha: Overlay transparency
                colormap: OpenCV colormap
                
            Returns:
                Superimposed image
            """
            # Convert PIL to numpy if needed
            if isinstance(original_img, Image.Image):
                original_img = np.array(original_img)
            
            # Resize heatmap to match original image
            heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            
            # Convert heatmap to RGB
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized),
                colormap
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Handle grayscale images
            if len(original_img.shape) == 2:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            
            # Normalize to [0, 255]
            if original_img.max() <= 1.0:
                original_img = (original_img * 255).astype(np.uint8)
            
            # Blend images
            superimposed = cv2.addWeighted(
                original_img.astype(np.uint8),
                1 - alpha,
                heatmap_colored,
                alpha,
                0
            )
            
            return superimposed

# ============================================================================
# PART 2: Create Sample Medical Images
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: Creating Sample Chest X-ray Images")
print("=" * 80)

# Create directory for sample images
images_dir = Path("sample_medical_images")
images_dir.mkdir(exist_ok=True)

print(f"\n📁 Directory: {images_dir}")

def create_synthetic_chest_xray(filename, width=224, height=224, add_pathology=False):
    """Create a synthetic chest X-ray image for demonstration."""
    # Create base image
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Gradient background (chest cavity)
    for i in range(height):
        img[i, :] = int(30 + (i / height) * 60)
    
    # Lung regions (darker ellipses)
    center_y, center_x = height // 2, width // 2
    
    # Left lung
    cv2.ellipse(img, (center_x - 40, center_y), (50, 70), 0, 0, 360, 75, -1)
    
    # Right lung
    cv2.ellipse(img, (center_x + 40, center_y), (50, 70), 0, 0, 360, 75, -1)
    
    # Heart shadow (brighter region)
    cv2.ellipse(img, (center_x - 5, center_y + 10), (35, 40), 20, 0, 360, 110, -1)
    
    # Ribcage texture
    for i in range(5):
        y = center_y - 60 + i * 25
        cv2.line(img, (20, y), (width - 20, y), int(np.random.uniform(80, 100)), 2)
    
    # Add realistic noise
    noise = np.random.normal(0, 8, (height, width))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Gaussian blur for smoother appearance
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    if add_pathology:
        # Add pathological findings
        # Infiltrate (pneumonia-like)
        cv2.circle(img, (center_x + 35, center_y - 15), 18, 160, -1)
        cv2.circle(img, (center_x + 35, center_y - 15), 18, 140, 2)
        
        # Additional opacity
        cv2.ellipse(img, (center_x - 30, center_y + 25), (22, 15), 30, 0, 360, 130, -1)
    
    # Save as grayscale
    cv2.imwrite(str(filename), img)
    
    # Also save as RGB for model input
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return img, img_rgb

print("\n📸 Creating synthetic chest X-ray images...")

# Create normal X-ray
normal_img_gray, normal_img_rgb = create_synthetic_chest_xray(
    images_dir / "chest_xray_normal.png",
    add_pathology=False
)
print(f"   ✅ Created: chest_xray_normal.png")

# Create abnormal X-ray (with pathology)
abnormal_img_gray, abnormal_img_rgb = create_synthetic_chest_xray(
    images_dir / "chest_xray_pneumonia.png",
    add_pathology=True
)
print(f"   ✅ Created: chest_xray_pneumonia.png")

# ============================================================================
# PART 3: Load Model and Apply GradCAM
# ============================================================================

if PYTORCH_AVAILABLE:
    print("\n" + "=" * 80)
    print("PART 2: Loading Medical Imaging Model")
    print("=" * 80)
    
    print("\n🔧 Loading ResNet50 (pre-trained on ImageNet)")
    print("   Note: For production, use a model trained on chest X-rays")
    
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    model.eval()
    
    print(f"✅ Model loaded: ResNet50")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get target layer (last conv layer)
    target_layer = model.layer4[-1].conv3
    print(f"   Target layer: layer4[-1].conv3")
    
    # Initialize GradCAM
    print("\n🔥 Initializing GradCAM...")
    gradcam = GradCAM(model, target_layer)
    print("✅ GradCAM ready")
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # =========================================================================
    # PART 4: Generate GradCAM Visualizations
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("PART 3: Generating GradCAM Visualizations")
    print("=" * 80)
    
    sample_files = [
        images_dir / "chest_xray_normal.png",
        images_dir / "chest_xray_pneumonia.png"
    ]
    
    for img_path in sample_files:
        print(f"\n{'=' * 80}")
        print(f"📊 Analyzing: {img_path.name}")
        print(f"{'=' * 80}")
        
        # Load and preprocess image
        pil_img = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(pil_img)
        input_batch = input_tensor.unsqueeze(0)
        
        # Generate GradCAM heatmap
        print("\n🔥 Computing GradCAM heatmap...")
        heatmap, predictions = gradcam.generate_heatmap(input_batch)
        
        # Get top prediction
        pred_idx = predictions.argmax()
        pred_score = predictions[0, pred_idx]
        
        print(f"✅ Heatmap generated")
        print(f"\n🎯 Model Prediction:")
        print(f"   Class Index: {pred_idx}")
        print(f"   Confidence: {pred_score:.1%}")
        
        # Load original image for visualization
        original_img = cv2.imread(str(img_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        superimposed = gradcam.overlay_heatmap(heatmap, original_img, alpha=0.5)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Chest X-ray', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('GradCAM Heatmap\n(Model Attention)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(superimposed)
        axes[2].set_title(f'GradCAM Overlay\nConfidence: {pred_score:.1%}', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'GradCAM Visualization - {img_path.stem}', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        output_path = f"gradcam_{img_path.stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
        
        # Heatmap statistics
        print(f"\n📈 Heatmap Analysis:")
        print(f"   Max attention: {heatmap.max():.3f}")
        print(f"   Mean attention: {heatmap.mean():.3f}")
        print(f"   Std attention: {heatmap.std():.3f}")
        
        # Find regions of high attention
        threshold = np.percentile(heatmap, 90)
        high_attention_mask = heatmap > threshold
        print(f"   High attention regions: {high_attention_mask.sum() / heatmap.size * 100:.1f}% of image")

else:
    # Fallback: Create simple attention visualization without PyTorch
    print("\n" + "=" * 80)
    print("PART 2: Creating Simulated Attention Maps")
    print("=" * 80)
    
    print("\n⚠️  PyTorch not available - creating simulated visualizations")
    
    sample_files = [
        images_dir / "chest_xray_normal.png",
        images_dir / "chest_xray_pneumonia.png"
    ]
    
    for img_path in sample_files:
        print(f"\n📊 Processing: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create simulated attention map (gradient-based)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection as proxy for attention
        edges = cv2.Canny(gray, 50, 150)
        edges_blur = cv2.GaussianBlur(edges.astype(float), (21, 21), 0)
        
        # Normalize
        heatmap = edges_blur / (edges_blur.max() + 1e-8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original X-ray', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Simulated Attention Map', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        axes[2].imshow(superimposed)
        axes[2].set_title('Attention Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Simulated Attention - {img_path.stem}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = f"gradcam_simulated_{img_path.stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")

# ============================================================================
# Summary and Documentation
# ============================================================================

print("\n" + "=" * 80)
print("📁 GENERATED FILES")
print("=" * 80)

print("\n✅ Sample Images:")
print(f"   - {images_dir}/chest_xray_normal.png")
print(f"   - {images_dir}/chest_xray_pneumonia.png")

print("\n✅ GradCAM Visualizations:")
if PYTORCH_AVAILABLE:
    print("   - gradcam_chest_xray_normal.png")
    print("   - gradcam_chest_xray_pneumonia.png")
else:
    print("   - gradcam_simulated_chest_xray_normal.png")
    print("   - gradcam_simulated_chest_xray_pneumonia.png")

print("\n" + "=" * 80)
print("🎓 UNDERSTANDING GRADCAM")
print("=" * 80)

print("""
What is GradCAM?
----------------
Gradient-weighted Class Activation Mapping visualizes which parts of an
image a deep learning model focuses on when making predictions.

How it works:
1. Forward pass: Image → CNN → Prediction
2. Backward pass: Compute gradients of prediction w.r.t. convolutional layer
3. Weight the activation maps by their importance (gradient-based)
4. Create heatmap showing important regions
5. Overlay heatmap on original image

Color Interpretation:
🔴 RED/HOT areas   = High importance (model focuses here)
🔵 BLUE/COLD areas = Low importance (model ignores)
🟡 YELLOW areas    = Medium importance

Medical Imaging Applications:
✓ Pneumonia detection in chest X-rays
✓ Brain tumor localization in MRI
✓ Diabetic retinopathy in fundus images
✓ COVID-19 detection in CT scans
✓ Skin cancer classification

Benefits for Clinicians:
1. Transparency: Shows what the AI "sees"
2. Trust: Verifies model looks at correct anatomy
3. Error detection: Identifies when model uses artifacts
4. Education: Helps understand AI reasoning
5. Regulatory: Required for FDA approval

Limitations:
⚠️  Shows correlation, not causation
⚠️  May not capture all relevant features
⚠️  Different layers give different insights
⚠️  Requires careful interpretation
""")

print("\n" + "=" * 80)
print("🔬 INTEGRATION WITH MIMIC DATA")
print("=" * 80)

print("""
Combining GradCAM with MIMIC Clinical Data:
--------------------------------------------

Current System:
• SHAP/LIME: Explains tabular predictions (vital signs, labs)
• GradCAM: Explains image-based predictions (X-rays, CT scans)

For Complete Explainability:
1. MIMIC-III (Clinical data) → SHAP/LIME explanations
   - Risk scores from vitals and labs
   - Feature importance for sepsis, mortality, etc.

2. MIMIC-CXR (Chest X-rays) → GradCAM explanations
   - Visual localization of pathology
   - Pneumonia, edema, cardiomegaly detection

3. Multi-modal Fusion:
   - Combine image + tabular features
   - Joint model with multiple explanation methods
   - More accurate and interpretable predictions

To Get MIMIC-CXR:
1. Register at PhysioNet: https://physionet.org/
2. Complete CITI training
3. Download MIMIC-CXR dataset
4. Link with MIMIC-III patient IDs
5. Train chest X-ray classification models
6. Apply GradCAM to explain predictions

Example Integration:
Patient presents with:
• Fever, elevated WBC (MIMIC-III) → SHAP shows infection risk
• Chest X-ray (MIMIC-CXR) → GradCAM highlights lung infiltrate
• Combined: High confidence pneumonia diagnosis with explanations
""")

print("\n✅ GradCAM implementation complete!\n")
