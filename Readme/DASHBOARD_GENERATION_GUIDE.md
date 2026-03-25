# Dashboard Figure Generation - Quick Reference Guide

## ✅ What Was Generated

Two dashboard figures have been created:

1. **`dashboard_publication_figure.png`** (300 DPI)
   - High-resolution for IEEE journal/conference submission
   - File size: ~3-5 MB
   - Print quality: Excellent
   - **USE THIS for your paper**

2. **`dashboard_preview.png`** (150 DPI)
   - Medium resolution for presentations/preview
   - File size: ~1-2 MB
   - Screen quality: Good
   - **USE THIS for PowerPoint slides**

---

## 📊 What the Dashboard Shows

### Top Panel: Clinical Case Overview
- **Demographics**: Age, weight, gender, admission type
- **Vital Signs**: HR, temperature, respiratory rate (with ↑↓ arrows for abnormal values)
- **Laboratory Values**: WBC, lactate, creatinine, hemoglobin (highlighted in red if abnormal)
- **AI Prediction**: 86.4% Sepsis Risk (HIGH RISK) in red box

### Middle-Left Panel: SHAP Waterfall Plot
- **Base value**: 0.183 (18.3% average sepsis risk in population)
- **Feature contributions**: 7 features with SHAP values
  - Red bars = increases risk (positive SHAP)
  - Blue bars = decreases risk (negative SHAP)
- **Final prediction**: 0.864 (86.4% risk) shown as red dashed line
- **Top contributors**: lactate (+0.121), wbc_count (+0.076), hr_bp_ratio (+0.033)

### Middle-Right Panel: LIME Explanations
- **5 local rules** explaining the prediction
- Format: "feature > threshold" (e.g., "creat >1.63")
- Warmer colors (orange/yellow) = stronger support for sepsis
- Weights show contribution to local linear model

### Bottom Panel: GradCAM Visual Attention
- **3 subplots**:
  1. Original chest X-ray (grayscale)
  2. Attention heatmap (jet colormap: blue=low, red=high)
  3. Overlay (heatmap blended with X-ray)
- **Attention regions**: Bilateral lower lung zones (sepsis infiltrates)
- **Model**: chest_xray_resnet50

---

## 🎨 How to Customize the Figure

### Change the Clinical Case

Edit the `ClinicalCase` class in `generate_dashboard_figure.py`:

```python
# Example: Create a kidney failure case instead
class ClinicalCase:
    def __init__(self, case_type='kidney_failure'):
        if case_type == 'kidney_failure':
            self.demographics = {
                'age': 68,
                'weight': 85,
                'gender': 'Male',
                'admission': 'ICU'
            }
            self.vitals = {
                'HR': (92, 'bpm', False, 60, 100),
                'Temp': (98.2, '°F', False, 97, 99),
                'RR': (18, '/min', False, 12, 20),
            }
            self.labs = {
                'Creatinine': (4.8, 'mg/dL', True, 0.6, 1.2),
                'BUN': (68, 'mg/dL', True, 7, 20),
                'Potassium': (6.2, 'mEq/L', True, 3.5, 5.0),
                'Hemoglobin': (9.2, 'g/dL', True, 12, 16)
            }
            self.prediction = {
                'disease': 'Kidney Failure',
                'risk': 78.5,
                'risk_level': 'HIGH RISK'
            }
            # ... update SHAP and LIME values accordingly
```

Then run:
```bash
python generate_dashboard_figure.py
```

### Adjust Figure Resolution

In the script:

```python
# For ultra-high quality (large file)
generate_dashboard(dpi=600, output_file='dashboard_ultra_hq.png')

# For web/email (smaller file)
generate_dashboard(dpi=72, output_file='dashboard_web.png')
```

### Change Color Scheme

Modify color constants at the top of the script:

```python
# Current colors
COLOR_SHAP_POS = '#FF4D4D'  # Red for positive SHAP
COLOR_SHAP_NEG = '#4D7CFF'  # Blue for negative SHAP

# Example: Colorblind-friendly palette
COLOR_SHAP_POS = '#D55E00'  # Orange
COLOR_SHAP_NEG = '#0173B2'  # Blue
```

### Add More Features to SHAP/LIME

Extend the `shap_values` and `lime_values` lists:

```python
self.shap_values = [
    ('lactate', 0.121, '+'),
    ('wbc_count', 0.076, '+'),
    # ... existing features ...
    ('platelet_count', -0.018, '-'),  # New feature
    ('albumin', -0.012, '-'),          # New feature
]
```

---

## 🔧 Advanced Customization

### Interactive Preview Mode

View the figure interactively before saving:

```bash
python generate_dashboard_figure.py --show
```

This opens matplotlib window where you can:
- Zoom into specific panels
- Adjust layout
- Export in different formats

### Export in Multiple Formats

Add to the script:

```python
# Save as vector graphic (scalable)
plt.savefig('dashboard.pdf', format='pdf', dpi=300)
plt.savefig('dashboard.svg', format='svg', dpi=300)

# Save as high-quality TIFF (journal requirement)
plt.savefig('dashboard.tiff', format='tiff', dpi=300, compression='tiff_lzw')
```

### Add Annotations/Arrows

Add after creating the figure panels:

```python
from matplotlib.patches import FancyArrowPatch

# Add arrow pointing to key feature
arrow = FancyArrowPatch((0.2, 0.5), (0.3, 0.6),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='red',
                        transform=fig.transFigure)
fig.patches.append(arrow)

# Add text annotation
ax.text(0.3, 0.6, 'Key Risk Driver!', fontsize=10,
        color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        transform=fig.transFigure)
```

---

## 📐 Figure Dimensions

### Current Settings:
- **Width**: 16 inches
- **Height**: 10 inches
- **Aspect ratio**: 1.6:1 (landscape)

### For Different Journals:

**IEEE Two-Column:**
```python
# Full width (across 2 columns)
fig = plt.figure(figsize=(7.16, 4.5))

# Single column width
fig = plt.figure(figsize=(3.5, 5))
```

**Nature/Science:**
```python
# Full page width
fig = plt.figure(figsize=(7.5, 4.7))
```

**PowerPoint Slide:**
```python
# Standard 16:9 slide
fig = plt.figure(figsize=(10, 5.625))
```

---

## 🎯 Best Practices for Publication

### For IEEE Submission:

1. **Use the 300 DPI version** (`dashboard_publication_figure.png`)
2. **Verify text legibility**: Zoom to 100% - all text should be readable
3. **File size**: Compress if >10 MB using:
   ```bash
   # Using ImageMagick
   magick convert dashboard_publication_figure.png -quality 95 dashboard_compressed.png
   
   # Using Python PIL
   from PIL import Image
   img = Image.open('dashboard_publication_figure.png')
   img.save('dashboard_compressed.png', optimize=True, quality=95)
   ```

4. **Caption**: Use the caption from `IEEE_FIGURES_AND_RESULTS.md` (Figure 8)

### For Presentations:

1. **Use the 150 DPI version** (`dashboard_preview.png`)
2. **Crop if needed**: Focus on specific panels for detail slides
3. **Add slide annotations**: Use PowerPoint drawing tools to highlight
4. **Consider split figures**: 
   - Slide 1: Overview + SHAP
   - Slide 2: LIME + GradCAM

### For Posters:

```python
# Extra large for poster printing
generate_dashboard(dpi=600, output_file='dashboard_poster.png')
```

---

## 🔍 Troubleshooting

### Issue: Text is too small

**Solution 1**: Increase base font size
```python
rcParams['font.size'] = 12  # Increase from 10
```

**Solution 2**: Make figure larger
```python
fig = plt.figure(figsize=(20, 12))  # Increase from (16, 10)
```

### Issue: Colors don't print well

**Solution**: Use colorblind-safe palette
```python
COLOR_SHAP_POS = '#E69F00'  # Orange
COLOR_SHAP_NEG = '#56B4E9'  # Sky blue
COLOR_LIME = ['#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
```

### Issue: File size too large

**Solution 1**: Reduce DPI
```python
generate_dashboard(dpi=200)  # Instead of 300
```

**Solution 2**: Use JPEG instead of PNG
```python
plt.savefig('dashboard.jpg', format='jpeg', dpi=300, quality=95)
```

### Issue: GradCAM heatmap not visible

**Solution**: Increase overlay alpha
```python
overlay = apply_gradcam_overlay(xray, heatmap, alpha=0.6)  # Increase from 0.4
```

---

## 📊 Data Sources for Realistic Cases

### Use Actual Model Outputs

If you have trained models, get real data:

```python
import joblib
import numpy as np

# Load trained model
model = joblib.load('trained_models/sepsis_model.pkl')
explainer = shap.TreeExplainer(model)

# Get real patient data
patient_data = {...}  # Your actual patient features

# Compute real SHAP values
shap_values = explainer.shap_values(patient_data)

# Update case with real values
case.shap_values = [
    (feature_names[i], shap_values[i], '+' if shap_values[i] > 0 else '-')
    for i in range(len(feature_names))
]
```

### Use MIMIC-III Examples

Extract real cases from your dataset:

```python
import pandas as pd

# Load MIMIC data
df = pd.read_csv('mimic_training_data.csv')

# Get high-risk sepsis patient
sepsis_case = df[df['sepsis'] == 1].iloc[0]

# Populate case
case.labs = {
    'WBC': (sepsis_case['wbc'], 'K/μL', True, 4, 11),
    'Lactate': (sepsis_case['lactate'], 'mmol/L', True, 0.5, 2.0),
    # ... etc
}
```

---

## 🎨 Example Variations

### Variation 1: Minimal Dashboard (Just SHAP + LIME)

```python
# Simpler 2-panel layout
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(1, 2, figure=fig)

ax_shap = fig.add_subplot(gs[0, 0])
plot_shap_waterfall(ax_shap, case)

ax_lime = fig.add_subplot(gs[0, 1])
plot_lime_explanation(ax_lime, case)
```

### Variation 2: GradCAM Focus (Large Heatmaps)

```python
# Emphasize visual explanations
fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 2])

# Small SHAP/LIME at top
ax_shap = fig.add_subplot(gs[0, :2])
ax_lime = fig.add_subplot(gs[0, 2])

# Large GradCAM at bottom
ax_gradcam = fig.add_subplot(gs[1, :])
```

### Variation 3: Multi-Patient Comparison

```python
# Show 2-3 patients side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, case in zip(axes, [case1, case2, case3]):
    plot_shap_waterfall(ax, case)
    ax.set_title(f"Patient {case.demographics['age']}y")
```

---

## 🚀 Quick Commands

```bash
# Generate default dashboard (sepsis case)
python generate_dashboard_figure.py

# Preview interactively
python generate_dashboard_figure.py --show

# Generate custom case (edit script first)
python generate_dashboard_figure.py

# Batch generate multiple cases
python -c "
from generate_dashboard_figure import *
for disease in ['sepsis', 'kidney_failure', 'heart_disease']:
    case = ClinicalCase(disease)
    generate_dashboard(case, f'dashboard_{disease}.png')
"
```

---

## 📝 Integration with IEEE Submission

### Step 1: Verify Quality

```bash
# Check file properties
file dashboard_publication_figure.png

# Expected output:
# PNG image data, 4800 x 3000, 8-bit/color RGB, non-interlaced
```

### Step 2: Add to Manuscript

In your LaTeX file:

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{dashboard_publication_figure.png}
\caption{Complete explainable AI dashboard for medical diagnosis
demonstrating integrated SHAP, LIME, and GradCAM explanations...
[full caption from IEEE_FIGURES_AND_RESULTS.md]}
\label{fig:dashboard}
\end{figure*}
```

### Step 3: Reference in Text

```latex
Figure~\ref{fig:dashboard} presents the complete system interface
integrating three complementary explainability methods in real-time...
```

---

## ✅ Checklist Before Submission

- [ ] Figure generated at 300 DPI
- [ ] All text is legible when zoomed to 100%
- [ ] Patient data is anonymized (no PHI)
- [ ] Colors are distinguishable in grayscale (print test)
- [ ] File size < 10 MB
- [ ] Caption copied from IEEE_FIGURES_AND_RESULTS.md
- [ ] Figure referenced in manuscript text
- [ ] Uploaded as separate file to submission system

---

**Your dashboard figure is publication-ready! 🎉**
