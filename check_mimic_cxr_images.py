import pandas as pd
import ast
import os

# Load CSV
df = pd.read_csv(r'practice\mimic_cxr_aug_train.csv', nrows=5)

# Check first image path
img_path_str = df['image'].iloc[0]
print(f"Raw image data: {img_path_str}")

try:
    img_paths = ast.literal_eval(img_path_str)
    if isinstance(img_paths, list) and len(img_paths) > 0:
        img_path = img_paths[0]
        print(f"\nSample image path: {img_path}")
        
        # Check if file exists in various locations
        base_paths = [
            r'C:\Users\ADMIN\.cache\kagglehub',
            r'practice',
            r'.',
            r'C:\Users\ADMIN\.cache\kagglehub\datasets',
        ]
        
        found = False
        for base in base_paths:
            full_path = os.path.join(base, img_path)
            if os.path.exists(full_path):
                print(f'✅ Image found at: {full_path}')
                found = True
                break
        
        if not found:
            print('❌ Image files not found on disk')
            print('\nTo use GradCAM, you need to download the actual MIMIC-CXR image files.')
except Exception as e:
    print(f"Error parsing image path: {e}")

print(f"\n📊 CSV contains {len(df)} training samples with metadata")
print(f"Columns: {list(df.columns)}")
