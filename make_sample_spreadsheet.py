# %%
import os 
import pandas as pd
from pathlib import Path
# %%
data = {'path': [], 'label':[]}

# the absolute path of the dir 
path = Path('/zpool/projects/visualizing_image_embeddings/Kather_texture_2016_image_tiles_5000')

for subdir in path.iterdir():
    for im_path in subdir.iterdir():
        data['path'].append(str(im_path))
        data['label'].append(subdir.stem)
    
# %%
# run below for Windows
#for root, dirs, files in os.walk(path):
#    for name in files:
#        im_path = os.path.join(root, name)
#        label = im_path.split("\\")[3]
#        data['file_names'].append(im_path)
#        data['labels'].append(label)
# %%

df = pd.DataFrame(data)
df.to_csv(r'impath_to_label.csv', index=False)
