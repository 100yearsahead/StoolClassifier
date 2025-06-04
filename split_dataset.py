import os
import shutil
import random

def split_dataset(source_dir, dest_dir, split_ratio=(0.7, 0.15, 0.15)):
    classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    for cls in classes:
        os.makedirs(os.path.join(dest_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'val', cls), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'test', cls), exist_ok=True)

        images = os.listdir(os.path.join(source_dir, cls))
        random.shuffle(images)
        train_split = int(split_ratio[0] * len(images))
        val_split = int(split_ratio[1] * len(images))

        for i, img in enumerate(images):
            if i < train_split:
                split = 'train'
            elif i < train_split + val_split:
                split = 'val'
            else:
                split = 'test'
            shutil.copy(os.path.join(source_dir, cls, img), os.path.join(dest_dir, split, cls, img))

# Example usage:
# split_dataset('path_to_downloaded_dataset', 'data')
# This is to preprare the data set split them into classes
