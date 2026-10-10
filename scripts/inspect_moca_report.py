"""
scripts/inspect_moca_report.py
==============================
Generates MoCA dataset report for YOLOv8 box generation.
Outputs:
  outputs/moca_yolov8_dataset_report.txt
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path

def main():
    seq_class_map = {}
    with open('outputs/sequence_class_mapping.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_class_map[row['sequence_name']] = row['normalized_animal_class']

    train_dir = Path('dataset/MoCA_Video/TrainDataset_per_sq')
    seq_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])

    rows = []
    total_imgs = 0
    total_masks = 0
    total_pairs = 0
    total_empty = 0
    total_valid_masks = 0
    total_valid_boxes = 0

    all_areas = []

    for seq in seq_dirs:
        seq_name = seq.name
        animal_class = seq_class_map.get(seq_name, seq_name)
        img_dir = seq / 'Imgs'
        gt_dir = seq / 'GT'

        imgs = sorted(list(img_dir.glob('*.*'))) if img_dir.is_dir() else []
        imgs = [f for f in imgs if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        masks = sorted(list(gt_dir.glob('*.*'))) if gt_dir.is_dir() else []
        masks = [f for f in masks if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]

        img_map = {f.stem: f for f in imgs}
        gt_map = {f.stem: f for f in masks}

        stems = sorted(set(img_map.keys()) & set(gt_map.keys()))

        n_imgs = len(imgs)
        n_masks = len(masks)
        n_pairs = len(stems)

        seq_empty = 0
        seq_valid = 0
        seq_boxes = 0
        seq_areas = []

        for stem in stems:
            m_path = gt_map[stem]
            m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            H, W = m.shape
            fg_pixels = int(np.sum(m > 127))
            fg_ratio = fg_pixels / (H * W)

            if fg_pixels == 0:
                seq_empty += 1
            elif fg_ratio < 0.0005:
                seq_empty += 1
            else:
                seq_valid += 1
                seq_boxes += 1
                seq_areas.append(fg_ratio * 100)
                all_areas.append(fg_ratio * 100)

        total_imgs += n_imgs
        total_masks += n_masks
        total_pairs += n_pairs
        total_empty += seq_empty
        total_valid_masks += seq_valid
        total_valid_boxes += seq_boxes

        mean_area = np.mean(seq_areas) if seq_areas else 0.0
        min_area = np.min(seq_areas) if seq_areas else 0.0
        max_area = np.max(seq_areas) if seq_areas else 0.0

        rows.append({
            'seq': seq_name,
            'class': animal_class,
            'imgs': n_imgs,
            'masks': n_masks,
            'pairs': n_pairs,
            'empty': seq_empty,
            'valid': seq_valid,
            'boxes': seq_boxes,
            'mean_area': mean_area,
            'min_area': min_area,
            'max_area': max_area
        })

    unique_classes = set(r['class'] for r in rows)

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/moca_yolov8_dataset_report.txt', 'w', encoding='utf-8') as f:
        f.write('=' * 110 + '\n')
        f.write('MoCA DATASET INSPECTION & YOLOv8 BOX GENERATION REPORT\n')
        f.write('=' * 110 + '\n\n')
        f.write(f'Total Sequence Folders Found : {len(seq_dirs)}\n')
        f.write(f'Total Normalized Classes     : {len(unique_classes)}\n')
        f.write(f'Total Images                 : {total_imgs}\n')
        f.write(f'Total GT Masks               : {total_masks}\n')
        f.write(f'Total Valid Image-Mask Pairs : {total_pairs}\n')
        f.write(f'Total Empty/Near-Empty Masks : {total_empty}\n')
        f.write(f'Total Valid Foreground Masks : {total_valid_masks}\n')
        f.write(f'Total Valid Bounding Boxes   : {total_valid_boxes}\n')
        f.write(f'Overall Foreground Area %    : Mean={np.mean(all_areas):.2f}%, Min={np.min(all_areas):.3f}%, Max={np.max(all_areas):.2f}%\n\n')

        f.write('=' * 110 + '\n')
        f.write(f'{"Sequence":<32} | {"Animal Class":<26} | {"Images":>6} | {"Masks":>6} | {"Pairs":>6} | {"Valid Boxes":>11} | {"Mean FG%":>8}\n')
        f.write('-' * 110 + '\n')
        for r in rows:
            f.write(f'{r["seq"][:32]:<32} | {r["class"][:26]:<26} | {r["imgs"]:>6} | {r["masks"]:>6} | {r["pairs"]:>6} | {r["boxes"]:>11} | {r["mean_area"]:>7.2f}%\n')
        f.write('=' * 110 + '\n')

    print(f'Successfully generated outputs/moca_yolov8_dataset_report.txt.')
    print(f'Total sequences: {len(seq_dirs)}, Total valid pairs: {total_pairs}, Valid boxes: {total_valid_boxes}')

if __name__ == '__main__':
    main()
