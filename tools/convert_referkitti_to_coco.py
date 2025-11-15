import os
import numpy as np
import json
import cv2


DATA_PATH = 'datasets/refer-kitti-v2'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
SPLITS = ['train', 'test']


def convert_bbox_yolo_to_coco(bbox, img_width, img_height):
    """
    Convert YOLO format (normalized x_center, y_center, width, height) 
    to COCO format (x_top_left, y_top_left, width, height in pixels)
    """
    x_center, y_center, width, height = bbox
    
    # Convert from normalized to pixel coordinates
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Convert from center format to top-left format
    x_topleft = x_center - width / 2
    y_topleft = y_center - height / 2
    
    return [x_topleft, y_topleft, width, height]


if __name__ == '__main__':
    
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    
    for split in SPLITS:
        # For refer-kitti-v2, we use training data for both train and test
        # You may need to modify this based on your actual train/test split
        data_path = os.path.join(DATA_PATH, 'KITTI/training/image_02')
        label_path = os.path.join(DATA_PATH, 'KITTI/labels_with_ids/image_02')
        
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'car'}]}
        
        seqs = sorted(os.listdir(data_path))
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        
        # Split sequences: first 80% for train, last 20% for test
        num_seqs = len(seqs)
        train_seqs = int(num_seqs * 0.8)
        
        if split == 'train':
            seqs = seqs[:train_seqs]
        else:
            seqs = seqs[train_seqs:]
        
        for seq in seqs:
            if '.DS_Store' in seq:
                continue
            
            video_cnt += 1
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            
            seq_img_path = os.path.join(data_path, seq)
            seq_label_path = os.path.join(label_path, seq)
            
            images = sorted([img for img in os.listdir(seq_img_path) if img.endswith('.png')])
            num_images = len(images)
            
            print(f'Processing {seq}: {num_images} images')
            
            for i, img_name in enumerate(images):
                img_file = os.path.join(seq_img_path, img_name)
                img = cv2.imread(img_file)
                if img is None:
                    print(f'Warning: Could not read {img_file}')
                    continue
                    
                height, width = img.shape[:2]
                
                image_info = {
                    'file_name': f'{seq}/{img_name}',
                    'id': image_cnt + i + 1,
                    'frame_id': i + 1,
                    'prev_image_id': image_cnt + i if i > 0 else -1,
                    'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                    'video_id': video_cnt,
                    'height': height,
                    'width': width
                }
                out['images'].append(image_info)
                
                # Process annotations
                label_file = os.path.join(seq_label_path, img_name.replace('.png', '.txt'))
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 6:
                            continue
                        
                        class_id = int(parts[0])
                        track_id = int(parts[1])
                        bbox_yolo = [float(x) for x in parts[2:6]]
                        
                        # Convert YOLO bbox to COCO bbox
                        bbox_coco = convert_bbox_yolo_to_coco(bbox_yolo, width, height)
                        
                        ann_cnt += 1
                        ann = {
                            'id': ann_cnt,
                            'category_id': 1,  # All objects are 'car' class
                            'image_id': image_cnt + i + 1,
                            'track_id': track_id,
                            'bbox': bbox_coco,
                            'conf': 1.0,
                            'iscrowd': 0,
                            'area': float(bbox_coco[2] * bbox_coco[3])
                        }
                        out['annotations'].append(ann)
            
            image_cnt += num_images
        
        print(f'Loaded {split}: {len(out["images"])} images and {len(out["annotations"])} annotations')
        json.dump(out, open(out_path, 'w'))
