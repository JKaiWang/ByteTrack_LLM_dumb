# Notes

## Double Dataset Initialization Pattern

In `yolox/exp/yolox_base.py` (lines 91-119), the `dataset` variable is initialized twice:

```python
# First initialization - base dataset
dataset = COCODataset(...)

# Second initialization - wrapper with augmentation
dataset = MosaicDetection(
    dataset,  # wraps the base dataset
    mosaic=not no_aug,
    ...
)
```

**This is intentional - it's a decorator/wrapper pattern:**

1. **Base layer**: `COCODataset` loads raw images and annotations from COCO-format JSON files
2. **Augmentation layer**: `MosaicDetection` wraps the base dataset and applies data augmentation (mosaic, mixup, etc.)

The final `dataset` contains both the data loading logic (from COCODataset) and augmentation logic (from MosaicDetection).

## What is COCODataset?

**COCO (Common Objects in Context)** is a standard format for object detection datasets.

COCO structure:

```json
{
  "images": [
    {
      "file_name": "000000000009.jpg",
      "height": 480,
      "width": 640,
      "id": 9
    }
  ],
  "annotations": [
    {
      "id": 45,
      "image_id": 9,
      "category_id": 18,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 18, "name": "dog"}
  ]
}
```

COCO directory structure:

```json
coco/
│
├── train2017/       # ~118k images
├── val2017/         # ~5k images
├── test2017/        # ~41k images
│
└── annotations/
      ├── instances_train2017.json
      ├── instances_val2017.json
      ├── person_keypoints_train2017.json
      ├── captions_train2017.json

```

## `image_demo()`

`bytetrack_inference.py` (118-124)

```python
self.json_path = os.path.join(
    self.args.rmot_path, "expression", seq_num[0], seq_num[1]
)
with open(self.json_path, "r") as f:
    json_info = json.load(f)
self.json_info = json_info
self.sentence = [json_info["sentence"]]
```
Path Structure: `{rmot_path}/expression/{video_id}/{expression_json}`
For example: `./datasets/refer-kitti/expression/0005/expression_0001.json`
