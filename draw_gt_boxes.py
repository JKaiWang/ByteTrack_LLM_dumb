#!/usr/bin/env python3
"""
Draw bounding boxes according to gt.txt and generate MP4 videos.

GT.txt format (MOT format):
frame_id, track_id, x, y, w, h, conf, class, visibility
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random


def generate_colors(num_colors):
    """Generate distinct colors for different track IDs."""
    random.seed(42)
    colors = {}
    for i in range(num_colors):
        colors[i] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
    return colors


def read_gt_file(gt_path):
    """Read gt.txt file and organize by frame."""
    frame_data = {}

    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            if frame_id not in frame_data:
                frame_data[frame_id] = []

            frame_data[frame_id].append({"track_id": track_id, "bbox": (x, y, w, h)})

    return frame_data


def draw_boxes_on_frame(frame, detections, colors, line_thickness=2):
    """Draw bounding boxes on a frame."""
    for det in detections:
        track_id = det["track_id"]
        x, y, w, h = det["bbox"]

        # Convert to integer coordinates
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Get color for this track ID
        color = colors.get(track_id, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

        # Draw track ID label
        label = f"ID: {track_id}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - baseline - 5),
            (x1 + label_size[0], y1),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return frame


def draw_single_frame(gt_path, images_dir, frame_id, output_path):
    """Draw bounding boxes on a single frame and save as image.
    
    Args:
        gt_path: Path to gt.txt file
        images_dir: Directory containing the image frames
        frame_id: Frame ID to draw (as it appears in gt.txt)
        output_path: Path where the output image will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Read gt.txt
    print(f"Reading {gt_path}...")
    frame_data = read_gt_file(gt_path)
    
    if not frame_data:
        print(f"No data found in {gt_path}")
        return False
    
    # Get all track IDs for color generation
    all_track_ids = set()
    for detections in frame_data.values():
        for det in detections:
            all_track_ids.add(det["track_id"])
    
    colors = generate_colors(max(all_track_ids) + 1 if all_track_ids else 100)
    
    # Construct image path (assuming format: 000000.png, 000001.png, etc.)
    img_name = f"{frame_id-1:06d}.png"
    img_path = os.path.join(images_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        return False
    
    # Read image
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error: Failed to read image: {img_path}")
        return False
    
    # Draw bounding boxes if detections exist for this frame
    if frame_id in frame_data:
        frame = draw_boxes_on_frame(frame, frame_data[frame_id], colors)
        print(f"Drew {len(frame_data[frame_id])} bounding boxes on frame {frame_id}")
    else:
        print(f"Warning: No detections found for frame {frame_id}")
    
    # Save the image
    cv2.imwrite(output_path, frame)
    print(f"Image saved to {output_path}")
    return True


def create_video_from_gt(gt_path, images_dir, output_path, fps=10):
    """Create video from images with bounding boxes drawn according to gt.txt."""

    # Read gt.txt
    print(f"Reading {gt_path}...")
    frame_data = read_gt_file(gt_path)

    if not frame_data:
        print(f"No data found in {gt_path}")
        return False

    # Get all track IDs for color generation
    all_track_ids = set()
    for detections in frame_data.values():
        for det in detections:
            all_track_ids.add(det["track_id"])

    colors = generate_colors(max(all_track_ids) + 1 if all_track_ids else 100)

    # Get sorted frame IDs
    frame_ids = sorted(frame_data.keys())
    min_frame = frame_ids[0]
    max_frame = frame_ids[-1]

    print(f"Processing frames {min_frame} to {max_frame}...")

    # Initialize video writer
    video_writer = None
    frame_count = 0

    # Process each frame
    for frame_id in tqdm(range(min_frame, max_frame + 1), desc="Creating video"):
        # Construct image path (assuming format: 000000.png, 000001.png, etc.)
        img_name = f"{frame_id-1:06d}.png"
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        # Read image
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Failed to read image: {img_path}")
            continue

        # Initialize video writer with first frame dimensions
        if video_writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Video dimensions: {width}x{height}, FPS: {fps}")

        # Draw bounding boxes if detections exist for this frame
        if frame_id in frame_data:
            frame = draw_boxes_on_frame(frame, frame_data[frame_id], colors)

        # Write frame to video
        video_writer.write(frame)
        frame_count += 1

    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {output_path} ({frame_count} frames)")
        return True
    else:
        print("Failed to create video")
        return False


def process_sequence(seq_dir, images_base_dir, output_base_dir, fps=10):
    """Process a single sequence directory containing multiple gt.txt files."""

    # Get sequence ID from path (e.g., "0011")
    path_parts = seq_dir.rstrip("/").split("/")
    seq_id = None
    for part in reversed(path_parts):
        if part.isdigit() and len(part) == 4:
            seq_id = part
            break

    if seq_id is None:
        print(f"Could not determine sequence ID from {seq_dir}")
        return

    # Find images directory
    images_dir = os.path.join(images_base_dir, seq_id)
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Processing sequence: {seq_id}")
    print(f"Sequence directory: {seq_dir}")
    print(f"Images directory: {images_dir}")
    print(f"{'='*80}\n")

    # Find all subdirectories with gt.txt
    for root, dirs, files in os.walk(seq_dir):
        if "gt.txt" in files:
            gt_path = os.path.join(root, "gt.txt")

            # Get subdirectory name (e.g., "persons-who-are-walking")
            subdir_name = os.path.basename(root)

            # Create output filename
            output_filename = f"{seq_id}_{subdir_name}.mp4"
            output_path = os.path.join(output_base_dir, output_filename)

            print(f"\nProcessing: {subdir_name}")
            print(f"GT file: {gt_path}")
            print(f"Output: {output_path}")

            # Create video
            create_video_from_gt(gt_path, images_dir, output_path, fps)


def main():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes from gt.txt and generate videos or single frame images"
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        required=True,
        help="Path to gt.txt file (e.g., exps/bytetrack/results_epoch52/0011/persons-who-are-walking/gt.txt)",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/home/seanachan/data/Dataset/refer-kitti/KITTI/training/image_02",
        help="Base directory containing image sequences",
    )
    parser.add_argument(
        "--output_dir", type=str, default="videos", help="Output directory for videos/images"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output filename (default: auto-generated from gt.txt path)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for output video"
    )
    parser.add_argument(
        "--frame_id",
        type=int,
        default=None,
        help="Draw only a single frame (by frame ID from gt.txt) and save as image",
    )

    args = parser.parse_args()

    # Verify gt.txt exists
    if not os.path.exists(args.gt_file):
        print(f"Error: gt.txt file not found: {args.gt_file}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract sequence ID from path
    path_parts = args.gt_file.split("/")
    seq_id = None
    for part in path_parts:
        if part.isdigit() and len(part) == 4:
            seq_id = part
            break

    if seq_id is None:
        print(f"Error: Could not determine sequence ID from {args.gt_file}")
        return

    # Find images directory
    images_dir = os.path.join(args.images_dir, seq_id)
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return

    # Check if single frame mode
    if args.frame_id is not None:
        # Single frame mode
        if args.output_name:
            output_filename = (
                args.output_name
                if args.output_name.endswith((".png", ".jpg"))
                else f"{args.output_name}.png"
            )
        else:
            # Auto-generate from path (e.g., "0011_persons-who-are-walking_frame123.png")
            subdir_name = os.path.basename(os.path.dirname(args.gt_file))
            output_filename = f"{seq_id}_{subdir_name}_frame{args.frame_id:06d}.png"
        
        output_path = os.path.join(args.output_dir, output_filename)
        
        print(f"\n{'='*80}")
        print("Processing single frame:")
        print(f"  GT file: {args.gt_file}")
        print(f"  Sequence: {seq_id}")
        print(f"  Images: {images_dir}")
        print(f"  Frame ID: {args.frame_id}")
        print(f"  Output: {output_path}")
        print(f"{'='*80}\n")
        
        # Draw single frame
        success = draw_single_frame(args.gt_file, images_dir, args.frame_id, output_path)
        
        if success:
            print(f"\n{'='*80}")
            print(f"Done! Image saved to {output_path}")
            print(f"{'='*80}")
        else:
            print(f"\nFailed to create image.")
    else:
        # Video mode
        # Generate output filename
        if args.output_name:
            output_filename = (
                args.output_name
                if args.output_name.endswith(".mp4")
                else f"{args.output_name}.mp4"
            )
        else:
            # Auto-generate from path (e.g., "0011_persons-who-are-walking.mp4")
            subdir_name = os.path.basename(os.path.dirname(args.gt_file))
            output_filename = f"{seq_id}_{subdir_name}.mp4"

        output_path = os.path.join(args.output_dir, output_filename)

        print(f"\n{'='*80}")
        print("Processing video:")
        print(f"  GT file: {args.gt_file}")
        print(f"  Sequence: {seq_id}")
        print(f"  Images: {images_dir}")
        print(f"  Output: {output_path}")
        print(f"{'='*80}\n")

        # Create video
        success = create_video_from_gt(args.gt_file, images_dir, output_path, args.fps)

        if success:
            print(f"\n{'='*80}")
            print(f"Done! Video saved to {output_path}")
            print(f"{'='*80}")
        else:
            print(f"\nFailed to create video.")


if __name__ == "__main__":
    main()
