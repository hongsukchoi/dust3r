"""
Save frames from the EgoExo dataset

There are videos in this root directory: /scratch/partial_datasets/egoexo4d_v2/download_20240718/takes

root_directory/
    indiana_bike_04_5/
        frame_aligned_videos/
            cam01.mp4
            cam02.mp4
            ...
    ...

I want to save frames from each video into a directory with the following structure:
I want to save only 30th frame from each video.
The save directory will be: /scratch/partial_datasets/egoexo/hongsuk/egoexo/frame_aligned_videos

save_directory/
    indiana_bike_04_5/
        cam01/
            frame_000030.jpg
        cam02/
            frame_000030.jpg
        ...
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

def save_frame(video_path, save_dir, frame_number=30):
    """
    Extract and save a specific frame from a video
    
    Args:
        video_path (Path): Path to the video file
        save_dir (Path): Directory to save the frame
        frame_number (int): Frame number to extract (default: 30)
    """
    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)  # 0-based index
    
    # Read frame
    ret, frame = cap.read()
    
    if ret:
        # Save frame
        frame_path = save_dir / f"frame_{frame_number:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
    
    # Release video capture
    cap.release()

# Later, if you want to save otehr frames,process important videos first. 
# In splits.json, extract video names from "train": "body": and "val": "body": sections.
def main():
    # Define root and save directories
    root_dir = Path("/scratch/partial_datasets/egoexo4d_v2/download_20240718/takes")
    save_base_dir = Path("/scratch/partial_datasets/egoexo/hongsuk/egoexo/frame_aligned_videos")
    
    # Iterate through all take directories
    for take_dir in tqdm(list(root_dir.iterdir()), desc="Processing takes"):
        if not take_dir.is_dir():
            continue
            
        # Get the frame_aligned_videos directory
        video_dir = take_dir / "frame_aligned_videos"
        if not video_dir.exists():
            continue
            
        # Process each video file - only those starting with 'cam'
        for video_path in video_dir.glob("cam*.mp4"):  # Changed from *.mp4 to cam*.mp4
            # Get camera name (e.g., "cam01" from "cam01.mp4")
            camera_name = video_path.stem
            
            # Create save directory path
            save_dir = save_base_dir / take_dir.name / camera_name
            
            # Save the 30th frame
            save_frame(video_path, save_dir)

if __name__ == "__main__":
    main()
