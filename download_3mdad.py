import kagglehub
import os

print("Starting 3MDAD dataset download via kagglehub...")
path = kagglehub.dataset_download("anouarbenkhalifa/3mdad-multimodal-driver-action-distraction")
print("\nDownload complete!")
print("Path to dataset files:", path)
