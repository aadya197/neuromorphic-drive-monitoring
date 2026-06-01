import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.event_from_video import video_to_events

output_folder = "data/events"
os.makedirs(output_folder, exist_ok=True)

# Scan Desktop/dmd AND all dmd* folders in Downloads — no copying needed
search_roots = [os.path.expanduser("~/Desktop/dmd")]
downloads = os.path.expanduser("~/Downloads")
for name in sorted(os.listdir(downloads)):
    if name.lower().startswith("dmd") and os.path.isdir(os.path.join(downloads, name)):
        search_roots.append(os.path.join(downloads, name))

print(f"Scanning {len(search_roots)} source folder(s):")
for r in search_roots:
    print(f"  {r}")
print()

processed = 0
skipped   = 0
seen      = set()   # deduplicate by filename

for root_folder in search_roots:
    for dirpath, dirs, files in os.walk(root_folder):
        for file in sorted(files):
            if not (file.endswith(".mp4") or file.endswith(".avi")):
                continue
            if "rgb_face" not in file:
                continue

            # Skip duplicates across folders
            if file in seen:
                continue
            seen.add(file)

            save_path = os.path.join(output_folder, file + ".npy")
            if os.path.exists(save_path):
                print(f"  Already cached: {file}")
                skipped += 1
                continue

            video_path = os.path.join(dirpath, file)
            print(f"Processing: {file}")
            events = video_to_events(video_path)
            np.save(save_path, events)
            print(f"  → Saved {len(events)} events")
            processed += 1

print(f"\nDone! Processed {processed} new videos, skipped {skipped} cached.")
print(f"Events saved in: {os.path.abspath(output_folder)}")