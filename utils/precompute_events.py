import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.event_from_video import video_to_events

OUTPUT_FOLDER = "data/events"

def process_single_video(video_path):
    """Worker function for multiprocessing."""
    file = os.path.basename(video_path)
    save_path = os.path.join(OUTPUT_FOLDER, file + ".npy")
    
    if os.path.exists(save_path):
        return (file, "skipped", 0)

    try:
        events = video_to_events(video_path)
        np.save(save_path, events)
        return (file, "processed", len(events))
    except Exception as e:
        return (file, "error", str(e))

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Gather roots to scan (DMD and Drive&Act folders)
    search_roots = [os.path.expanduser("~/Desktop/dmd")]
    downloads = os.path.expanduser("~/Downloads")
    
    if os.path.exists(downloads):
        for name in sorted(os.listdir(downloads)):
            if (name.lower().startswith("dmd") or 
                name in ["a_column_driver", "inner_mirror", "activities_3s"]):
                full_path = os.path.join(downloads, name)
                if os.path.isdir(full_path):
                    search_roots.append(full_path)

    print(f"Scanning {len(search_roots)} source folder(s):")
    for r in search_roots:
        print(f"  {r}")
    print()

    # 2. Find all unique video files
    video_files = []
    seen = set()
    
    for root_folder in search_roots:
        for dirpath, dirs, files in os.walk(root_folder):
            for file in sorted(files):
                if not (file.endswith(".mp4") or file.endswith(".avi")):
                    continue
                # We process ALL videos (face, body, IR, depth, etc.)
                if file in seen:
                    continue
                seen.add(file)
                video_files.append(os.path.join(dirpath, file))
                
    print(f"Found {len(video_files)} unique videos to check/process.")

    # 3. Process in parallel
    processed = 0
    skipped = 0
    errors = 0
    
    # Use max available CPU cores, leaving 1 for system responsiveness
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Starting ProcessPoolExecutor with {num_workers} workers...\n")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_video, vp): vp for vp in video_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            file, status, info = future.result()
            
            if status == "skipped":
                skipped += 1
            elif status == "processed":
                processed += 1
                print(f"[{i}/{len(video_files)}] Processed: {file} → Saved {info} events")
            elif status == "error":
                errors += 1
                print(f"[{i}/{len(video_files)}] Error on {file}: {info}")
                
            # Print a subtle progress indicator for skips to avoid terminal spam
            if status == "skipped" and (i % 50 == 0 or i == len(video_files)):
                print(f"[{i}/{len(video_files)}] Progress (skipped {skipped} cached files)...")

    print(f"\nDone! Processed {processed} new videos, skipped {skipped} cached, {errors} errors.")
    print(f"Events saved in: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()