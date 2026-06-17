import os
import cv2
import base64
import requests
from PIL import Image
import io
import json
import random
import subprocess
import math
import time
from inference_prompts_direct import ass, epic
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from google import genai
import time
import os

os.environ["GEMINI_API_KEY"] = "API_KEY"

def clean_response(response):
    cleaned = response.strip()        # remove leading/trailing whitespace
    cleaned = cleaned.strip('"')      # remove starting and ending quotes

    data = json.loads(cleaned)
    return data


def ask_gemini(video_path, prompt):
    client = genai.Client()

    # Upload the file
    myfile = client.files.upload(file=video_path)

    print(f"Uploaded file: {myfile.name}")

    # Wait for file to be processed with timeout
    max_wait_time = 300  # 5 minutes timeout
    wait_interval = 5  # Check every 5 seconds
    elapsed_time = 0

    while myfile.state != "ACTIVE" and elapsed_time < max_wait_time:
        print(f"File state: {myfile.state}. Waiting...")
        time.sleep(wait_interval)
        elapsed_time += wait_interval
        myfile = client.files.get(name=myfile.name)

    if myfile.state != "ACTIVE":
        raise TimeoutError(f"File processing timed out after {max_wait_time} seconds. Current state: {myfile.state}")

    print("File is ready!")

    # Generate content
    response = client.models.generate_content(
        model="gemini-robotics-er-1.5-preview", 
        contents=[myfile, prompt]
    )

    output = response.text
    
    return output


def process_single_video_item(item, write_lock, output_file, processed_videos_lock, processed_videos):
    """Process a single video item and return the result"""

    if "question_video" in item.keys():
        video_path = item["question_video"]
        orig_video = item["video"]
    else:
        video_path = item["video"]
        orig_video = item["video"]

    question = item["question"]
    ground_truth = item["ground_truth"]
    
    # Check if already processed
    with processed_videos_lock:
        if video_path in processed_videos:
            print(f"⏭️  Skipping already processed: {os.path.basename(video_path)}")
            return None
    
    print(f"\n🎬 Processing: {os.path.basename(video_path)}")
    
    try:
        # Select appropriate prompt
        if "/assembly/" in video_path or "/disassembly/" in video_path:
            prompt = ass
        else:
            prompt = epic
        
        query = prompt + question
        
        # Call Qwen API
        response = ask_gemini(video_path, query)
        
        # Try to clean response
        try:
            response = clean_response(response)
        except:
            pass
        
        print(f"✅ Success for: {os.path.basename(video_path)}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Error for {os.path.basename(video_path)}: {e}")
        response = None
    
    result = {
        "video": orig_video,
        "question_video": video_path,
        "question": question,
        "ground_truth": ground_truth,
        "agent_response": response
    }
    
    # Write result immediately with thread safety
    with write_lock:
        # Read existing data
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        # Append new result
        data.append(result)
        
        # Write back
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Update processed videos set
    with processed_videos_lock:
        processed_videos.add(video_path)
    
    return result


def main(max_num_workers=5):  # Reduced default workers since Qwen might be resource intensive
    """Main function with multithreading support and resume capability"""
    
    input_file = "benchmarking/benchmark_300_files.json"
    output_file = '300_performance_gemini-er.json'
    
    write_lock = Lock()
    processed_videos_lock = Lock()
    
    # Load existing data and get processed videos
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            processed_videos = set([item["ground_truth"] for item in existing_data])
            print(f"📂 Found existing file with {len(processed_videos)} processed videos")
    else:
        existing_data = []
        processed_videos = set()
        # Initialize output file
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    # Load benchmark file
    with open(input_file, 'r') as f:
        file = json.load(f)
    
    # Filter out already processed videos
    videos_to_process = [item for item in file if item["ground_truth"] not in processed_videos]
    
    print(f"🚀 Starting processing with {max_num_workers} workers...")
    print(f"📊 Total videos: {len(file)}")
    print(f"✅ Already processed: {len(processed_videos)}")
    print(f"📋 Videos to process: {len(videos_to_process)}")
    
    if len(videos_to_process) == 0:
        print("✨ All videos already processed!")
        return
    
    start_time = time.time()
    completed = 0
    
    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=max_num_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(
                process_single_video_item, 
                item, 
                write_lock, 
                output_file,
                processed_videos_lock,
                processed_videos
            ): item 
            for item in videos_to_process
        }
        
        # Process completed tasks
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                if result is not None:  # Only count if actually processed
                    completed += 1
                    print(f"\n📈 Progress: {completed}/{len(videos_to_process)} ({completed/len(videos_to_process)*100:.1f}%)")
            except Exception as exc:
                print(f"\n❌ Video {item['video']} generated an exception: {exc}")
                completed += 1
    
    end_time = time.time()
    print(f"\n✅ Processing complete!")
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    if completed > 0:
        print(f"📊 Average time per video: {(end_time - start_time) / completed:.2f} seconds")
    print(f"💾 Results saved to: {output_file}")
    
    # Final count
    with open(output_file, 'r') as f:
        final_data = json.load(f)
    print(f"📁 Total videos in output file: {len(final_data)}")


if __name__ == "__main__":
    main(max_num_workers=10)



