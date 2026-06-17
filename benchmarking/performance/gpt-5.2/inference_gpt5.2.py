import os
import base64
import openai
from typing import List
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path
from inference_prompts_direct import ass, epic
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time



API = "API_KEY"
URL = "URL"


class OpenAIModel:
    def __init__(self,
                 model_id="dvue-aoai-001-gpt-5.2",
                 model_api_version='2024-12-01-preview',
                 api_key=None):
        self.model_id = model_id
        self.model_api_version = model_api_version

        url = URL
        headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
        self.client = openai.AzureOpenAI(
            api_key='dummy',
            api_version=self.model_api_version,
            base_url=url,
            default_headers=headers
        )
        self.client.base_url = '{0}/openai/deployments/{1}'.format(url, self.model_id)

    def generate(self,
                 messages: List,
                 temperature=1,
                 presence_penalty=0,
                 frequency_penalty=0,
                 max_tokens=500) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            n=1,
            stream=False,
            stop=None,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=None,
            user=None
        )

        if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
            raise ValueError("No response choices returned from the API.")

        return response.choices[0].message.content


def clean_response(response):
    cleaned = response.strip()        # remove leading/trailing whitespace
    cleaned = cleaned.strip('"')      # remove starting and ending quotes

    data = json.loads(cleaned)
    return data


def process_video(VIDEO_PATH, NUM_FRAMES=50):
    # Check if file exists
    if os.path.exists(VIDEO_PATH):
        print(f"Video found! Size: {os.path.getsize(VIDEO_PATH) / (1024*1024):.2f} MB")
    else:
        print("Video file not found!")
        return None

    print(f"\nProcessing video: {os.path.basename(VIDEO_PATH)}")

    cap = cv2.VideoCapture(VIDEO_PATH)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames!")

    # Pick evenly spaced frame indices
    frame_indices = [int(i * total_frames / NUM_FRAMES) for i in range(NUM_FRAMES)]

    image_contents = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')
        frame_data_url = f"data:image/jpeg;base64,{base64_frame}"

        # Append to message content
        image_contents.append({"type": "image_url", "image_url": {"url": frame_data_url}})

    cap.release()
    return image_contents


def process_single_video_item(item, model, write_lock, output_file, processed_videos_lock, processed_videos):
    """Process a single video item and return the result"""

    if "question_video" in item.keys():
        video_path = item["question_video"]
        orig_video = item["video"]
    else:
        video_path = item["video"]
        orig_video = item["video"]

    # video_path = item.get("question_video") or item["video"]
    question = item["question"]
    ground_truth = item["ground_truth"]
    
    # Check if already processed
    with processed_videos_lock:
        if video_path in processed_videos:
            print(f"⏭️  Skipping already processed: {os.path.basename(video_path)}")
            return None
    
    print(f"\n🎬 Processing: {os.path.basename(video_path)}")
    
    try:
        image_contents = process_video(video_path)
        
        if image_contents is None:
            raise ValueError("Failed to process video")
        
        # Select appropriate prompt
        if "/assembly/" in video_path or "/disassembly/" in video_path:
            prompt = ass
        else:
            prompt = epic
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + question}
                ] + image_contents
            }
        ]
        
        response = model.generate(messages)
        print(f"✅ Success for: {os.path.basename(video_path)}")
        
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


def main(max_num_workers=10):
    """Main function with multithreading support and resume capability"""
    
    # Create model instance
    model = OpenAIModel(
            model_id="dvue-aoai-001-gpt-5.2",
            model_api_version="2025-04-01-preview",
            api_key=API
        )

    
    # Load benchmark file
    with open("benchmarking/benchmark_300_files.json", 'r') as f:
        file = json.load(f)
    
    output_file = '300_performance_gpt5.2_again.json'
    write_lock = Lock()
    processed_videos_lock = Lock()
    
    # Load existing data and get processed videos
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            processed_videos = set([item["video"] for item in existing_data])
            print(f"📂 Found existing file with {len(processed_videos)} processed videos")
    else:
        existing_data = []
        processed_videos = set()
        # Initialize output file
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    # Filter out already processed videos
    videos_to_process = [item for item in file if item["video"] not in processed_videos]
    
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
                model, 
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
    # You can adjust the number of workers here
    main(max_num_workers=3)
