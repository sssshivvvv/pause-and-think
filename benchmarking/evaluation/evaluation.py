import os
import base64
import openai
from typing import List
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path
from evaluation_prompt import evaluate_agent_performance_direct, evaluator_score_prompt_direct
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

API = "API_KEY"
URL = "URL"

class OpenAIModel:
    def __init__(self,
                 model_id="dvue-aoai-002-gpt-4o",
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


def evaluate_single_item(item, model, write_lock, output_file, processed_videos_lock, processed_videos):
    """Evaluate a single item and return the result"""
    
    if "question_video" in item.keys():
        question_video = item["question_video"]
    else:
        question_video = item["video"]

    video_path = item["video"]
    question = item["question"]
    ground_truth = item["ground_truth"]
    agent_response = item["agent_response"]
    
    # Check if already processed
    with processed_videos_lock:
        if video_path in processed_videos:
            print(f"⏭️  Skipping already evaluated: {os.path.basename(video_path)}")
            return None
    
    print(f"\n🎬 Evaluating: {os.path.basename(video_path)}")
    
    try:
        query = f"""
                question: {question}
                ground_truth: {ground_truth}
                agent_response: {agent_response}

                """
        
        image_contents = process_video(video_path)
        
        if image_contents is None:
            raise ValueError("Failed to process video")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": evaluator_score_prompt_direct + query}
                ] + image_contents
            }
        ]
        
        response = model.generate(messages)
        print(f"✅ Success for: {os.path.basename(video_path)}")
        response = clean_response(response)
        
    except Exception as e:
        print(f"❌ Error for {os.path.basename(video_path)}: {e}")
        response = None
    
    result = {
        "video": video_path,
        "question_video": question_video,
        "question": question,
        "ground_truth": ground_truth,
        "agent_response": agent_response,  # Fixed typo: "agent_respnse" -> "agent_response"
        "evaluation": response
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
    
    input_file = '<performance_file_path>'
    output_file = "evaluation.json"
    
    write_lock = Lock()
    processed_videos_lock = Lock()
    
    # Load existing evaluations and get processed videos
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            processed_videos = set([item["ground_truth"] for item in existing_data])
            print(f"📂 Found existing file with {len(processed_videos)} evaluated videos")
    else:
        existing_data = []
        processed_videos = set()
        # Initialize output file
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    # Load input file
    with open(input_file, 'r') as f:
        file = json.load(f)
    
    # Filter out already processed videos
    items_to_process = [item for item in file if item["ground_truth"] not in processed_videos]
    
    print(f"🚀 Starting evaluation with {max_num_workers} workers...")
    print(f"📊 Total items: {len(file)}")
    print(f"✅ Already evaluated: {len(processed_videos)}")
    print(f"📋 Items to evaluate: {len(items_to_process)}")
    
    if len(items_to_process) == 0:
        print("✨ All items already evaluated!")
        return
    
    start_time = time.time()
    completed = 0
    
    # Process items in parallel
    with ThreadPoolExecutor(max_workers=max_num_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(
                evaluate_single_item, 
                item, 
                model, 
                write_lock, 
                output_file,
                processed_videos_lock,
                processed_videos
            ): item 
            for item in items_to_process
        }
        
        # Process completed tasks
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                if result is not None:  # Only count if actually processed
                    completed += 1
                    print(f"\n📈 Progress: {completed}/{len(items_to_process)} ({completed/len(items_to_process)*100:.1f}%)")
            except Exception as exc:
                print(f"\n❌ Item {item['video']} generated an exception: {exc}")
                completed += 1
    
    end_time = time.time()
    print(f"\n✅ Evaluation complete!")
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    if completed > 0:
        print(f"📊 Average time per evaluation: {(end_time - start_time) / completed:.2f} seconds")
    print(f"💾 Results saved to: {output_file}")
    
    # Final count
    with open(output_file, 'r') as f:
        final_data = json.load(f)
    print(f"📁 Total evaluations in output file: {len(final_data)}")


if __name__ == "__main__":
    # You can adjust the number of workers here
    main(max_num_workers=5)
