# main.py
import os
import re
import shlex
import json
import subprocess
import cv2

from preprocessing.frame_extraction import extract_frames
from preprocessing.scene_detector import SceneDetector
from utils.subtitle_generator import generate_srt
from utils.scene_merger import SceneMerger
from models.summary_generator import SummaryGenerator
from utils.outlier_filter import OutlierFilter
from models.clip_video_encoder import extract_clip_video_embedding
from models.frame_selector import FrameSelector
from utils.text_utils import fix_overlapping_scenes


class VideoCaptioning:
    """
    Video captioning pipeline coordinator. Handles:
    - Frame extraction
    - Scene segmentation
    - Frame-level captioning
    - Semantic scene merging
    - Summary generation
    - Optional subtitle embedding
    """
    def __init__(self, folder_name, generate_subtitles):
        self.folder_name = folder_name
        self.generate_subtitles = generate_subtitles

        # Initialize all pipeline components
        self.frame_selector = FrameSelector()
        self.scene_detector = SceneDetector()
        self.scene_merger = SceneMerger()
        self.summary_generator = SummaryGenerator()
        self.outlier_filter = OutlierFilter()

    def get_video_paths(self):
        """Get list of video file paths from a folder."""
        base_folder = os.path.join("data", "videos", self.folder_name)
        if not os.path.isdir(base_folder):
            raise ValueError(f"❌ Folder not found: {base_folder}")
        return sorted([
            os.path.join(base_folder, f)
            for f in os.listdir(base_folder)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ])

    def get_video_name(self, path):
        """Extract file name without extension."""
        return os.path.splitext(os.path.basename(path))[0]

    def save_results_as_json(self, video_id, summary_caption, scene_captions, output_folder):
        """Save the generated captions and summary to a JSON file."""
        result = {
            "video_id": video_id,
            "summary": summary_caption,
            "scene_captions": [
                {
                    "start_time": sc["start_time"],
                    "end_time": sc["end_time"],
                    "caption": sc["caption"]
                } for sc in scene_captions
            ]
        }
        json_path = os.path.join(output_folder, f"{video_id}_captions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Captions and summary saved as JSON: {json_path}")

    def process_video(self, video_path):
        """Complete processing of a single video file."""
        video_name = self.get_video_name(video_path)
        print(f"\n🔹 Processing video: {video_name}")

        # Setup output folders
        frames_output_folder = os.path.join("data", "frames", video_name)
        os.makedirs(frames_output_folder, exist_ok=True)
        base_output = os.path.join("results", self.folder_name, video_name)
        captions_output_folder = os.path.join(base_output, "captions")
        subtitled_output_folder = os.path.join(base_output, "subtitled")
        os.makedirs(captions_output_folder, exist_ok=True)
        if self.generate_subtitles:
            os.makedirs(subtitled_output_folder, exist_ok=True)

        # Step 1: Extract 1 frame per second
        print("🔹 Extracting frames...")
        extract_frames(video_path, frames_output_folder, interval=1)

        # Step 2: Detect scenes based on hash and motion
        print("\n🔹 Detecting scenes...")
        scenes = self.scene_detector.detect(video_path, frames_output_folder)

        # Step 3: Get video-level embedding for thematic alignment
        print("\n🔹 Extracting global CLIP embedding...")
        clip_video_embedding = extract_clip_video_embedding(frames_output_folder)

        # Step 4: Generate captions per scene using motion-aware frame selection
        print("\n🔹 Generating scene captions...")
        frame_files = sorted(os.listdir(frames_output_folder), key=lambda x: int(re.findall(r'\d+', x)[0]))
        scene_captions = []
        for scene in scenes:
            caption = self.frame_selector.select_best_caption(scene, frame_files, frames_output_folder, clip_video_embedding)
            scene_captions.append({
                "scene_id": scene["scene_id"],
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "caption": caption
            })
            print(f"Scene {scene['scene_id']} [{scene['start_time']}s - {scene['end_time']}s]: {caption}")

        # Step 5: Merge scenes and clean overlaps
        print("\n🔹 Merging scenes...")
        scene_captions = self.outlier_filter.filter_by_lof(scene_captions)
        scene_captions = self.scene_merger.merge(scene_captions)
        scene_captions = fix_overlapping_scenes(scene_captions)
        scene_captions = sorted(scene_captions, key=lambda x: x["start_time"])

        # Step 6: Generate a video-level summary
        print("\n🔹 Generating summary caption...")
        summary = self.summary_generator.generate(scene_captions, clip_video_embedding, max_length=35)
        print(f"\n✅ Video Summary: {summary}")

        # Step 7: Save captions and summary
        self.save_results_as_json(video_name, summary, scene_captions, captions_output_folder)

        # Step 8: Optionally generate SRT and embed subtitles
        if self.generate_subtitles:
            srt_path = os.path.join(subtitled_output_folder, f"{video_name}.srt")
            generate_srt(scene_captions, srt_path)

            output_video = os.path.join(subtitled_output_folder, f"{video_name}_subtitled.mp4")
            ffmpeg_command = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"subtitles={shlex.quote(srt_path.replace(os.sep, '/'))}",
                "-c:a", "copy",
                output_video
            ]
            try:
                subprocess.run(ffmpeg_command, check=True)
                print(f"✅ Subtitled video saved: {output_video}")
            except subprocess.CalledProcessError as e:
                print("❌ FFmpeg error:", e)

    def run(self):
        """Run the full pipeline for all videos in folder."""
        try:
            video_paths = self.get_video_paths()
        except ValueError as e:
            print(e)
            return

        for video_path in video_paths:
            self.process_video(video_path)


if __name__ == "__main__":
    folder_name = input("🔹 Enter the folder name containing the videos: ").strip()
    generate_subtitles = input("🔹 Generate videos with embedded subtitles? (yes/no): ").strip().lower() == "yes"

    pipeline = VideoCaptioning(folder_name, generate_subtitles)
    pipeline.run()
