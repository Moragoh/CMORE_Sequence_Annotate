import cv2
import numpy as np
import pandas as pd
import os
import sys
import argparse
import time

# --- DEPENDENCY CHECK ---
try:
    from ultralytics import YOLO
except ImportError:
    print("CRITICAL ERROR: 'ultralytics' is not installed.")
    print("Run: pip install ultralytics opencv-python pandas torch")
    sys.exit(1)

# --- IMPORT LOCAL MODULE ---
try:
    from keypoint_detector import BoxDetector
except ImportError:
    print("CRITICAL ERROR: Could not import 'BoxDetector' from 'keypoint_detector.py'")
    print("Ensure 'keypoint_detector.py' is in the same directory as this script.")
    sys.exit(1)

# --- CONSTANTS ---
SHRINK_FACTOR = 0.9
TEXT_COLOR = (0, 255, 0)       # Green
WARN_COLOR = (0, 0, 255)       # Red
ACTIVE_COLOR = (0, 165, 255)   # Orange
FONT = cv2.FONT_HERSHEY_SIMPLEX
DETECTION_INTERVAL = 10        # Run detection every N frames
DEFAULT_MODEL_PATH = "./keypoint_detector.pt" # Model path is now fixed

class GroundTruthAnnotator:
    # --- MODIFIED __init__ ---
    def __init__(self, video_path, output_csv, handedness, start_frame=0):
        self.video_path = video_path
        self.output_csv = output_csv
        
        print(f"[INFO] Opening video: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Loading Model: {DEFAULT_MODEL_PATH}")
        try:
            # Use the hardcoded model path
            self.box_detector = BoxDetector(DEFAULT_MODEL_PATH)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(1)
            
        self.latest_keypoints = None 
        
        # State Variables
        self.current_frame_idx = start_frame
        self.success_sequences = [] 
        self.active_start = None   
        self.feedback_msg = ""     
        self.feedback_timer = 0    
        
        # Handedness is now set from the constructor
        self.handedness = handedness 
        print(f"[INFO] Handedness set to: {self.handedness}")

        if os.path.exists(output_csv):
            print(f"[INFO] Found existing CSV. Note: Will be overwritten on save.")
            pass

    def format_time(self, frame_idx):
        if self.fps <= 0: return "00:00"
        seconds = frame_idx / self.fps
        m, s = divmod(seconds, 60)
        return f"{int(m):02d}:{int(s):02d}"

    # --- GEOMETRY LOGIC ---
    def shrink_centroid(self, points, factor: float):
        centroid = np.mean(points, axis=0)
        vectors = points - centroid
        new_points = centroid + vectors * factor
        return new_points.astype(np.int32)

    def get_contour_points(self, keypoints, handedness):
        if keypoints is None: return None
        try:
            # Linear Interpolation for Back Wall
            denom = (keypoints['Back top left'][0] - keypoints['Back top right'][0])
            if denom == 0: denom = 0.001 
            
            m = (keypoints['Back top left'][1] - keypoints['Back top right'][1]) / denom
            c = keypoints['Back top left'][1] - m * keypoints['Back top left'][0]
            back_top_middle_y = m * keypoints['Back divider top'][0] + c
            
            # Select points based on Handedness
            if handedness == 'Left':
                points = np.array([
                    keypoints['Back top left'],
                    [min(keypoints['Back divider top'][0], keypoints['Front divider top'][0]), back_top_middle_y],
                    keypoints['Front top middle'],
                    keypoints['Front top left']
                ], dtype=np.int32)
            else: # Right
                points = np.array([
                    [max(keypoints['Back divider top'][0], keypoints['Front divider top'][0]), back_top_middle_y],
                    keypoints['Back top right'],
                    keypoints['Front top right'],
                    keypoints['Front top middle']
                ], dtype=np.int32)

            return self.shrink_centroid(points, SHRINK_FACTOR)
        except Exception:
            return None

    # --- LOGIC ---
    def check_rewind_deletions(self):
        # 1. Rewound past Active Start
        if self.active_start is not None and self.current_frame_idx < self.active_start:
            removed = self.active_start
            self.active_start = None
            self.feedback_msg = f"REWIND: Deleted 'Start' at {removed}"
            self.feedback_timer = 30

        # 2. Rewound into a completed sequence
        if self.success_sequences:
            last_start, last_stop = self.success_sequences[-1]
            if self.current_frame_idx < last_stop:
                self.success_sequences.pop()
                self.active_start = last_start
                self.feedback_msg = f"REWIND: Re-opened 'Start' at {last_start}"
                self.feedback_timer = 30
                self.check_rewind_deletions() # Recursive check

    # --- MODIFIED run ---
    def run(self):
        print("\n--- CONTROLS ---")
        print(" [k] : Next Frame")
        print(" [j] : Previous Frame")
        print(" [1] : Mark Success START")
        print(" [2] : Mark Success STOP")
        print(" [q] : Save & Quit")
        print("----------------\n")
        
        video_ended = False
        
        try:
            while True:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                ret, frame = self.cap.read()
                
                if not ret:
                    video_ended = True
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    video_ended = False

                if not video_ended:
                    if self.latest_keypoints is None or (self.current_frame_idx % DETECTION_INTERVAL == 0):
                        success, result = self.box_detector.detect(frame)
                        if success:
                            self.latest_keypoints = result

                self.check_rewind_deletions()

                # 1. Contour
                contour = self.get_contour_points(self.latest_keypoints, self.handedness)
                if contour is not None and not video_ended:
                    pts = contour.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 255))
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

                # 2. UI Overlay
                font_scale = max(0.6, self.width / 1500) 
                line_spacing = int(40 * font_scale)
                
                info_lines = [
                    f"Frame: {self.current_frame_idx} / {self.total_frames}",
                    f"Time: {self.format_time(self.current_frame_idx)}",
                    f"Handedness: {self.handedness} (Set at launch)",
                    "Controls: [j] Prev, [k] Next", # Added controls
                ]
                
                if self.active_start is not None:
                    info_lines.append(f"STATUS: ATTEMPT STARTED ({self.active_start})")
                else:
                    info_lines.append("STATUS: WAITING [1]")
                    
                if self.success_sequences:
                    last = self.success_sequences[-1]
                    info_lines.append(f"Last Saved: {last[0]} -> {last[1]}")

                if video_ended:
                    info_lines.append("END OF VIDEO. Press [q] to save.")

                # Create a semi-transparent overlay for the UI box
                overlay = frame.copy()
                box_height = int(len(info_lines) * line_spacing + 40) # Adjusted padding
                box_width = int(self.width * 0.45) # Made box slightly wider for new text
                cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
                
                # Blend the overlay with the frame
                alpha = 0.5 # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                for i, line in enumerate(info_lines):
                    color = TEXT_COLOR
                    if "ATTEMPT STARTED" in line: color = ACTIVE_COLOR
                    if "END OF VIDEO" in line: color = WARN_COLOR
                    # Draw text with thickness 1
                    cv2.putText(frame, line, (10, 40 + i * line_spacing), FONT, font_scale, color, 1) # Thickness changed to 1

                if self.feedback_timer > 0:
                    # Changed thickness to 1
                    cv2.putText(frame, self.feedback_msg, (10, int(self.height - 50)), FONT, font_scale, WARN_COLOR, 1)
                    self.feedback_timer -= 1

                display_frame = frame
                if self.width > 1920:
                    display_frame = cv2.resize(frame, (1280, 720))

                cv2.imshow("Annotator", display_frame)

                # --- INPUT ---
                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    print("Saving and Exiting...")
                    break
                elif key == ord('k'):
                    if not video_ended: self.current_frame_idx += 1
                elif key == ord('j'):
                    if self.current_frame_idx > 0: self.current_frame_idx -= 1
                elif key == ord('1'):
                    if self.active_start is None:
                        self.active_start = self.current_frame_idx
                        self.feedback_msg = "Start Marked"
                        self.feedback_timer = 20
                    else:
                        self.feedback_msg = "Error: Already Started"
                        self.feedback_timer = 20
                elif key == ord('2'):
                    if self.active_start is not None:
                        if self.current_frame_idx > self.active_start:
                            self.success_sequences.append((self.active_start, self.current_frame_idx))
                            self.active_start = None
                            self.feedback_msg = "Sequence Saved"
                            self.feedback_timer = 20
                        else:
                            self.feedback_msg = "Error: Stop < Start"
                            self.feedback_timer = 20
                    else:
                        self.feedback_msg = "Error: No Start Marked"
                        self.feedback_timer = 20
                # 'h' key logic removed

        finally:
            self.save_csv()
            self.cap.release()
            cv2.destroyAllWindows()

    def save_csv(self):
        if not self.success_sequences:
            print("[INFO] No sequences recorded.")
            return
        
        try:
            df = pd.DataFrame(self.success_sequences, columns=['Start Frame', 'End Frame'])
            df.to_csv(self.output_csv, index=False)
            print(f"[SUCCESS] Saved {len(self.success_sequences)} annotations to {self.output_csv}")
        except Exception as e:
            print(f"[ERROR] Could not save CSV: {e}")
            print("DUMPING DATA TO CONSOLE:")
            print(self.success_sequences)

# --- MODIFIED __main__ ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground Truth Video Annotator")
    parser.add_argument("--video", required=True, help="Path to the video file")
    # --model argument removed
    parser.add_argument("--output", help="Optional: Output CSV filename. Defaults to [video_name]_sequence_annotations.csv")
    
    # --- Handedness Flag Group ---
    hand_group = parser.add_mutually_exclusive_group(required=True)
    hand_group.add_argument("--R", "--right", action="store_true", help="Set handedness to Right")
    hand_group.add_argument("--L", "--left", action="store_true", help="Set handedness to Left")
    # --- End Handedness Group ---

    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        sys.exit(1)
    
    # Check for the hardcoded model path
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"Error: Model file not found at '{DEFAULT_MODEL_PATH}'")
        print("Please ensure 'best_model.pt' is in the same directory as the script.")
        sys.exit(1)

    # 1. Determine Handedness
    handedness = "Right" if args.R else "Left"

    # 2. Determine Output Filename
    if args.output:
        output_filename = args.output
    else:
        video_basename = os.path.basename(args.video)
        video_name_no_ext = os.path.splitext(video_basename)[0]
        output_filename = f"{video_name_no_ext}_sequence_annotations.csv"
    
    print(f"[INFO] Annotations will be saved to: {output_filename}")

    # 3. Pass handedness to the constructor (model_path removed)
    annotator = GroundTruthAnnotator(
        video_path=args.video, 
        output_csv=output_filename,
        handedness=handedness
    )
    annotator.run()