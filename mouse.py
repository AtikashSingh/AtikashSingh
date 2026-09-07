import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time
import pygame
import os

# --- Configuration ---
CAM_WIDTH, CAM_HEIGHT = 640, 480
FRAME_REDUCTION = 100 # ROI padding from edges of camera frame
SMOOTHING = 5 # Higher is smoother but slightly laggy. Lower is faster but jittery.

# Hand bone connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Wrist to pinky base
]

# --- Setup ---
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False # Disable failsafe so mouse can reach edges

# Sound Setup
pygame.mixer.init()
sound_path = os.path.join("sounds", "sound.mp3")
try:
    sound_effect = pygame.mixer.Sound(sound_path)
except Exception as e:
    print(f"Warning: Could not load sound file at {sound_path}. Please make sure you added the file! Error: {e}")
    sound_effect = None

last_sound_time = 0

# State variables for smoothing
plocX, plocY = 0, 0
clocX, clocY = 0, 0

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return np.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Global variable to store the latest result from the async callback
latest_result = None

def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Configure MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=result_callback
)

print("Virtual Mouse Started. Press 'q' in the camera window to quit.")

# Initialize the detector
with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            break
        
        # Flip the image horizontally for intuitive mirror-like movement
        img = cv2.flip(img, 1)
        
        # Draw Region of Interest (ROI) rectangle
        cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION), 
                      (CAM_WIDTH - FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (255, 0, 255), 2)
        
        # Convert image to MediaPipe format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Send frame to landmarker
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        
        # Process the latest available result
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                h, w, c = img.shape
                
                # --- NEW FEATURE: Draw Proper Hand Bones ---
                for connection in HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    x_start = int(hand_landmarks[start_idx].x * w)
                    y_start = int(hand_landmarks[start_idx].y * h)
                    x_end = int(hand_landmarks[end_idx].x * w)
                    y_end = int(hand_landmarks[end_idx].y * h)
                    
                    # Draw bone lines
                    cv2.line(img, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)
                    # Draw joints
                    cv2.circle(img, (x_start, y_start), 4, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x_end, y_end), 4, (0, 0, 255), cv2.FILLED)
                
                # Get coordinates for logic
                index_tip = hand_landmarks[8]
                thumb_tip = hand_landmarks[4]
                middle_tip = hand_landmarks[12]
                wrist = hand_landmarks[0]
                index_base = hand_landmarks[5]
                
                x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
                x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x_middle, y_middle = int(middle_tip.x * w), int(middle_tip.y * h)
                x_wrist, y_wrist = int(wrist.x * w), int(wrist.y * h)
                x_base, y_base = int(index_base.x * w), int(index_base.y * h)
                
                # --- NEW FEATURE: Extreme Accuracy Dynamic Scaling ---
                # Calculate how big the hand is relative to the camera
                hand_scale = calculate_distance((x_wrist, y_wrist), (x_base, y_base))
                dynamic_threshold = hand_scale * 0.35 # Scales accurately no matter how far hand is!
                
                index_thumb_dist = calculate_distance((x_index, y_index), (x_thumb, y_thumb))
                middle_thumb_dist = calculate_distance((x_middle, y_middle), (x_thumb, y_thumb))
                
                # Check for Sound Trigger (Middle finger touching Thumb)
                if middle_thumb_dist < dynamic_threshold:
                    current_time = time.time()
                    # 1 second cooldown so it doesn't spam the sound
                    if current_time - last_sound_time > 1.0:
                        if sound_effect:
                            sound_effect.play()
                        last_sound_time = current_time
                        
                        # Visual indicator
                        cv2.putText(img, "SOUND PLAYED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                
                # --- MOUSE MOVEMENT ---
                # Constrain values so they stay within the ROI bounding box
                x1_constrained = np.clip(x_index, FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION)
                y1_constrained = np.clip(y_index, FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION)

                # Convert coordinates from ROI to Screen Resolution
                screen_x = np.interp(x1_constrained, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, SCREEN_WIDTH))
                screen_y = np.interp(y1_constrained, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, SCREEN_HEIGHT))
                
                # Smoothing
                clocX = plocX + (screen_x - plocX) / SMOOTHING
                clocY = plocY + (screen_y - plocY) / SMOOTHING
                
                # Move Mouse
                try:
                    pyautogui.moveTo(clocX, clocY)
                except Exception as e:
                     pass # Catch edge cases
                
                plocX, plocY = clocX, clocY
                
                # --- MOUSE CLICK (Index finger touches thumb) ---
                if index_thumb_dist < dynamic_threshold:
                    cv2.circle(img, (x_index, y_index), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()
                    # Small delay to prevent rapid multiple clicks
                    time.sleep(0.2)

        cv2.imshow("Virtual Mouse Tracker", img)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
