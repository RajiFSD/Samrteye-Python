import cv2
import numpy as np
from collections import defaultdict, deque
import time
import os

class SimplePeopleCounter:
    def __init__(self):
        # Use OpenCV's background subtractor for motion detection
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Counting variables
        self.entered_count = {'Male': 0, 'Female': 0, 'Total': 0}
        self.exited_count = {'Male': 0, 'Female': 0, 'Total': 0}
        self.counting_line_y = None
        
        # Track history for object tracking
        self.track_history = defaultdict(lambda: deque(maxlen=15))
        self.next_object_id = 0
        
        # Labor uniform colors (adjust these based on your staff uniforms)
        self.labor_colors = {
            'blue': ([90, 50, 50], [130, 255, 255]),      # Blue uniforms
            'orange': ([5, 50, 50], [15, 255, 255]),      # Orange uniforms
            'gray': ([0, 0, 0], [180, 50, 100]),          # Gray uniforms
            'red': ([0, 50, 50], [10, 255, 255])          # Red uniforms
        }

    def detect_movement(self, frame):
        """Detect moving objects using background subtraction"""
        # Apply background subtraction
        fg_mask = self.backSub.apply(frame)
        
        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Filter by size (adjust based on camera distance)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (typical person proportions)
                aspect_ratio = h / w
                if 1.0 < aspect_ratio < 4.0:  # Person-like aspect ratio
                    detections.append((x, y, w, h))
                    
        return detections, fg_mask

    def is_labor_uniform(self, frame, bbox):
        """Check if the person is wearing labor uniform"""
        x, y, w, h = bbox
        
        # Focus on upper body for uniform detection
        upper_y = y + int(h * 0.1)
        upper_h = int(h * 0.4)
        
        upper_body = frame[upper_y:min(upper_y + upper_h, frame.shape[0]), 
                          x:min(x + w, frame.shape[1])]
        
        if upper_body.size == 0:
            return False
            
        hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
        
        for color_name, (lower, upper) in self.labor_colors.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            color_ratio = np.sum(mask > 0) / mask.size
            
            if color_ratio > 0.3:  # If 30% of upper body has uniform color
                return True
                
        return False

    def estimate_gender_simple(self, bbox):
        """Simple gender estimation based on body proportions"""
        x, y, w, h = bbox
        
        # Simple heuristic: males tend to have broader shoulders
        shoulder_ratio = w / h
        
        if shoulder_ratio > 0.45:  # Broader shoulders
            return "Male", min(0.8, shoulder_ratio * 1.5)
        else:  # Narrower shoulders
            return "Female", min(0.8, (1 - shoulder_ratio) * 1.5)

    def track_objects(self, detections):
        """Simple object tracking using centroid distance"""
        current_objects = {}
        used_detections = set()
        
        # For each existing track, find the closest detection
        for obj_id in list(self.track_history.keys()):
            if len(self.track_history[obj_id]) > 0:
                last_center = self.track_history[obj_id][-1]
                
                best_match_idx = -1
                min_distance = 50  # Maximum distance to match (adjust based on video resolution)
                
                for i, (x, y, w, h) in enumerate(detections):
                    if i in used_detections:
                        continue
                        
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    distance = np.sqrt((center_x - last_center[0])**2 + (center_y - last_center[1])**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match_idx = i
                
                if best_match_idx != -1:
                    x, y, w, h = detections[best_match_idx]
                    center_x = x + w // 2
                    center_y = y + h // 2
                    current_objects[obj_id] = (center_x, center_y, x, y, w, h)
                    used_detections.add(best_match_idx)
        
        # Create new tracks for unmatched detections
        for i, (x, y, w, h) in enumerate(detections):
            if i not in used_detections:
                center_x = x + w // 2
                center_y = y + h // 2
                obj_id = self.next_object_id
                self.next_object_id += 1
                current_objects[obj_id] = (center_x, center_y, x, y, w, h)
        
        return current_objects

    def count_crossings(self, frame, current_objects):
        """Count people crossing the line"""
        height, width = frame.shape[:2]
        
        if self.counting_line_y is None:
            self.counting_line_y = height // 2
        
        for obj_id, (cx, cy, x, y, w, h) in current_objects.items():
            history = self.track_history[obj_id]
            
            if len(history) >= 2:
                prev_cy = history[-1][1]
                curr_cy = cy
                
                # Entering (coming from top)
                if prev_cy < self.counting_line_y and curr_cy >= self.counting_line_y:
                    if not self.is_labor_uniform(frame, (x, y, w, h)):
                        gender, confidence = self.estimate_gender_simple((x, y, w, h))
                        self.entered_count[gender] += 1
                        self.entered_count['Total'] += 1
                        print(f"ENTER: {gender} (Confidence: {confidence:.2f}) - Total entered: {self.entered_count['Total']}")
                
                # Exiting (going to top)
                elif prev_cy > self.counting_line_y and curr_cy <= self.counting_line_y:
                    if not self.is_labor_uniform(frame, (x, y, w, h)):
                        gender, confidence = self.estimate_gender_simple((x, y, w, h))
                        self.exited_count[gender] += 1
                        self.exited_count['Total'] += 1
                        print(f"EXIT: {gender} (Confidence: {confidence:.2f}) - Total exited: {self.exited_count['Total']}")
            
            # Update history
            history.append((cx, cy))
        
        # Clean up old tracks (remove tracks without recent updates)
        current_ids = set(current_objects.keys())
        for obj_id in list(self.track_history.keys()):
            if obj_id not in current_ids:
                del self.track_history[obj_id]

    def draw_ui(self, frame, detections, current_objects):
        """Draw user interface"""
        height, width = frame.shape[:2]
        
        # Draw counting line
        cv2.line(frame, (0, self.counting_line_y), (width, self.counting_line_y), (0, 255, 0), 3)
        cv2.putText(frame, "COUNTING LINE", (10, self.counting_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw detections
        for obj_id, (cx, cy, x, y, w, h) in current_objects.items():
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            # Draw object ID
            cv2.putText(frame, f"ID: {obj_id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw statistics
        net_male = self.entered_count['Male'] - self.exited_count['Male']
        net_female = self.entered_count['Female'] - self.exited_count['Female']
        net_total = self.entered_count['Total'] - self.exited_count['Total']
        
        stats = [
            f"ENTERED - M: {self.entered_count['Male']} F: {self.entered_count['Female']} T: {self.entered_count['Total']}",
            f"EXITED - M: {self.exited_count['Male']} F: {self.exited_count['Female']} T: {self.exited_count['Total']}",
            f"NET INSIDE - M: {net_male} F: {net_female} T: {net_total}",
            f"Active Tracks: {len(current_objects)}"
        ]
        
        for i, text in enumerate(stats):
            y_pos = 30 + i * 25
            # Background for better readability
            cv2.rectangle(frame, (5, y_pos - 20), (width - 5, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def process_frame(self, frame):
        """Process a single frame"""
        # Resize for better performance
        frame = cv2.resize(frame, (640, 480))
        
        # Detect moving objects
        detections, fg_mask = self.detect_movement(frame)
        
        # Track objects
        current_objects = self.track_objects(detections)
        
        # Count crossings
        self.count_crossings(frame, current_objects)
        
        # Draw UI
        self.draw_ui(frame, detections, current_objects)
        
        return frame, fg_mask

def main():
    print("Simple People Counter for Temple/Mall Entrance")
    print("=" * 50)
    print("This version uses motion detection and doesn't require dlib")
    print("It will detect people based on movement and body proportions")
    
    # Initialize counter
    counter = SimplePeopleCounter()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        # Try different camera indices
        for i in range(1, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera found at index {i}")
                break
        else:
            print("No camera found. Please check your camera connection.")
            return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reset counts")
    print("  'l' - Toggle counting line position")
    print("  'm' - Toggle motion mask display")
    print("\nStarting people counter...")
    
    show_mask = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame
        processed_frame, fg_mask = counter.process_frame(frame)
        
        # Display
        if show_mask:
            # Resize mask to match frame size for display
            fg_mask_display = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Motion Mask', fg_mask_display)
        
        cv2.imshow('People Counter - Temple/Mall Entrance', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.entered_count = {'Male': 0, 'Female': 0, 'Total': 0}
            counter.exited_count = {'Male': 0, 'Female': 0, 'Total': 0}
            counter.track_history.clear()
            counter.next_object_id = 0
            print("Counts reset")
        elif key == ord('l'):
            # Toggle counting line between thirds
            height = processed_frame.shape[0]
            if counter.counting_line_y == height // 2:
                counter.counting_line_y = height // 3
            else:
                counter.counting_line_y = height // 2
            print(f"Counting line moved to y={counter.counting_line_y}")
        elif key == ord('m'):
            show_mask = not show_mask
            if show_mask:
                print("Motion mask display enabled")
            else:
                cv2.destroyWindow('Motion Mask')
                print("Motion mask display disabled")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 50)
    print("FINAL STATISTICS:")
    print(f"Total Entered - Male: {counter.entered_count['Male']}, Female: {counter.entered_count['Female']}, Total: {counter.entered_count['Total']}")
    print(f"Total Exited - Male: {counter.exited_count['Male']}, Female: {counter.exited_count['Female']}, Total: {counter.exited_count['Total']}")
    net_male = counter.entered_count['Male'] - counter.exited_count['Male']
    net_female = counter.entered_count['Female'] - counter.exited_count['Female']
    net_total = counter.entered_count['Total'] - counter.exited_count['Total']
    print(f"Currently Inside - Male: {net_male}, Female: {net_female}, Total: {net_total}")

if __name__ == "__main__":
    main()