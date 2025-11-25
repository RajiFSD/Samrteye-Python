import cv2
import numpy as np
from collections import defaultdict, deque

class SimplePeopleCounter:
    def __init__(self):
        # Use OpenCV's built-in background subtractor
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        
        # Counting variables
        self.entered_count = 0
        self.exited_count = 0
        self.counting_line_y = None
        
        # Track history
        self.track_history = defaultdict(lambda: deque(maxlen=20))
        
    def process_frame(self, frame):
        height, width = frame.shape[:2]
        
        if self.counting_line_y is None:
            self.counting_line_y = height // 2
        
        # Resize for performance
        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]
        
        # Apply background subtraction
        fg_mask = self.backSub.apply(frame)
        
        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_objects = {}
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                current_objects[i] = (center_x, center_y, x, y, w, h)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Track and count
        for obj_id, (cx, cy, x, y, w, h) in current_objects.items():
            history = self.track_history[obj_id]
            
            if len(history) >= 2:
                prev_cy = history[-1][1]
                curr_cy = cy
                
                # Entering
                if prev_cy < self.counting_line_y and curr_cy >= self.counting_line_y:
                    self.entered_count += 1
                    print(f"ENTER: Total entered: {self.entered_count}")
                
                # Exiting
                elif prev_cy > self.counting_line_y and curr_cy <= self.counting_line_y:
                    self.exited_count += 1
                    print(f"EXIT: Total exited: {self.exited_count}")
            
            history.append((cx, cy))
        
        # Clean old tracks
        active_ids = list(current_objects.keys())
        for obj_id in list(self.track_history.keys()):
            if obj_id not in active_ids:
                del self.track_history[obj_id]
        
        # Draw UI
        cv2.line(frame, (0, self.counting_line_y), (width, self.counting_line_y), (0, 255, 0), 3)
        cv2.putText(frame, f"Entered: {self.entered_count} | Exited: {self.exited_count} | Inside: {max(0, self.entered_count - self.exited_count)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, f"Entered: {self.entered_count} | Exited: {self.exited_count} | Inside: {max(0, self.entered_count - self.exited_count)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

# Usage for simple version
cap = cv2.VideoCapture(0)
counter = SimplePeopleCounter()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed = counter.process_frame(frame)
    cv2.imshow('Simple People Counter', processed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()