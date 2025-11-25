# given by deepseek for camera feed at temple/mall entrance
import cv2
import numpy as np
import dlib
from collections import defaultdict, deque
import time
import os

class PeopleCounter:
    def __init__(self):
        # Load models
        self.face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        self.gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
        
        # Gender classification labels
        self.GENDER_LIST = ['Male', 'Female']
        
        # Tracking variables
        self.trackers = []
        self.tracked_objects = {}
        self.object_ids = 0
        self.entered_count = {'Male': 0, 'Female': 0}
        self.exited_count = {'Male': 0, 'Female': 0}
        
        # Line for counting (horizontal line across the frame)
        self.counting_line_y = None
        
        # Labor uniform detection (color-based)
        self.labor_colors = {
            'blue': ([100, 150, 0], [140, 255, 255]),  # Blue uniform range
            'orange': ([10, 100, 100], [25, 255, 255]), # Orange uniform range
            'gray': ([0, 0, 50], [180, 50, 200])       # Gray uniform range
        }
        
        # Object history for smoothing
        self.history = defaultdict(lambda: deque(maxlen=30))
        
    def detect_uniform_color(self, frame, bbox):
        """Detect if person is wearing labor uniform based on color"""
        x, y, w, h = bbox
        # Extract upper body region (where uniform is most visible)
        upper_body = frame[max(0, y):min(y + h//2, frame.shape[0]), 
                          max(0, x):min(x + w, frame.shape[1])]
        
        if upper_body.size == 0:
            return False
            
        hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
        
        for color_name, (lower, upper) in self.labor_colors.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            color_ratio = np.sum(mask > 0) / mask.size
            
            # If significant portion has uniform color, classify as labor
            if color_ratio > 0.3:  # 30% of upper body has uniform color
                return True
                
        return False
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                   (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x1, y1, x2-x1, y2-y1))
                
        return faces
    
    def predict_gender(self, face_roi):
        """Predict gender from face ROI"""
        try:
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227),
                                       (78.4263377603, 87.7689143744, 114.895847746),
                                       swapRB=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.GENDER_LIST[gender_preds[0].argmax()]
            confidence = gender_preds[0].max()
            return gender, confidence
        except:
            return "Unknown", 0.0
    
    def update_tracking(self, frame, faces):
        """Update object tracking"""
        current_objects = {}
        
        # Update existing trackers
        for tracker in self.trackers[:]:
            tracking_quality = tracker.update(frame)
            if tracking_quality >= 8:  # Good tracking
                pos = tracker.get_position()
                x1 = int(pos.left())
                y1 = int(pos.top())
                x2 = int(pos.right())
                y2 = int(pos.bottom())
                
                # Find the best matching face
                best_match_idx = -1
                best_iou = 0.3
                
                for i, (fx, fy, fw, fh) in enumerate(faces):
                    face_rect = (fx, fy, fx+fw, fy+fh)
                    tracker_rect = (x1, y1, x2, y2)
                    
                    # Calculate IoU
                    xi1 = max(face_rect[0], tracker_rect[0])
                    yi1 = max(face_rect[1], tracker_rect[1])
                    xi2 = min(face_rect[2], tracker_rect[2])
                    yi2 = min(face_rect[3], tracker_rect[3])
                    
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    box1_area = (face_rect[2] - face_rect[0]) * (face_rect[3] - face_rect[1])
                    box2_area = (tracker_rect[2] - tracker_rect[0]) * (tracker_rect[3] - tracker_rect[1])
                    iou = inter_area / float(box1_area + box2_area - inter_area)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                
                if best_match_idx >= 0:
                    obj_id = id(tracker)
                    current_objects[obj_id] = (x1, y1, x2, y2)
                    # Remove matched face
                    faces.pop(best_match_idx)
            else:
                # Remove poor quality tracker
                self.trackers.remove(tracker)
        
        # Add new trackers for unmatched faces
        for (x, y, w, h) in faces:
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x, y, x+w, y+h)
            tracker.start_track(frame, rect)
            self.trackers.append(tracker)
            obj_id = id(tracker)
            current_objects[obj_id] = (x, y, x+w, y+h)
        
        return current_objects
    
    def count_people(self, current_objects, frame):
        """Count people crossing the line and classify gender"""
        h, w = frame.shape[:2]
        if self.counting_line_y is None:
            self.counting_line_y = h // 2  # Middle of frame
        
        for obj_id, (x1, y1, x2, y2) in current_objects.items():
            center_y = (y1 + y2) // 2
            
            # Get object history
            history = self.history[obj_id]
            history.append(center_y)
            
            if len(history) >= 2:
                # Check if crossed the line
                prev_y = history[-2]
                curr_y = history[-1]
                
                # Entering (coming from top)
                if prev_y < self.counting_line_y and curr_y >= self.counting_line_y:
                    # Detect gender and check for uniform
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        gender, confidence = self.predict_gender(face_roi)
                        if gender != "Unknown" and confidence > 0.7:
                            # Check for labor uniform
                            if not self.detect_uniform_color(frame, (x1, y1, x2-x1, y2-y1)):
                                self.entered_count[gender] += 1
                                print(f"ENTER: {gender} (Confidence: {confidence:.2f})")
                
                # Exiting (going to top)
                elif prev_y > self.counting_line_y and curr_y <= self.counting_line_y:
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        gender, confidence = self.predict_gender(face_roi)
                        if gender != "Unknown" and confidence > 0.7:
                            if not self.detect_uniform_color(frame, (x1, y1, x2-x1, y2-y1)):
                                self.exited_count[gender] += 1
                                print(f"EXIT: {gender} (Confidence: {confidence:.2f})")
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Resize frame for faster processing
        frame = cv2.resize(frame, (800, 600))
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Update tracking
        current_objects = self.update_tracking(frame, faces)
        
        # Count people
        self.count_people(current_objects, frame)
        
        # Draw UI
        self.draw_ui(frame)
        
        return frame
    
    def draw_ui(self, frame):
        """Draw user interface"""
        h, w = frame.shape[:2]
        
        # Draw counting line
        cv2.line(frame, (0, self.counting_line_y), (w, self.counting_line_y), (0, 255, 0), 2)
        cv2.putText(frame, "COUNTING LINE", (10, self.counting_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw statistics
        y_offset = 30
        cv2.putText(frame, f"ENTERED - Male: {self.entered_count['Male']} Female: {self.entered_count['Female']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"EXITED - Male: {self.exited_count['Male']} Female: {self.exited_count['Female']}", 
                   (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"NET - Male: {self.entered_count['Male'] - self.exited_count['Male']} "
                          f"Female: {self.entered_count['Female'] - self.exited_count['Female']}", 
                   (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw current count
        cv2.putText(frame, f"Current Inside: Male: {max(0, self.entered_count['Male'] - self.exited_count['Male'])}, "
                          f"Female: {max(0, self.entered_count['Female'] - self.exited_count['Female'])}", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def main():
    # Initialize people counter
    counter = PeopleCounter()
    
    # Initialize camera (use 0 for default camera, or RTSP URL for CCTV)
    # cap = cv2.VideoCapture(0)  # For webcam
    # cap = cv2.VideoCapture("rtsp://your_camera_ip/stream")  # For CCTV
    
    # For testing with video file
    cap = cv2.VideoCapture("entrance_video.mp4")  # Replace with your video file
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Starting people counter...")
    print("Press 'q' to quit, 'r' to reset counts")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = counter.process_frame(frame)
        
        # Display
        cv2.imshow('People Counter - Temple/Mall Entrance', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.entered_count = {'Male': 0, 'Female': 0}
            counter.exited_count = {'Male': 0, 'Female': 0}
            print("Counts reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n=== FINAL STATISTICS ===")
    print(f"Total Entered - Male: {counter.entered_count['Male']}, Female: {counter.entered_count['Female']}")
    print(f"Total Exited - Male: {counter.exited_count['Male']}, Female: {counter.exited_count['Female']}")
    print(f"Currently Inside - Male: {max(0, counter.entered_count['Male'] - counter.exited_count['Male'])}, "
          f"Female: {max(0, counter.entered_count['Female'] - counter.exited_count['Female'])}")

if __name__ == "__main__":
    main()