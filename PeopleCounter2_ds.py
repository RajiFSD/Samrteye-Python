import cv2
import numpy as np
from collections import defaultdict, deque
import time
import os

class PeopleCounter:
    def __init__(self):
        # Load YOLO for person detection (more reliable)
        print("Loading YOLO model...")
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Load COCO names (for YOLO)
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Person class ID in COCO
        self.person_class_id = self.classes.index("person")
        
        # Simple gender detection using face aspect ratio and clothing colors
        self.entered_count = {'Male': 0, 'Female': 0}
        self.exited_count = {'Male': 0, 'Female': 0}
        
        # Tracking
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.counting_line_y = None
        
        # Labor uniform colors
        self.labor_colors = {
            'blue': ([90, 50, 50], [130, 255, 255]),
            'orange': ([5, 50, 50], [15, 255, 255]),
            'gray': ([0, 0, 0], [180, 50, 100])
        }

    def detect_people_yolo(self, frame):
        """Detect people using YOLO"""
        height, width = frame.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id == self.person_class_id and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                final_boxes.append((x, y, w, h))
                
        return final_boxes

    def detect_uniform(self, frame, bbox):
        """Detect labor uniform based on color"""
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
            
            if color_ratio > 0.3:
                return True
                
        return False

    def estimate_gender_simple(self, frame, bbox):
        """Simple gender estimation based on body proportions and colors"""
        x, y, w, h = bbox
        
        # Calculate aspect ratio (width/height)
        aspect_ratio = w / h
        
        # Analyze clothing colors in different body regions
        upper_body = frame[y:min(y + h//3, frame.shape[0]), x:min(x + w, frame.shape[1])]
        lower_body = frame[min(y + 2*h//3, frame.shape[0]):min(y + h, frame.shape[0]), 
                          x:min(x + w, frame.shape[1])]
        
        if upper_body.size == 0 or lower_body.size == 0:
            return "Unknown", 0.5
            
        # Convert to HSV for color analysis
        upper_hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
        lower_hsv = cv2.cvtColor(lower_body, COLOR_BGR2HSV)
        
        # Common color ranges for male/female clothing
        male_colors = [
            ([0, 0, 0], [180, 50, 100]),    # Dark colors
            ([100, 50, 50], [130, 255, 255]) # Blue colors
        ]
        
        female_colors = [
            ([0, 50, 50], [10, 255, 255]),  # Red/Pink colors
            ([140, 50, 50], [180, 255, 255]) # Purple colors
        ]
        
        male_score = 0
        female_score = 0
        
        # Check upper body colors
        for lower, upper in male_colors:
            mask = cv2.inRange(upper_hsv, np.array(lower), np.array(upper))
            male_score += np.sum(mask > 0) / mask.size
            
        for lower, upper in female_colors:
            mask = cv2.inRange(upper_hsv, np.array(lower), np.array(upper))
            female_score += np.sum(mask > 0) / mask.size
        
        # Aspect ratio bias (males tend to be broader)
        if aspect_ratio > 0.4:  # Wider frame
            male_score += 0.2
        else:  # Narrower frame
            female_score += 0.2
            
        total = male_score + female_score
        if total == 0:
            return "Unknown", 0.5
            
        male_confidence = male_score / total
        female_confidence = female_score / total
        
        if male_confidence > female_confidence:
            return "Male", male_confidence
        else:
            return "Female", female_confidence

    def track_and_count(self, frame, detections):
        """Track detections and count crossings"""
        height, width = frame.shape[:2]
        
        if self.counting_line_y is None:
            self.counting_line_y = height // 2
            
        current_centers = {}
        
        # Simple centroid tracking
        for i, (x, y, w, h) in enumerate(detections):
            center_x = x + w // 2
            center_y = y + h // 2
            current_centers[i] = (center_x, center_y, x, y, w, h)
            
            # Draw detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Update tracking history and count crossings
        for obj_id, (cx, cy, x, y, w, h) in current_centers.items():
            history = self.track_history[obj_id]
            
            if len(history) >= 2:
                prev_cy = history[-1][1]
                curr_cy = cy
                
                # Entering (from top)
                if prev_cy < self.counting_line_y and curr_cy >= self.counting_line_y:
                    gender, confidence = self.estimate_gender_simple(frame, (x, y, w, h))
                    if gender != "Unknown" and confidence > 0.6:
                        if not self.detect_uniform(frame, (x, y, w, h)):
                            self.entered_count[gender] += 1
                            print(f"ENTER: {gender} (Confidence: {confidence:.2f})")
                            
                # Exiting (to top)
                elif prev_cy > self.counting_line_y and curr_cy <= self.counting_line_y:
                    gender, confidence = self.estimate_gender_simple(frame, (x, y, w, h))
                    if gender != "Unknown" and confidence > 0.6:
                        if not self.detect_uniform(frame, (x, y, w, h)):
                            self.exited_count[gender] += 1
                            print(f"EXIT: {gender} (Confidence: {confidence:.2f})")
            
            history.append((cx, cy))
        
        # Clean up old tracks
        active_ids = list(current_centers.keys())
        for obj_id in list(self.track_history.keys()):
            if obj_id not in active_ids:
                del self.track_history[obj_id]

    def draw_ui(self, frame):
        """Draw user interface"""
        height, width = frame.shape[:2]
        
        # Draw counting line
        cv2.line(frame, (0, self.counting_line_y), (width, self.counting_line_y), (0, 255, 0), 3)
        cv2.putText(frame, "COUNTING LINE", (10, self.counting_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw statistics
        net_male = max(0, self.entered_count['Male'] - self.exited_count['Male'])
        net_female = max(0, self.entered_count['Female'] - self.exited_count['Female'])
        
        stats = [
            f"ENTERED - M: {self.entered_count['Male']} F: {self.entered_count['Female']}",
            f"EXITED - M: {self.exited_count['Male']} F: {self.exited_count['Female']}",
            f"NET INSIDE - M: {net_male} F: {net_female}",
            f"TOTAL INSIDE: {net_male + net_female}"
        ]
        
        for i, text in enumerate(stats):
            y_pos = 30 + i * 30
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_frame(self, frame):
        """Process a single frame"""
        # Resize for better performance
        frame = cv2.resize(frame, (640, 480))
        
        # Detect people
        detections = self.detect_people_yolo(frame)
        
        # Track and count
        self.track_and_count(frame, detections)
        
        # Draw UI
        self.draw_ui(frame)
        
        return frame

def download_yolo_files():
    """Download YOLO files if not present"""
    import urllib.request
    import ssl
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    files = {
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"✓ {filename} downloaded")
            except Exception as e:
                print(f"✗ Error downloading {filename}: {e}")
                return False
    return True

def main():
    print("People Counter for Temple/Mall Entrance")
    print("=" * 50)
    
    # Download YOLO files if needed
    if not download_yolo_files():
        print("Failed to download required files. Using webcam without YOLO...")
        # You can implement a fallback method here
    
    # Initialize counter
    counter = PeopleCounter()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reset counts")
    print("  'c' - Change counting line position")
    print("\nStarting people counter...")
    
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
        elif key == ord('c'):
            # Change counting line position with mouse
            print("Click on the frame to set counting line position")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 50)
    print("FINAL STATISTICS:")
    print(f"Total Entered - Male: {counter.entered_count['Male']}, Female: {counter.entered_count['Female']}")
    print(f"Total Exited - Male: {counter.exited_count['Male']}, Female: {counter.exited_count['Female']}")
    net_male = counter.entered_count['Male'] - counter.exited_count['Male']
    net_female = counter.entered_count['Female'] - counter.exited_count['Female']
    print(f"Currently Inside - Male: {net_male}, Female: {net_female}")
    print(f"Total People Inside: {net_male + net_female}")

if __name__ == "__main__":
    main()