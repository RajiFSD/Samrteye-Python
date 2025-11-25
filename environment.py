# For better accuracy in your specific location:
class TemplePeopleCounter(PeopleCounter):
    def __init__(self):
        super().__init__()
        # Adjust for temple/mall lighting
        self.face_confidence = 0.6  # Lower if poor lighting
        self.gender_confidence = 0.6  # Adjust threshold
        
    def is_labor(self, frame, bbox):
        # Add specific rules for your location
        x, y, w, h = bbox
        # Check for specific uniform patterns or badges
        # Add logo detection if uniforms have logos
        return super().detect_uniform_color(frame, bbox)