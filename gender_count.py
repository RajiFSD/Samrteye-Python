import cv2
import numpy as np

# ==== Load Models ====

# Face detector (SSD)
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Gender classifier
gender_net = cv2.dnn.readNetFromCaffe(
    "deploy_gender.prototxt",
    "gender_net.caffemodel"
)
gender_list = ["Male", "Female"]

# ==== Video Input ====

video_path = "video3.mp4"   # your downloaded video
cap = cv2.VideoCapture(video_path)

male_count = 0
female_count = 0

# List of embeddings for uniqueness
unique_faces = []

def get_face_embedding(face_img):
    """Simple 32x32 flattened embedding."""
    resized = cv2.resize(face_img, (32, 32))
    return resized.flatten()

def is_same_face(e1, e2, threshold=3000):
    """Euclidean distance based similarity."""
    return np.linalg.norm(e1 - e2) < threshold


frame_num = 0
frame_skip = 5   # process 1 in every 5 frames for speed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    if frame_num % frame_skip != 0:
        continue

    h, w = frame.shape[:2]

    # ==== FACE DETECTION ====
    # blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
    #                              (104, 117, 123), False, False)
    blob = cv2.dnn.blobFromImage(
        face_img,
        scalefactor=1.0,
        size=(227, 227),
        mean=(0, 0, 0),   # IMPORTANT
        swapRB=True,      # IMPORTANT
        crop=False
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        # Clamp coords
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue

        # ==== UNIQUENESS CHECK ====
        emb = get_face_embedding(face_img)

        is_new_face = True
        for saved_emb in unique_faces:
            if is_same_face(emb, saved_emb):
                is_new_face = False
                break

        if not is_new_face:
            continue

        # Register this face
        unique_faces.append(emb)

        # ==== GENDER PREDICTION ====
        face_blob = cv2.dnn.blobFromImage(
            face_img,
            1.0,
            (227, 227),
            (78.426, 87.769, 114.896),
            swapRB=False
        )
        gender_net.setInput(face_blob)
        predictions = gender_net.forward()
        gender = gender_list[predictions[0].argmax()]

        if gender == "Male":
            male_count += 1
        else:
            female_count += 1

        # ==== DRAW BOUNDING BOX ====
        color = (255, 0, 0) if gender == "Male" else (255, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, gender, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show live preview (optional)
    cv2.imshow("Gender Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("===================================")
print(f"Unique detected Males:   {male_count}")
print(f"Unique detected Females: {female_count}")
print("===================================")
