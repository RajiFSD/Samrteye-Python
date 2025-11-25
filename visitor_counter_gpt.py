#!/usr/bin/env python3 given by ChatGPT for our video
"""
visitor_counter_option_c.py

Option C: Face-first gender detection with body fallback + tracking.
Uses local OpenCV DNN models (uploaded to /models) for face & gender.
Falls back to YOLO (if ultralytics installed) or OpenCV HOG person detector.
Counts unique visitors (men/women/kids), excludes employees wearing blue coat,
and writes an annotated output video.

Files expected (already uploaded):
 - /mnt/data/deploy.prototxt
 - /mnt/data/res10_300x300_ssd_iter_140000_fp16.caffemodel
 - /mnt/data/deploy_gender.prototxt
 - /mnt/data/gender_net.caffemodel
 - /mnt/data/video3.mp4 (input)

Outputs:
 - out_annotated_option_c.mp4 (annotated video)
 - prints final summary counts
"""

import cv2
import numpy as np
import math
import time
from collections import Counter

# ---------- Config ----------
INPUT_VIDEO = "video3.mp4"
OUTPUT_VIDEO = "out_annotated_option_c.mp4"

# model paths (already uploaded)
FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
GENDER_PROTO = "models/deploy_gender.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_LABELS = ["Male", "Female"]

# person detector fallback: try ultralytics YOLO if available, else HOG
USE_YOLO_IF_AVAILABLE = True

# thresholds
FACE_CONF_THRESHOLD = 0.5
PERSON_MIN_CONF_HOG = 0.5
MIN_FRAMES_TO_COUNT = 5         # track must be seen this many frames before eligible to count
MAX_DISAPPEARED = 50            # frames before a track is removed
PROCESS_FPS = 6                 # how many frames per second to run detection
KID_HEIGHT_FRAC = 0.45          # bbox height fraction of frame -> kid if <= this
BLUE_HUE_MIN = 90               # blue uniform hue range
BLUE_HUE_MAX = 140
BLUE_SAT_MIN = 60
BLUE_VAL_MIN = 40
BLUE_TORSO_RATIO_THRESHOLD = 0.18

# -------------------------------

# ---------- Utilities ----------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

def centroid(box):
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

def clamp_box(box, w, h):
    x1,y1,x2,y2 = box
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(int(x2), w); y2 = min(int(y2), h)
    return (x1,y1,x2,y2)

def is_blue_torso(torso_bgr):
    if torso_bgr is None or torso_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([BLUE_HUE_MIN, BLUE_SAT_MIN, BLUE_VAL_MIN])
    upper = np.array([BLUE_HUE_MAX, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    blue_ratio = (mask > 0).sum() / float(mask.size)
    return blue_ratio >= BLUE_TORSO_RATIO_THRESHOLD

def is_kid_by_height(box, frame_h):
    h = box[3] - box[1]
    return (h / float(frame_h)) <= KID_HEIGHT_FRAC

def heuristic_gender_from_bbox(box):
    w = max(1, box[2]-box[0]); h = max(1, box[3]-box[1])
    aspect = w/float(h)
    # tuned heuristic: broader shoulder (higher width/height) -> male
    return "male" if aspect > 0.48 else "female"

# ---------- Tracker ----------
class Track:
    def __init__(self, tid, bbox, first_frame, is_employee=False, gender_guess=None, face_used=False):
        self.id = tid
        self.bbox = bbox
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.frames_seen = 1
        self.disappeared = 0
        self.is_employee = is_employee
        self.gender_votes = []  # list of strings 'male'/'female'/'kid'
        if gender_guess is not None:
            self.gender_votes.append(gender_guess)
        self.face_used = face_used
        self.counted = False
        self.initial_centroid = centroid(bbox)

class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}  # id -> Track

    def update(self, detections, frame_no):
        # detections: list of dict {bbox, is_employee, gender_guess, face_used}
        if len(self.tracks) == 0:
            for d in detections:
                t = Track(self.next_id, d['bbox'], frame_no, d.get('is_employee', False), d.get('gender_guess'), d.get('face_used', False))
                self.tracks[self.next_id] = t
                self.next_id += 1
            return

        # matching by IoU + centroid proximity
        track_items = list(self.tracks.items())  # (tid, track)
        used_tracks = set()
        used_dets = set()
        scores = []
        for ti, (tid, tr) in enumerate(track_items):
            for di, d in enumerate(detections):
                sc_iou = iou(tr.bbox, d['bbox'])
                c_old = centroid(tr.bbox); c_new = centroid(d['bbox'])
                dist = math.hypot(c_old[0]-c_new[0], c_old[1]-c_new[1])
                diag = math.hypot(max(1, tr.bbox[2]-tr.bbox[0]), max(1, tr.bbox[3]-tr.bbox[1]))
                nd = min(1.0, dist/(diag+1e-6))
                score = sc_iou*0.7 + (1-nd)*0.3
                scores.append((score, ti, di))
        scores.sort(reverse=True, key=lambda x: x[0])

        mapping = {}
        for score, ti, di in scores:
            if score < 0.12:
                break
            tid, tr = track_items[ti]
            if tid in used_tracks or di in used_dets:
                continue
            mapping[di] = tid
            used_tracks.add(tid); used_dets.add(di)

        # update matched
        for di, tid in mapping.items():
            d = detections[di]
            tr = self.tracks[tid]
            tr.bbox = d['bbox']
            tr.last_frame = frame_no
            tr.frames_seen += 1
            tr.disappeared = 0
            tr.is_employee = tr.is_employee or d.get('is_employee', False)
            if 'gender_guess' in d:
                tr.gender_votes.append(d['gender_guess'])
            tr.face_used = tr.face_used or d.get('face_used', False)

        # unmatched tracks -> disappeared++
        for tid, tr in self.tracks.items():
            if tid not in used_tracks:
                tr.disappeared += 1

        # remove long disappeared
        to_remove = [tid for tid, tr in self.tracks.items() if tr.disappeared > MAX_DISAPPEARED]
        for tid in to_remove:
            del self.tracks[tid]

        # create tracks for unmatched detections
        for di, d in enumerate(detections):
            if di not in mapping:
                tr = Track(self.next_id, d['bbox'], frame_no, d.get('is_employee', False), d.get('gender_guess'), d.get('face_used', False))
                self.tracks[self.next_id] = tr
                self.next_id += 1

    def eligible_to_count(self):
        res = []
        for tid, tr in self.tracks.items():
            if tr.counted:
                continue
            if tr.is_employee:
                continue
            if tr.frames_seen >= MIN_FRAMES_TO_COUNT:
                res.append((tid, tr))
        return res

# ---------- Load models ----------
print("Loading face & gender models from disk...")
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
print("OK")

# try to import ultralytics YOLO; if not available we will use HOG
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("Ultralytics YOLO available - will use YOLO person detector if present")
except Exception:
    YOLO_AVAILABLE = False
    print("Ultralytics not available - falling back to HOG person detector")

# prepare HOG only if YOLO not available
hog = None
if not YOLO_AVAILABLE:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ---------- Processing loop ----------
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit("Cannot open video: " + INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

# prepare writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

tracker = SimpleTracker()

# counters
total_count = 0
male_count = 0
female_count = 0
kid_count = 0
excluded_employees = 0

frame_no = 0
process_every = max(1, int(fps // PROCESS_FPS))

print("Starting processing...")
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret: 
        break
    frame_no += 1

    # only run detection on sampled frames (speed)
    if frame_no % process_every != 0:
        # still optionally annotate active tracks
        vis = frame.copy()
        for tid, tr in tracker.tracks.items():
            x1,y1,x2,y2 = map(int, tr.bbox)
            color = (0,180,255) if not tr.is_employee else (200,100,50)
            label = f"ID:{tid}"
            if tr.face_used:
                label += " (F)"
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            cv2.putText(vis, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        # overlay current totals
        txt = f"Total:{total_count}  Men:{male_count}  Women:{female_count}  Kids:{kid_count}  ExclEmp:{excluded_employees}"
        cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,200), 2, cv2.LINE_AA)
        out.write(vis)
        continue

    # ---------- Person detection ----------
    detections = []  # list of dicts
    if YOLO_AVAILABLE:
        try:
            # use small YOLOv8 model from ultralytics (will auto-download if needed)
            y = YOLO("yolov8n.pt")
            results = y(frame, imgsz=640)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results, "boxes") else np.array([])
            scores = results.boxes.conf.cpu().numpy() if hasattr(results, "boxes") else np.array([])
            classes = results.boxes.cls.cpu().numpy() if hasattr(results, "boxes") else np.array([])
            for i, b in enumerate(boxes):
                cls = int(classes[i]) if len(classes)>0 else 0
                conf = float(scores[i]) if len(scores)>0 else 0.5
                # COCO class 0 is person
                if cls != 0 or conf < 0.3:
                    continue
                x1,y1,x2,y2 = map(int, b)
                x1,y1,x2,y2 = clamp_box((x1,y1,x2,y2), frame_w, frame_h)
                detections.append({'bbox':(x1,y1,x2,y2)})
        except Exception as e:
            # fallback to HOG if YOLO invocation failed
            YOLO_AVAILABLE = False

    if not YOLO_AVAILABLE:
        rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
        for (x,y,w,h), wt in zip(rects, weights):
            if float(wt) < PERSON_MIN_CONF_HOG:
                continue
            x1,y1,x2,y2 = int(x), int(y), int(x+w), int(y+h)
            x1,y1,x2,y2 = clamp_box((x1,y1,x2,y2), frame_w, frame_h)
            detections.append({'bbox':(x1,y1,x2,y2)})

    # ---------- Face detection (global) ----------
    face_detections = []
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,117,123), swapRB=False, crop=False)
    face_net.setInput(blob)
    face_out = face_net.forward()
    for i in range(face_out.shape[2]):
        conf = float(face_out[0,0,i,2])
        if conf < FACE_CONF_THRESHOLD:
            continue
        box = face_out[0,0,i,3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
        x1,y1,x2,y2 = box.astype(int)
        x1,y1,x2,y2 = clamp_box((x1,y1,x2,y2), frame_w, frame_h)
        face_detections.append({'bbox':(x1,y1,x2,y2), 'conf':conf})

    # ---------- Associate faces to person boxes (if overlapping) ----------
    # For each person detection, attempt to find a face inside it (by IoU with face box).
    for det in detections:
        pb = det['bbox']
        best_face = None
        best_iou = 0.0
        for f in face_detections:
            i = iou(pb, f['bbox'])
            if i > best_iou:
                best_iou = i
                best_face = f
        # if good overlap found, crop face and predict gender
        det['face_used'] = False
        det['is_employee'] = False
        det['gender_guess'] = None
        if best_face and best_iou > 0.02:
            fx1,fy1,fx2,fy2 = best_face['bbox']
            face_img = frame[fy1:fy2, fx1:fx2].copy()
            if face_img.size > 0:
                # gender classification
                face_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227,227), (78.426,87.769,114.896), swapRB=False)
                gender_net.setInput(face_blob)
                preds = gender_net.forward()
                g_idx = int(np.argmax(preds[0]))
                gender_str = GENDER_LABELS[g_idx].lower()
                # normalize outputs to 'male'/'female'
                det['gender_guess'] = 'kid' if is_kid_by_height(det['bbox'], frame_h) else ('male' if gender_str.startswith('m') else 'female')
                det['face_used'] = True

        # if no face or no confident gender, fallback to body heuristics
        if det['gender_guess'] is None:
            det['gender_guess'] = 'kid' if is_kid_by_height(det['bbox'], frame_h) else heuristic_gender_from_bbox(det['bbox'])

        # detect torso and check for blue uniform
        x1,y1,x2,y2 = det['bbox']
        w = x2-x1; h = y2-y1
        torso_y1 = y1 + int(0.12 * h)
        torso_y2 = y1 + int(0.55 * h)
        torso_x1 = x1 + int(0.12 * w)
        torso_x2 = x2 - int(0.12 * w)
        torso_x1, torso_x2 = max(0, torso_x1), min(frame_w, torso_x2)
        torso_y1, torso_y2 = max(0, torso_y1), min(frame_h, torso_y2)
        torso = frame[torso_y1:torso_y2, torso_x1:torso_x2].copy()
        det['is_employee'] = is_blue_torso(torso)

    # ---------- Update tracker ----------
    tracker.update(detections, frame_no)

    # ---------- Count eligible tracks ----------
    for tid, tr in tracker.eligible_to_count():
        # decide final type by majority votes
        votes = tr.gender_votes
        if not votes:
            final_type = 'unknown'
        else:
            cnt = Counter(votes)
            final_type = cnt.most_common(1)[0][0]
        tr.counted = True

        if tr.is_employee:
            excluded_employees += 1
            continue
        # update counters
        if final_type == 'kid':
            kid_count += 1
        elif final_type == 'male':
            male_count += 1
        elif final_type == 'female':
            female_count += 1
        else:
            # unknown -> count as adult (best effort)
            male_count += 0
        total_count += 1

    # ---------- Annotate & write frame ----------
    vis = frame.copy()
    for tid, tr in tracker.tracks.items():
        x1,y1,x2,y2 = map(int, tr.bbox)
        color = (0,140,255) if not tr.is_employee else (0,120,200)
        label = f"ID:{tid}"
        if tr.face_used:
            label += " (face)"
        if tr.counted:
            label += " ✓"
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # totals overlay
    txt = f"Total:{total_count}  Men:{male_count}  Women:{female_count}  Kids:{kid_count}  ExclEmp:{excluded_employees}"
    cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,200), 2, cv2.LINE_AA)

    out.write(vis)

    # progress log
    if frame_no % (process_every * 50) == 0:
        print(f"Frame {frame_no}/{total_frames}  active_tracks={len(tracker.tracks)}  counted={total_count}")

# finished
cap.release()
out.release()
elapsed = time.time() - start_time
print("Processing finished in {:.1f}s".format(elapsed))
print("Final counts:")
print(" Total visitors:", total_count)
print(" Men:", male_count)
print(" Women:", female_count)
print(" Kids:", kid_count)
print(" Excluded employees (blue coat):", excluded_employees)
print("Annotated video saved to:", OUTPUT_VIDEO)
