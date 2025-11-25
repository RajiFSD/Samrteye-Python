#!/usr/bin/env python3
"""
visitor_counter_inout.py

Option C + IN/OUT counting for front-view camera.

Requirements:
- OpenCV (cv2), numpy
- (Optional) ultralytics installed -> uses YOLOv8n (auto-download) for person detection
- Model files (already uploaded):
    /mnt/data/deploy.prototxt
    /mnt/data/res10_300x300_ssd_iter_140000_fp16.caffemodel
    /mnt/data/deploy_gender.prototxt
    /mnt/data/gender_net.caffemodel
- Input video: /mnt/data/video3.mp4

Output:
- out_inout_annotated.mp4 (annotated video)
- Prints IN/OUT counts per category
"""

import cv2
import numpy as np
import math
import time
from collections import Counter

# ---------------- CONFIG ----------------
INPUT_VIDEO = "video3.mp4"
OUTPUT_VIDEO = "out_inout_annotated.mp4"

# models (uploaded)
FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
GENDER_PROTO = "models/deploy_gender.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_LABELS = ["Male", "Female"]

# detection/tracking params
FACE_CONF_THRESHOLD = 0.5
MIN_FRAMES_TO_COUNT = 4
MAX_DISAPPEARED = 50
PROCESS_FPS = 6            # detections per second
KID_HEIGHT_FRAC = 0.45

# blue uniform detection (adjust if needed)
BLUE_HUE_MIN = 90
BLUE_HUE_MAX = 140
BLUE_SAT_MIN = 60
BLUE_VAL_MIN = 40
BLUE_TORSO_RATIO_THRESHOLD = 0.18

# counting line placement (fraction from top): 0..1
# for front-view gate, place near center or wherever the gate line is
COUNT_LINE_POS = 0.48  # 48% from top -> horizontal line y = COUNT_LINE_POS * frame_h

# Direction convention for front-view camera:
# "top_to_bottom" means centroid crossing from above line to below line => IN
# "bottom_to_top" means centroid crossing from below to above => IN
IN_DIRECTION = "top_to_bottom"  # change if your camera orientation is opposite

# thresholds for matching and scoring
IOU_SCORE_WEIGHT = 0.7
CENTROID_SCORE_WEIGHT = 0.3
MATCH_SCORE_THRESHOLD = 0.12

# ---------------- utilities ----------------
def clamp_box(box, w, h):
    x1,y1,x2,y2 = box
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(int(x2), w); y2 = min(int(y2), h)
    return (x1,y1,x2,y2)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

def centroid(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

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
    w = max(1, box[2] - box[0]); h = max(1, box[3] - box[1])
    aspect = w / float(h)
    return "male" if aspect > 0.48 else "female"

# ---------------- tracker classes ----------------
class Track:
    def __init__(self, tid, bbox, frame_no, is_employee=False, gender_guess=None, face_used=False):
        self.id = tid
        self.bbox = bbox
        self.first_frame = frame_no
        self.last_frame = frame_no
        self.frames_seen = 1
        self.disappeared = 0
        self.is_employee = is_employee
        self.gender_votes = []
        if gender_guess is not None:
            self.gender_votes.append(gender_guess)
        self.face_used = face_used
        self.counted = False
        self.initial_centroid = centroid(bbox)
        self.final_centroid = self.initial_centroid
        self.direction = None  # will be 'in' or 'out' when decided

class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}  # id -> Track

    def update(self, detections, frame_no):
        # detections: list of dict{'bbox','is_employee','gender_guess','face_used'}
        if len(self.tracks) == 0:
            for d in detections:
                t = Track(self.next_id, d['bbox'], frame_no, d.get('is_employee', False), d.get('gender_guess'), d.get('face_used', False))
                self.tracks[self.next_id] = t
                self.next_id += 1
            return

        track_items = list(self.tracks.items())  # (tid, track)
        used_tracks = set()
        used_dets = set()
        scores = []
        for ti, (tid, tr) in enumerate(track_items):
            for di, d in enumerate(detections):
                sc_iou = iou(tr.bbox, d['bbox'])
                c_old = centroid(tr.bbox); c_new = centroid(d['bbox'])
                dist = math.hypot(c_old[0] - c_new[0], c_old[1] - c_new[1])
                diag = math.hypot(max(1, tr.bbox[2]-tr.bbox[0]), max(1, tr.bbox[3]-tr.bbox[1]))
                nd = min(1.0, dist/(diag+1e-6))
                score = sc_iou * IOU_SCORE_WEIGHT + (1-nd) * CENTROID_SCORE_WEIGHT
                scores.append((score, ti, di))
        scores.sort(reverse=True, key=lambda x: x[0])

        mapping = {}
        for score, ti, di in scores:
            if score < MATCH_SCORE_THRESHOLD:
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
            tr.final_centroid = centroid(d['bbox'])

        # unmatched tracks -> disappeared++
        for tid, tr in self.tracks.items():
            if tid not in used_tracks:
                tr.disappeared += 1

        # remove long disappeared
        to_remove = [tid for tid, tr in self.tracks.items() if tr.disappeared > MAX_DISAPPEARED]
        for tid in to_remove:
            del self.tracks[tid]

        # create new tracks for unmatched detections
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

# ---------------- load models ----------------
print("Loading face & gender models from disk...")
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
print("Models loaded.")

# try YOLO (ultralytics) first
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("Ultralytics available: will use YOLOv8n for person detection if possible.")
except Exception:
    print("Ultralytics not available — falling back to HOG person detector.")

hog = None
if not YOLO_AVAILABLE:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ---------------- processing loop ----------------
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit("Cannot open input video: " + INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

tracker = SimpleTracker()

# counters for IN/OUT
in_total = 0; out_total = 0
in_male = 0; in_female = 0; in_kid = 0
out_male = 0; out_female = 0; out_kid = 0
excluded_employees = 0

frame_no = 0
process_every = max(1, int(fps // PROCESS_FPS))

print("Starting main loop...")
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    # draw counting line on frame and use in overlay
    line_y = int(frame_h * COUNT_LINE_POS)

    # sample frames for detection
    do_detect = (frame_no % process_every == 0)

    detections = []
    if do_detect:
        # person detection
        if YOLO_AVAILABLE:
            try:
                y = YOLO("yolov8n.pt")  # will auto-download if needed
                results = y(frame, imgsz=640)[0]
                boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results, "boxes") else np.array([])
                scores = results.boxes.conf.cpu().numpy() if hasattr(results, "boxes") else np.array([])
                classes = results.boxes.cls.cpu().numpy() if hasattr(results, "boxes") else np.array([])
                for i, b in enumerate(boxes):
                    cls = int(classes[i]) if len(classes)>0 else 0
                    conf = float(scores[i]) if len(scores)>0 else 0.5
                    if cls != 0 or conf < 0.3:
                        continue
                    x1,y1,x2,y2 = map(int, b)
                    x1,y1,x2,y2 = clamp_box((x1,y1,x2,y2), frame_w, frame_h)
                    detections.append({'bbox':(x1,y1,x2,y2)})
            except Exception:
                # fallback to hog if YOLO call fails
                YOLO_AVAILABLE = False

        if not YOLO_AVAILABLE:
            rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
            for (x,y,w,h), wt in zip(rects, weights):
                if float(wt) < 0.5:
                    continue
                x1,y1,x2,y2 = int(x), int(y), int(x+w), int(y+h)
                x1,y1,x2,y2 = clamp_box((x1,y1,x2,y2), frame_w, frame_h)
                detections.append({'bbox':(x1,y1,x2,y2)})

        # face detection global
        face_blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,117,123), swapRB=False, crop=False)
        face_net.setInput(face_blob)
        face_out = face_net.forward()
        face_boxes = []
        for i in range(face_out.shape[2]):
            conf = float(face_out[0,0,i,2])
            if conf < FACE_CONF_THRESHOLD:
                continue
            box = face_out[0,0,i,3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            fx1,fy1,fx2,fy2 = box.astype(int)
            fx1,fy1,fx2,fy2 = clamp_box((fx1,fy1,fx2,fy2), frame_w, frame_h)
            face_boxes.append((fx1,fy1,fx2,fy2))

        # associate faces -> people & get gender
        for det in detections:
            pb = det['bbox']
            best_i = -1; best_iou = 0.0
            for i, fb in enumerate(face_boxes):
                val = iou(pb, fb)
                if val > best_iou:
                    best_iou = val; best_i = i
            det['face_used'] = False
            det['is_employee'] = False
            det['gender_guess'] = None
            if best_i >= 0 and best_iou > 0.02:
                fx1,fy1,fx2,fy2 = face_boxes[best_i]
                face_crop = frame[fy1:fy2, fx1:fx2].copy()
                if face_crop.size > 0:
                    fb_blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227,227), (78.426,87.769,114.896), swapRB=False)
                    gender_net.setInput(fb_blob)
                    preds = gender_net.forward()
                    idx = int(np.argmax(preds[0]))
                    gender_str = GENDER_LABELS[idx].lower()
                    det['gender_guess'] = 'kid' if is_kid_by_height(det['bbox'], frame_h) else ('male' if gender_str.startswith('m') else 'female')
                    det['face_used'] = True

            if det['gender_guess'] is None:
                det['gender_guess'] = 'kid' if is_kid_by_height(det['bbox'], frame_h) else heuristic_gender_from_bbox(det['bbox'])

            # torso check for blue uniform
            x1,y1,x2,y2 = det['bbox']
            w = x2-x1; h = y2-y1
            torso_y1 = y1 + int(0.12 * h); torso_y2 = y1 + int(0.55 * h)
            torso_x1 = x1 + int(0.12 * w); torso_x2 = x2 - int(0.12 * w)
            torso_x1, torso_x2 = max(0, torso_x1), min(frame_w, torso_x2)
            torso_y1, torso_y2 = max(0, torso_y1), min(frame_h, torso_y2)
            torso = frame[torso_y1:torso_y2, torso_x1:torso_x2].copy()
            det['is_employee'] = is_blue_torso(torso)

    # update tracker
    tracker.update(detections if do_detect else [], frame_no)

    # check eligible tracks and decide IN/OUT when counting
    for tid, tr in tracker.eligible_to_count():
        # majority vote for final_type
        votes = tr.gender_votes
        if not votes:
            final_type = 'unknown'
        else:
            cnt = Counter(votes)
            final_type = cnt.most_common(1)[0][0]

        # determine motion direction relative to counting line
        init_y = tr.initial_centroid[1]
        last_y = tr.final_centroid[1]
        # crossing check: require that track has actually crossed the line between initial and final
        crossed = (init_y < line_y and last_y >= line_y) or (init_y >= line_y and last_y < line_y)

        direction_decided = None
        if crossed:
            # movement direction: positive delta = moved downward (top->bottom)
            dy = last_y - init_y
            if IN_DIRECTION == "top_to_bottom":
                if dy > 0:
                    direction_decided = "in"
                elif dy < 0:
                    direction_decided = "out"
            else:  # bottom_to_top
                if dy < 0:
                    direction_decided = "in"
                elif dy > 0:
                    direction_decided = "out"

        # If not crossed, you may still want to infer from last vs line (optional)
        # We'll only count when crossed to reduce false positives
        tr.counted = True  # mark counted either way to avoid repeated attempts

        if tr.is_employee:
            excluded_employees += 1
            continue

        if direction_decided == "in":
            in_total += 1
            if final_type == 'kid':
                in_kid += 1
            elif final_type == 'male':
                in_male += 1
            elif final_type == 'female':
                in_female += 1
            else:
                pass
        elif direction_decided == "out":
            out_total += 1
            if final_type == 'kid':
                out_kid += 1
            elif final_type == 'male':
                out_male += 1
            elif final_type == 'female':
                out_female += 1
            else:
                pass
        else:
            # Not crossed line — we skip counting to be safe (or you can enable fallback)
            pass

    # annotate
    vis = frame.copy()
    # draw counting line
    cv2.line(vis, (0, line_y), (frame_w, line_y), (0,0,255), 2)
    cv2.putText(vis, "Counting line", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    for tid, tr in tracker.tracks.items():
        x1,y1,x2,y2 = map(int, tr.bbox)
        color = (0,200,255) if not tr.is_employee else (100,100,200)
        label = f"ID:{tid}"
        if tr.face_used:
            label += " (F)"
        if tr.counted:
            label += " ✓"
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        # draw initial and final centroid
        icx, icy = map(int, tr.initial_centroid)
        fcx, fcy = map(int, tr.final_centroid)
        cv2.circle(vis, (icx, icy), 3, (0,255,0), -1)
        cv2.circle(vis, (fcx, fcy), 3, (255,0,0), -1)

    # overlay counts
    overlay = f"IN T:{in_total} (M:{in_male} W:{in_female} K:{in_kid})  OUT T:{out_total} (M:{out_male} W:{out_female} K:{out_kid})  ExclEmp:{excluded_employees}"
    cv2.putText(vis, overlay, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10,10,200), 2, cv2.LINE_AA)

    out.write(vis)

    if frame_no % (process_every * 40) == 0:
        print(f"Frame {frame_no}/{total_frames}  active_tracks={len(tracker.tracks)}  IN={in_total} OUT={out_total}")

# finish
cap.release()
out.release()
elapsed = time.time() - start_time
print("Done in {:.1f}s".format(elapsed))
print("FINAL:")
print("IN total:", in_total, " IN male:", in_male, " IN female:", in_female, " IN kids:", in_kid)
print("OUT total:", out_total, " OUT male:", out_male, " OUT female:", out_female, " OUT kids:", out_kid)
print("Excluded employees (blue coat):", excluded_employees)
print("Annotated video saved to:", OUTPUT_VIDEO)
