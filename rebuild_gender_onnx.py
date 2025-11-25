import base64
import os

# --- EDIT THIS to the folder you chose ---
OUT_DIR = r"F:\SmartEye\Samrteye-Python\models"
OUT_NAME = "gender_classifier.onnx"
OUT_PATH = os.path.join(OUT_DIR, OUT_NAME)

# --- Paste the base64 chunks below as Python strings ---
# Example:
# chunks = [
#   "BASE64_CHUNK_001",
#   "BASE64_CHUNK_002",
#   ...
# ]
chunks = [
    # <<< PASTE CHUNK STRINGS HERE, in order, each as a Python string >>>
]

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

print("Reassembling", OUT_PATH)
with open(OUT_PATH, "wb") as fout:
    for i, c in enumerate(chunks, start=1):
        print(f" Writing chunk {i}/{len(chunks)}")
        b = base64.b64decode(c.encode("ascii"))
        fout.write(b)

print("Done. Saved ONNX to:", OUT_PATH)
print("File size (bytes):", os.path.getsize(OUT_PATH))
