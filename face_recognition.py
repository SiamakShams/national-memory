import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

# Initialize both models
# buffalo_l: High Precision
app_l = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_l.prepare(ctx_id=0, det_size=(640, 640))

# buffalo_s: High Sensitivity (The blurry/small image savior)
app_s = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app_s.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(img, model_type="l"):
    """Extracts embedding using either the Large or Small model."""
    target_app = app_l if model_type == "l" else app_s
    
    # Set threshold based on model type
    target_app.models['detection'].det_thresh = 0.4 if model_type == "l" else 0.15
    
    faces = target_app.get(img)
    
    # If using small model and it failed, try the enhancement trick
    if not faces and model_type == "s":
        img_enhanced = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img_enhanced = cv2.resize(img_enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        faces = target_app.get(img_enhanced)
        
    if not faces:
        return None
        
    # Return the largest face
    return sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1].normed_embedding

def match_faces(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None or img2 is None: return "File Error"

    # --- STEP 1: Attempt High Precision Match (buffalo_l) ---
    feat1_l = get_embedding(img1, "l")
    feat2_l = get_embedding(img2, "l")
    
    if feat1_l is not None and feat2_l is not None:
        score = float(np.dot(feat1_l, feat2_l))
        return f"Model: High-Grade | Score: {score:.4f}"

    # --- STEP 2: Fallback to Small Model (buffalo_s) if High-Grade fails ---
    # We re-run BOTH in 's' mode to ensure the math is compatible
    feat1_s = get_embedding(img1, "s")
    feat2_s = get_embedding(img2, "s")
    
    if feat1_s is not None and feat2_s is not None:
        score = float(np.dot(feat1_s, feat2_s))
        return f"Model: Recovery (Small) | Score: {score:.4f}"

    return "MATCH FAILED: Face could not be isolated in one of the images."

def find_face_in_crowd(target_path, crowd_path, threshold=0.4):
    """
    Finds a target face in a crowd, saves all individual faces,
    and creates a result image highlighting the best match.
    """
    output_folder = "extracted_faces"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    target_img = cv2.imread(target_path)
    crowd_img = cv2.imread(crowd_path)
    
    if target_img is None or crowd_img is None:
        return "File Error"

    # 1. Get the target fingerprint using your existing cascading logic
    target_feat = get_embedding(target_img, "l")
    current_model = "l"
    
    if target_feat is None:
        target_feat = get_embedding(target_img, "s")
        current_model = "s"
        
    if target_feat is None:
        return "Target face could not be isolated."

    # 2. Detect ALL faces in the crowd using the chosen model
    target_app = app_l if current_model == "l" else app_s
    crowd_faces = target_app.get(crowd_img)

    if not crowd_faces:
        return f"No faces detected in crowd using {current_model} model."

    print(f"Detected {len(crowd_faces)} faces. Extracting and searching...")

    best_match = None
    max_score = -1
    crowd_filename = os.path.splitext(os.path.basename(crowd_path))[0]
    
    # Create a copy of the crowd image for the visual highlight
    vis_img = crowd_img.copy()

    # 3. Iterate through all faces, save them, and find the best match
    for i, face in enumerate(crowd_faces):
        bbox = face.bbox.astype(int)
        y1, y2 = max(0, bbox[1]), min(crowd_img.shape[0], bbox[3])
        x1, x2 = max(0, bbox[0]), min(crowd_img.shape[1], bbox[2])
        face_chip = crowd_img[y1:y2, x1:x2]

        # Save individual face crop
        face_filename = f"{crowd_filename}_face_{i}.jpg"
        cv2.imwrite(os.path.join(output_folder, face_filename), face_chip)

        # Compare for matching
        feat = face.normed_embedding
        score = float(np.dot(target_feat, feat))
        
        if score > max_score:
            max_score = score
            best_match = face

    # 4. Final Result & Visual Marking
    if max_score >= threshold and best_match is not None:
        # Draw green box for the match
        b = best_match.bbox.astype(int)
        cv2.rectangle(vis_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
        cv2.putText(vis_img, f"MATCH: {max_score:.2f}", (b[0], b[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the result overview
        result_name = f"MATCH_RESULT_{crowd_filename}.jpg"
        cv2.imwrite(result_name, vis_img)
        
        return f"MATCH FOUND! Score: {max_score:.4f} | Saved {len(crowd_faces)} faces | Result: {result_name}"
    else:
        return f"No match found. Saved {len(crowd_faces)} faces. Best score: {max_score:.4f}"

def extract_faces(image_path):
    """
    Extracts all faces from a single image and saves them to the 'faces' folder.
    """
    output_folder = "faces"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img = cv2.imread(image_path)
    if img is None:
        return "File Error"

    # Use the high sensitivity model to find more faces
    target_app = app_s
    faces = target_app.get(img)

    if not faces:
        return "No faces detected in the image."

    print(f"Detected {len(faces)} faces. Extracting...")

    image_filename = os.path.splitext(os.path.basename(image_path))[0]

    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        y1, y2 = max(0, bbox[1]), min(img.shape[0], bbox[3])
        x1, x2 = max(0, bbox[0]), min(img.shape[1], bbox[2])
        face_chip = img[y1:y2, x1:x2]

        # Save individual face crop
        face_filename = f"{image_filename}_face_{i}.jpg"
        cv2.imwrite(os.path.join(output_folder, face_filename), face_chip)

    return f"Saved {len(faces)} faces to the '{output_folder}' folder."

def get_face_boxes(image_path, model_type="l"):
    img = cv2.imread(image_path)
    if img is None:
        return None, []
    
    target_app = app_l if model_type == "l" else app_s
    faces = target_app.get(img)
    boxes = [face.bbox.astype(int) for face in faces] if faces else []
    return img, boxes

def save_face(img, box, filename_base, index, output_folder="faces"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    y1, y2 = max(0, box[1]), min(img.shape[0], box[3])
    x1, x2 = max(0, box[0]), min(img.shape[1], box[2])
    face_img = img[y1:y2, x1:x2]
    save_path = os.path.join(output_folder, f"{filename_base}_face_{index}.jpg")
    cv2.imwrite(save_path, face_img)

qdrant_client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "faces"

def ensure_collection_exists():
    try:
        collections = qdrant_client.get_collections()
        if not any(c.name == COLLECTION_NAME for c in collections.collections):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
            )
    except Exception as e:
        print(f"Qdrant Error: {e}")

def index_extracted_face(img, box, original_filepath, face_index):
    try:
        # Crop the face
        y1, y2 = max(0, box[1]), min(img.shape[0], box[3])
        x1, x2 = max(0, box[0]), min(img.shape[1], box[2])
        face_img = img[y1:y2, x1:x2]
        
        # Get embedding
        embedding = get_embedding(face_img, "l")
        if embedding is None:
            embedding = get_embedding(face_img, "s")
            
        if embedding is not None:
            ensure_collection_exists()
            
            metadata = {
                "original_filename": os.path.basename(original_filepath),
                "full_path": original_filepath,
                "face_index": face_index,
                "bbox": [int(b) for b in box]
            }
            
            point_id = str(uuid.uuid4())
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=metadata
                    )
                ]
            )
            return True
    except Exception as e:
        print(f"Failed to index face: {e}")
    return False

def configure_qdrant(url, port):
    global qdrant_client
    try:
        qdrant_client = QdrantClient(url=url, port=int(port))
    except Exception as e:
        print(f"Failed to configure Qdrant client: {e}")