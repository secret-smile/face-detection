import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image, ImageEnhance
import time
import torch
from torchvision.ops import nms


# Load  YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [0]  # Only detect people
    return model

def detect_faces(image, model, conf_threshold=0.5, iou_threshold=0.4):
    results = model(image)
    boxes = results.xyxy[0].cpu().numpy()
    confidences = boxes[:, 4]
    boxes = boxes[:, :4]
    
    keep = nms(torch.tensor(boxes), torch.tensor(confidences), iou_threshold)
    boxes = boxes[keep]
    confidences = confidences[keep]
    
    faces = []
    for box, confidence in zip(boxes, confidences):
        if confidence > conf_threshold:
            x1, y1, x2, y2 = map(int, box[:4])
            faces.append({
                'box': (x1, y1, x2, y2),
                'confidence': float(confidence)
            })
    
    return faces

def draw_faces(image, faces):
    for i, face in enumerate(faces):
        (x1, y1, x2, y2) = face['box']
        confidence = face['confidence']
        color = (0, 255 - 30*i, 0)  # Varying shades of green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"Face {i+1}: {confidence:.2f}%"
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return image

def process_batch(frames, model, conf_threshold, iou_threshold):
    batch_results = model(frames)
    processed_frames = []
    face_counts = []
    
    for i, result in enumerate(batch_results.xyxy):
        faces = []
        boxes = result.cpu().numpy()
        confidences = boxes[:, 4]
        boxes = boxes[:, :4]
        
        keep = nms(torch.tensor(boxes), torch.tensor(confidences), iou_threshold)
        boxes = boxes[keep]
        confidences = confidences[keep]
        
        for box, confidence in zip(boxes, confidences):
            if confidence > conf_threshold:
                x1, y1, x2, y2 = map(int, box[:4])
                faces.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': float(confidence)
                })
        
        processed_frame = draw_faces(frames[i], faces)
        processed_frames.append(processed_frame)
        face_counts.append(len(faces))
    
    return processed_frames, face_counts

def process_video(video_path, model, conf_threshold, iou_threshold, batch_size=4, frame_skip=2):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        output = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        face_count = 0
        processed_count = 0
        start_time = time.time()
        
        frames_batch = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            if processed_count % frame_skip == 0:
                frames_batch.append(frame)
            
            if len(frames_batch) == batch_size:
                processed_frames, batch_face_counts = process_batch(frames_batch, model, conf_threshold, iou_threshold)
                for processed_frame in processed_frames:
                    output.write(processed_frame)
                face_count += sum(batch_face_counts)
                frames_batch = []
            
            processed_count += 1
        
        if frames_batch:
            processed_frames, batch_face_counts = process_batch(frames_batch, model, conf_threshold, iou_threshold)
            for processed_frame in processed_frames:
                output.write(processed_frame)
            face_count += sum(batch_face_counts)
        
        end_time = time.time()
        processing_time = end_time - start_time
        actual_fps = processed_count / processing_time
        
        output.release()
        video.release()
        
        return temp_file.name, frame_count, face_count, actual_fps

def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    img = Image.fromarray(image)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return np.array(img)

def main():
    st.title("Efficient Multi-Face Detection App")
    st.write("Upload an image or video to detect multiple faces efficiently.")

    model = load_model()
    if model is None:
        return

    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.4, 0.01)

    file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if file is not None:
        file_extension = file.name.split(".")[-1].lower()

        if file_extension in ["jpg", "jpeg", "png"]:
            image = Image.open(file)
            image = np.array(image)
            
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image enhancement options
            st.sidebar.header("Image Enhancement")
            brightness = st.sidebar.slider("Brightness", 0.5, 1.5, 1.0, 0.1)
            contrast = st.sidebar.slider("Contrast", 0.5, 1.5, 1.0, 0.1)
            sharpness = st.sidebar.slider("Sharpness", 0.5, 1.5, 1.0, 0.1)
            
            if st.button("Detect Faces"):
                enhanced_image = enhance_image(image, brightness, contrast, sharpness)
                faces = detect_faces(enhanced_image, model, conf_threshold, iou_threshold)
                result = draw_faces(enhanced_image.copy(), faces)
                st.image(result, caption="Result", use_column_width=True)
                st.write(f"Number of faces detected: {len(faces)}")
                
                for i, face in enumerate(faces):
                    st.write(f"Face {i+1}:")
                    st.write(f"  Confidence: {face['confidence']:.2f}%")
                    st.write(f"  Bounding Box: {face['box']}")

        elif file_extension in ["mp4", "avi"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tfile:
                tfile.write(file.read())
                video_path = tfile.name

            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    output_file, frame_count, face_count, fps = process_video(video_path, model, conf_threshold, iou_threshold)
                
                st.video(output_file)
                st.write(f"Processed {frame_count} frames")
                st.write(f"Detected a total of {face_count} faces")
                st.write(f"Average number of faces per frame: {face_count / frame_count:.2f}")
                st.write(f"Average FPS: {fps:.2f}")

if __name__ == "__main__":
    main()