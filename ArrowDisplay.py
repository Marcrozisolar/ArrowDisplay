import cv2
import numpy as np
from PIL import Image

def load_video(video_path):
    return cv2.VideoCapture(video_path)

def load_overlay(image_path, scale_factor=0.2):
    image = Image.open(image_path).convert("RGBA")
    image_array = np.array(image)
    image_bgra = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)
    
    new_size = (int(image_bgra.shape[1] * scale_factor), int(image_bgra.shape[0] * scale_factor))
    resized_image = cv2.resize(image_bgra, new_size)
    
    if resized_image.shape[2] == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2BGRA)
    
    return resized_image

def blend_overlay(frame, overlay, position):
    x, y = position
    overlay_bgr, alpha = cv2.split(overlay)[:3], cv2.split(overlay)[3].astype(float) / 255.0
    overlay_bgr = cv2.merge(overlay_bgr)
    
    h, w, _ = overlay_bgr.shape
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        print("Overlay exceeds frame boundaries. Adjust position or scale down the overlay.")
        return frame
    
    roi = frame[y:y+h, x:x+w].astype(float)
    overlay_bgr = overlay_bgr.astype(float)
    blended = (1 - alpha[..., None]) * roi + (alpha[..., None]) * overlay_bgr
    frame[y:y+h, x:x+w] = blended.astype(np.uint8)
    return frame

def main(video_path, image_path, position=(950, 50)):
    video = load_video(video_path)
    overlay = load_overlay(image_path)
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame = blend_overlay(frame, overlay, position)
        cv2.imshow("Overlayed Video", frame)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("/Users/pl1000790/Desktop/video.mp4", "/Users/pl1000790/Desktop/up-arrow.png")
