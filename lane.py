import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from collections import deque


# Global variables for temporal smoothing, vanishing point, and centerline history

def undistort_frame(frame):
   """
   Undistort the frame using dummy camera calibration parameters.
   Replace these parameters with your actual calibration data if available.
   """
   h, w = frame.shape[:2]
   camera_matrix = np.array([[w, 0, w / 2],
                             [0, w, h / 2],
                             [0, 0, 1]], dtype=np.float32)
   dist_coeffs = np.zeros((5, 1), dtype=np.float32)
   undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
   return undistorted


def region_of_interest(img, vertices):
   """
   Applies a mask to keep only the region of interest defined by 'vertices'.
   """
   mask = np.zeros_like(img)
   cv2.fillPoly(mask, vertices, 255)
   masked = cv2.bitwise_and(img, mask)
   return masked


def detect_lanes(frame):
   """
   Detect lanes using the Hough transform method.
   Returns:
     left_line, right_line, center_line, roi_vertices, and a flag indicating if valid lanes were detected.
   Each lane line is represented as [x1, y1, x2, y2] (with (x1, y1) at the bottom).
   """
   height, width = frame.shape[:2]
   bottom_offset = int(0.1 * height)


   # Define a taller trapezoidal ROI.
   roi_vertices = np.array([[
       (200, height - bottom_offset),                # Bottom left
       (int(0.4 * width), int(0.75 * height)),         # Top left (raised)
       (int(0.6 * width), int(0.75 * height)),         # Top right (raised)
       (width-250, height - bottom_offset)               # Bottom right
   ]], dtype=np.int32)


   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(gray, (5, 5), 0)
   edges = cv2.Canny(blur, 50, 150)


   cropped_edges = region_of_interest(edges, roi_vertices)
   lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180,
                           threshold=2, minLineLength=20, maxLineGap=150)


   left_lines = []
   right_lines = []
   if lines is not None:
       for line in lines:
           for x1, y1, x2, y2 in line:
               if x2 - x1 == 0:  # Avoid division by zero
                   continue
               slope = (y2 - y1) / (x2 - x1)
               if slope < -0.5:
                   left_lines.append([x1, y1, x2, y2])
               elif slope > 0.5:
                   right_lines.append([x1, y1, x2, y2])


   valid_detected = (lines is not None) and (len(left_lines) > 0 or len(right_lines) > 0)
   if not valid_detected:
       return None, None, None, roi_vertices, False


   def average_line(lines):
       if len(lines) == 0:
           return None
       x_coords, y_coords = [], []
       for line in lines:
           x_coords.extend([line[0], line[2]])
           y_coords.extend([line[1], line[3]])
       if len(x_coords) == 0:
           return None
       poly = np.polyfit(y_coords, x_coords, 1)
       slope = poly[0]
       intercept = poly[1]
       y_bottom = height - bottom_offset
       y_top = int(0.7 * height)
       x_bottom = int(slope * y_bottom + intercept)
       x_top = int(slope * y_top + intercept)
       return [x_bottom, y_bottom, x_top, y_top]


   left_avg = average_line(left_lines)
   right_avg = average_line(right_lines)


   default_left_line = [int(0.3 * width), height - bottom_offset, int(0.3 * width), int(0.5 * height)]
   default_right_line = [int(0.7 * width), height - bottom_offset, int(0.7 * width), int(0.5 * height)]
   if left_avg is None:
       left_avg = default_left_line
   if right_avg is None:
       right_avg = default_right_line

   return left_avg, right_avg, roi_vertices, True

def overlay_lanes(frame, left_line, right_line, roi_vertices):
   """
   Draws the ROI polygon, detected lane lines, centerline, and overlays a small
   perspective-transformed (bird's-eye) view in the top left corner.
   """
   overlay = frame.copy()
   cv2.polylines(overlay, [roi_vertices], isClosed=True, color=(0, 255, 255), thickness=3)
   if left_line is not None:
       cv2.line(overlay, (left_line[0], left_line[1]),
                (left_line[2], left_line[3]), (0, 255, 0), 5)
   if right_line is not None:
       cv2.line(overlay, (right_line[0], right_line[1]),
                (right_line[2], right_line[3]), (0, 255, 0), 5)
   return overlay

def main():
   global prev_left_line, prev_right_line
   video_path = "/Users/pl261721/Downloads/2679.mov"  # Replace with your video path
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
       print("Error: Could not open video.")
       return


   # Determine the video's FPS and calculate delay per frame
   fps = cap.get(cv2.CAP_PROP_FPS)
   wait_time = int(1000 / fps) if fps > 0 else 30

   while True:
       ret, frame = cap.read()
       if not ret:
           break


       # Step 1: Undistort the frame and convert to grayscale.
       frame = undistort_frame(frame)
       height, width = frame.shape[:2]
       bottom_offset = int(0.1 * height)
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


       # Step 2: Lane detection
       left_line, right_line, roi_vertices, valid = detect_lanes(frame)



       prev_left_line = left_line
       prev_right_line = right_line

       # Step 4: Perspective transform and final overlay.
       output_frame = overlay_lanes(frame, left_line, right_line, roi_vertices)


       cv2.imshow("Lane Detection Overlay", output_frame)
       prev_gray = gray.copy()  # Update the previous frame for any future optical flow if needed.
       if cv2.waitKey(wait_time) & 0xFF == ord('q'):
           break


   cap.release()
   cv2.destroyAllWindows()


if __name__ == "__main__":
   main()
