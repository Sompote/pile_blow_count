import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm
from scipy.optimize import curve_fit

class DeformationTracker:
    def __init__(self, aruco_dict_type, marker_size_mm):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.marker_size_mm = marker_size_mm
        self.initial_position = None
        self.initial_marker_size_pixels = None

    def calculate_deformation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is not None and 1 in ids:
            marker_index = np.where(ids == 1)[0][0]
            marker_corners = corners[marker_index][0]
            marker_center = np.mean(marker_corners, axis=0)
            marker_size_pixels = np.linalg.norm(marker_corners[0] - marker_corners[1])

            if self.initial_position is None:
                self.initial_position = marker_center
                self.initial_marker_size_pixels = marker_size_pixels
                return 0.0
            else:
                pixel_displacement = np.linalg.norm(marker_center - self.initial_position)
                pixel_to_mm_ratio = self.marker_size_mm / self.initial_marker_size_pixels
                deformation_mm = pixel_displacement * pixel_to_mm_ratio
                return deformation_mm
        else:
            return None

def process_video(video_path, aruco_dict_type, marker_size_mm):
    if not os.path.exists(video_path):
        st.error(f"Error: Video file '{video_path}' not found.")
        return None, None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    deformation_tracker = DeformationTracker(aruco_dict_type, marker_size_mm)
    deformations = []
    frame_numbers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        deformation = deformation_tracker.calculate_deformation(frame)
        
        if deformation is not None:
            deformations.append(deformation)
            frame_numbers.append(frame_count)
        
        progress_bar.progress((frame_count + 1) / total_frames)
        status_text.text(f"Processing frame {frame_count + 1}/{total_frames}")
    
    cap.release()
    status_text.text("Video processing complete!")
    
    return np.array(frame_numbers), np.array(deformations)

def polynomial_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def fit_curve(x, y):
    popt, _ = curve_fit(polynomial_func, x, y)
    return popt

def plot_deformation(frame_numbers, deformations):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(frame_numbers, deformations, label='Deformation Data', alpha=0.5)
    
    popt_deformation = fit_curve(frame_numbers, deformations)
    x_smooth = np.linspace(frame_numbers.min(), frame_numbers.max(), 500)
    y_smooth_deformation = polynomial_func(x_smooth, *popt_deformation)
    ax.plot(x_smooth, y_smooth_deformation, 'r-', label='Deformation Curve Fit')

    ax.set_title('Deformation over Time (Marker 1)')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Deformation (mm)')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

def save_data(frame_numbers, deformations, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Deformation (mm)'])
        for frame, deformation in zip(frame_numbers, deformations):
            writer.writerow([frame, deformation])
    st.success(f"Data saved to {output_file}")

def main():
    st.title("Pile Driving blow count")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        video_path = "temp_video.mp4"
        aruco_dict_type = cv2.aruco.DICT_5X5_250
        marker_size_mm = st.number_input("Marker size (mm)", value=150, min_value=1)
        output_file = "deformation_data_marker1.csv"

        if st.button("Process Video"):
            frame_numbers, deformations = process_video(video_path, aruco_dict_type, marker_size_mm)

            if frame_numbers is not None and deformations is not None:
                plot_deformation(frame_numbers, deformations)
                save_data(frame_numbers, deformations, output_file)
                
                st.download_button(
                    label="Download CSV",
                    data=open(output_file, 'rb').read(),
                    file_name="deformation_data.csv",
                    mime="text/csv"
                )
            else:
                st.error("Failed to process video or no deformation data available for marker 1.")
        
        os.remove("temp_video.mp4")

if __name__ == "__main__":
    main()