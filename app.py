import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from PIL import Image

class DualDeformationTracker:
    def __init__(self, aruco_dict_type, marker_size_mm, use_cuda, detect_hammer=False):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.marker_size_mm = marker_size_mm
        self.initial_positions = {1: None}
        self.initial_marker_sizes_pixels = {1: None}
        if detect_hammer:
            self.initial_positions[0] = None
            self.initial_marker_sizes_pixels[0] = None
        self.use_cuda = use_cuda
        self.detect_hammer = detect_hammer
        if self.use_cuda:
            self.gpu_frame = cv2.cuda_GpuMat()

    def calculate_deformation(self, frame):
        if self.use_cuda:
            self.gpu_frame.upload(frame)
            gpu_gray = cv2.cuda.cvtColor(self.gpu_frame, cv2.COLOR_BGR2GRAY)
            gray = gpu_gray.download()
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = self.detector.detectMarkers(gray)

        deformations = {1: None}
        if self.detect_hammer:
            deformations[0] = None

        if ids is not None:
            for marker_id in deformations.keys():
                if marker_id in ids:
                    marker_index = np.where(ids == marker_id)[0][0]
                    marker_corners = corners[marker_index][0]
                    marker_center = np.mean(marker_corners, axis=0)
                    marker_size_pixels = np.linalg.norm(marker_corners[0] - marker_corners[1])

                    if self.initial_positions[marker_id] is None:
                        self.initial_positions[marker_id] = marker_center
                        self.initial_marker_sizes_pixels[marker_id] = marker_size_pixels
                        deformations[marker_id] = 0.0
                    else:
                        pixel_displacement = np.linalg.norm(marker_center - self.initial_positions[marker_id])
                        pixel_to_mm_ratio = self.marker_size_mm / self.initial_marker_sizes_pixels[marker_id]
                        deformations[marker_id] = pixel_displacement * pixel_to_mm_ratio

        return deformations

def replace_nan_with_previous(arr):
    """Replace NaN values with the previous non-NaN value."""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]

def process_video(video_path, aruco_dict_type, marker_size_mm, skip_frames, use_cuda, detect_hammer):
    if not os.path.exists(video_path):
        st.error(f"Error: Video file '{video_path}' not found.")
        return None, None, None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return None, None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    deformation_tracker = DualDeformationTracker(aruco_dict_type, marker_size_mm, use_cuda, detect_hammer)
    hammer_deformations = [] if detect_hammer else None
    pile_deformations = []
    frame_numbers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for frame_count in range(0, total_frames, skip_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
        
        deformations = deformation_tracker.calculate_deformation(frame)
        
        if detect_hammer:
            hammer_deformations.append(deformations[0] if deformations[0] is not None else np.nan)
        pile_deformations.append(deformations[1] if deformations[1] is not None else np.nan)
        frame_numbers.append(frame_count)
        
        progress_bar.progress((frame_count + 1) / total_frames)
        status_text.text(f"Processing frame {frame_count + 1}/{total_frames}")
    
    cap.release()
    status_text.text("Video processing complete!")
    
    # Replace NaN values with previous non-NaN values
    if detect_hammer:
        hammer_deformations = replace_nan_with_previous(np.array(hammer_deformations))
    pile_deformations = replace_nan_with_previous(np.array(pile_deformations))
    
    return np.array(frame_numbers), pile_deformations, hammer_deformations

def polynomial_func(x, *params):
    return np.polyval(params, x)

def fit_curve(x, y):
    degree = min(len(x) - 1, 3)  # Adjust degree based on number of points
    if degree < 1:
        return None
    try:
        popt, _ = curve_fit(lambda x, *params: polynomial_func(x, *params), x, y, p0=[0]*(degree+1))
        return popt
    except:
        return None

def find_hammer_peaks(frame_numbers, hammer_deformations, min_distance=50):
    """Find the true peaks of hammer movement by identifying maximum values in each cycle."""
    # Find all peaks as potential cycle starts
    all_peaks, _ = find_peaks(hammer_deformations, distance=min_distance)
    
    peak_frames = []
    peak_deformations = []
    
    for i in range(len(all_peaks) - 1):
        start = all_peaks[i]
        end = all_peaks[i+1]
        cycle_max_index = start + np.argmax(hammer_deformations[start:end])
        peak_frames.append(frame_numbers[cycle_max_index])
        peak_deformations.append(hammer_deformations[cycle_max_index])
    
    # Add the last peak
    last_peak_index = all_peaks[-1] + np.argmax(hammer_deformations[all_peaks[-1]:])
    peak_frames.append(frame_numbers[last_peak_index])
    peak_deformations.append(hammer_deformations[last_peak_index])
    
    return np.array(peak_frames), np.array(peak_deformations)

def plot_deformation(frame_numbers, pile_deformations, hammer_deformations=None):
    if hammer_deformations is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # Pile Movement Plot
    ax1.scatter(frame_numbers, pile_deformations, label='Pile Movement (ID 1)', alpha=0.5, color='blue')
    popt_pile = fit_curve(frame_numbers, pile_deformations)
    
    if popt_pile is not None:
        x_smooth = np.linspace(frame_numbers.min(), frame_numbers.max(), 500)
        y_smooth_pile = polynomial_func(x_smooth, *popt_pile)
        ax1.plot(x_smooth, y_smooth_pile, 'b-', label='Pile Movement Curve Fit')
    else:
        st.warning("Not enough data points to fit a curve for pile movement.")

    ax1.set_title('Pile Movement over Time')
    ax1.set_ylabel('Deformation (mm)')
    ax1.legend()
    ax1.grid(True)

    if hammer_deformations is not None:
        # Hammer Movement Plot
        ax2.scatter(frame_numbers, hammer_deformations, label='Hammer Movement (ID 0)', alpha=0.5, color='red')
        
        # Find true peaks for hammer movement
        peak_frames, peak_deformations = find_hammer_peaks(frame_numbers, hammer_deformations)
        
        # Plot only the peak points without any curve fitting
        ax2.scatter(peak_frames, peak_deformations, color='darkred', s=50, label='Peak Values')

        ax2.set_title('Hammer Movement over Time (Peak Values)')
        ax2.set_ylabel('Deformation (mm)')
        ax2.legend()
        ax2.grid(True)

    ax1.set_xlabel('Frame Number')
    plt.tight_layout()
    st.pyplot(fig)

def save_data(frame_numbers, pile_deformations, hammer_deformations=None, output_file="deformation_data.csv"):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if hammer_deformations is not None:
            writer.writerow(['Frame', 'Pile Deformation (mm)', 'Hammer Deformation (mm)'])
            for frame, pile, hammer in zip(frame_numbers, pile_deformations, hammer_deformations):
                writer.writerow([frame, pile, hammer])
        else:
            writer.writerow(['Frame', 'Pile Deformation (mm)'])
            for frame, pile in zip(frame_numbers, pile_deformations):
                writer.writerow([frame, pile])
    st.success(f"Data saved to {output_file}")

def add_logo(logo_path, size=(200, 150)):
    logo = Image.open(logo_path)
    logo = logo.resize(size)
    st.image(logo, use_column_width=False)

def main():
    st.title("Pile Driving Blow Count")
    add_logo("logoAI.png")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        video_path = "temp_video.mp4"
        aruco_dict_type = cv2.aruco.DICT_5X5_250
        marker_size_mm = st.number_input("Marker size (mm)", value=150, min_value=1)
        skip_frames = st.number_input("Skip frames", value=20, min_value=1, help="Process every nth frame")
        detect_hammer = st.checkbox("Detect Hammer (ID 0)", value=False)
        output_file = "deformation_data.csv"

        use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        st.write(f"CUDA is {'available' if use_cuda else 'not available'} for processing")

        if st.button("Process Video"):
            frame_numbers, pile_deformations, hammer_deformations = process_video(video_path, aruco_dict_type, marker_size_mm, skip_frames, use_cuda, detect_hammer)

            if frame_numbers is not None and pile_deformations is not None:
                plot_deformation(frame_numbers, pile_deformations, hammer_deformations)
                save_data(frame_numbers, pile_deformations, hammer_deformations, output_file)
                
                st.download_button(
                    label="Download CSV",
                    data=open(output_file, 'rb').read(),
                    file_name="deformation_data.csv",
                    mime="text/csv"
                )
            else:
                st.error("Failed to process video or no deformation data available.")
        
        os.remove("temp_video.mp4")

if __name__ == "__main__":
    main()