# Weightlifting Trajectory Analysis App

A Streamlit-based web application for analyzing weightlifting movements by tracking and visualizing barbell trajectories using YOLO object detection models.

 <!-- Replace with your actual demo video -->
 [![Tracker App]()](https://weightlifting-trajectory-analysis-app.streamlit.app/)
 
[![Wathch Video](https://github.com/andymarlonrubioerazo/Barbell_tracker-/blob/main/tracking_image.png)](https://github.com/andymarlonrubioerazo/Barbell_tracker-/blob/main/video_final.mp4)





## Features

- **Multiple Model Support**: Choose from three different YOLO models:
  - Barbell extremity tracking
  - Barbell disk tracking
  - Combined extremity and disk tracking
  
- **Customizable Tracking Window**: Adjust the trajectory display duration from 0 to 10 seconds

- **Video Processing**: Upload and process up to 3 videos simultaneously

- **Visualization**: Overlay smooth trajectory lines on your weightlifting videos

- **Data Export**: Download both the processed videos with trajectories and Excel files containing coordinate data

## How It Works

1. **Upload Videos**: Select up to 3 weightlifting videos (MP4, AVI, or MOV format)
2. **Choose Model**: Select the appropriate tracking model for your analysis
3. **Set Duration**: Adjust how many seconds of trajectory to display (default: 3 seconds)
4. **Process**: The app will analyze the video using YOLO object detection
5. **Download**: Get your processed video with trajectory overlay and coordinate data

## Technical Details

- Built with Python using Streamlit for the web interface
- Utilizes Ultralytics YOLO for object detection
- Implements cubic spline interpolation for smooth trajectory visualization
- Automatically converts output to MP4 format for compatibility
- Provides coordinate data in Excel format for further analysis


## Usage Notes

- Videos longer than 60 seconds will not be processed
- GPU acceleration is recommended for faster processing
- The app supports tracking of both barbell extremities and disks
- Trajectory data includes frame-by-frame coordinates of detected objects


## Requirements

- Python 3.12
- Streamlit
- PyTorch
- OpenCV
- Pandas
- NumPy
- Ultralytics YOLO
- FFmpeg (for video conversion)

For detailed package versions, see the requirements.txt file.
