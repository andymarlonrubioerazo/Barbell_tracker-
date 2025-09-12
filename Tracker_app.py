import streamlit as st
import pandas as pd
import os
import base64
from ultralytics import YOLO
import torch
from scipy.interpolate import CubicSpline
import cv2
import subprocess
import io
from pandas import DataFrame
import numpy as np
from typing import Dict, List, Optional


class VideoProcessor:
    """Class to handle video processing operations"""

    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def interpolation_points(t, x, new_t):
        """Interpolate points using cubic spline"""
        cs = CubicSpline(t, x, bc_type='natural')
        return cs(new_t).astype(int)
    
    def trajectory_interpolate_every_class(self, df: DataFrame, clase: str, n_max: int) -> DataFrame:
        """Interpolate trajectory for a specific class"""
        df_filter = df[df['class'] == clase].copy()
        df_filter = df_filter.loc[df_filter.groupby('frame')['conf'].idxmax()].reset_index(drop=True)
        
        df_filter['y'] = df_filter[['y1', 'y2']].sum(axis=1) // 2
        df_filter['y'] = df_filter['y'].astype(int)

        df_filter['x'] = df_filter[['x1', 'x2']].sum(axis=1) // 2
        df_filter['x'] = df_filter['x'].astype(int)
        
        df_frames = pd.DataFrame(list(range(n_max)), columns=['frame'])
        df_frames = df_frames.merge(df_filter, how='left', on='frame')
        
        df_frames['x'] = self.interpolation_points(df_filter.frame, df_filter.x, df_frames.frame)
        df_frames['y'] = self.interpolation_points(df_filter.frame, df_filter.y, df_frames.frame)
        df_frames['class'] = clase
        
        return df_frames
    
    def draw_points_lines_every_class(self, frame: np.ndarray, df_spline: DataFrame, 
                                    ni_frames: int = 135, count: int = 0) -> np.ndarray:
        """Draw points and lines for each class on the frame"""
        colores = {0: (0, 0, 255), 1: (255, 0, 0)}
        
        for i, clase in enumerate(df_spline['class'].unique()):
            dfi = df_spline.loc[(df_spline['frame'].isin(range(count - ni_frames, count + 1))) & 
                               (df_spline['class'] == clase)].copy()
                
            x_puntos = dfi.loc[:, 'x'].tolist()
            y_puntos = dfi.loc[:, 'y'].tolist()

            x0, y0 = x_puntos[0], y_puntos[0]
            for x1, y1 in zip(x_puntos[1:], y_puntos[1:]):
                cv2.line(frame, (x0, y0), (x1, y1), colores[i], 2)
                x0, y0 = x1, y1
        
        return frame
    
    def write_video_with_trajectory(self, path_video: str, df_spline: DataFrame, 
                                   out_filename: str = 'video_avi.avi', delay: float = 3.0) -> bool:
        """Create video with trajectory overlay"""
        ratio_reduce = 0.6  # Scale factor to reduce dimensions
        
        cap = cv2.VideoCapture(path_video)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * ratio_reduce)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * ratio_reduce)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ni_frames = int(fps * delay)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        try:
            output = cv2.VideoWriter(
                filename=out_filename,
                fourcc=fourcc,
                fps=fps,
                frameSize=(width, height)
            )
            
            count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = self.draw_points_lines_every_class(frame, df_spline, ni_frames=ni_frames, count=count)
                frame0 = cv2.resize(frame, (width, height))
                output.write(frame0)
                count += 1
                
            return True
            
        except Exception as e:
            st.error(f"Error writing video: {str(e)}")
            return False
            
        finally:
            cap.release()
            if 'output' in locals():
                output.release()
    
    @staticmethod
    def convert_avi_to_mp4(input_file: str, output_file: str) -> bool:
        """Convert AVI video to MP4 format"""
        try:
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite file
                '-i', input_file,
                '-vcodec', 'libx264',
                output_file
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            st.error(f"Error in conversion: {e.stderr}")
            return False
        except FileNotFoundError:
            st.error("Error: ffmpeg is not installed or path is incorrect")
            return False
    
    def yolo_detection_videos(self, model_selected: str, path_video: str, model_classes: Dict) -> DataFrame:
        """Perform YOLO object detection on video"""
        model = YOLO(model=model_selected, task="detect")
                # Verificar si todas las clases existen

        try:
            results = model(
                source=path_video,
                show=False,
                device=self.device,
                verbose=False,
                stream=False,
                iou=0.25,
                max_det=2,
                half=True,
                conf=0.4,
    ####onnxx
                nms=True,

            )
        except Exception as e:
            st.error(f"Error al ejecutar el modelo en yolo_detections_videos: {str(e)}")
            st.write(f"Tipo de error: {type(e).__name__}")

        
        df = pd.DataFrame([])
        
        for i, re in enumerate(results):
            r = re.boxes.data.cpu().numpy()
            
            dfi = pd.DataFrame(r)
            
            if not dfi.empty:
                dfi.columns = ['x1', 'y1', 'x2', 'y2', 'conf', 'class']
                dfi['frame'] = i
                df = pd.concat([df, dfi], ignore_index=True)
        
        df['class'] = df['class'].replace(model_classes)
        n_max = i
        
        df_spline = None
        for i, clase in enumerate(df['class'].unique()):
            dfi = self.trajectory_interpolate_every_class(df, clase=clase, n_max=n_max)
            
            if i == 0:
                df_spline = dfi.copy()
            else:
                df_spline = pd.concat([df_spline, dfi], ignore_index=True)
        st.write('final del analisis')
        return df_spline


class WeightliftingApp:
    """Main application class for weightlifting analysis"""
    
    def __init__(self):
        self.img_path = "Gemini_Generated_Image_o0i5a5o0i5a5o0i5.png"
        self.number_maximum_videos = 3
        self.video_processor = VideoProcessor()
        path_modelos = os.path.join('modelos')
        # Initialize session state
        if "df" not in st.session_state:
            st.session_state.df = None
        
        # Set page configuration
        st.set_page_config(
            layout="wide",
            initial_sidebar_state='expanded',
            page_title='IA in Weightlifting',
            page_icon=self.img_path,

        )
        
        # Load models
        self.modelos ={

            'barbell_extremity': {
                'path': os.path.join(path_modelos, 'n_extremity_better_color.onnx'),
                'classes': {0: 'barbell_extremity'}
                        },
            'barbell_disk': {
                'path': os.path.join(path_modelos, 'n_disk_better_color.onnx'),
                'classes': {0: 'barbell_disk'}
                            },
            'barbell_extremity_and_disk': {
                'path': os.path.join(path_modelos, 'n_disk_extremity_better_color.onnx'),
                'classes': {0: 'barbell_extremity', 1: 'barbell_disk'}
                    },
            
            }
        
        
    @staticmethod
    def img_to_base64(image_path: str) -> Optional[str]:
        """Convert image to base64."""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            st.error(f"Error converting image to base64: {str(e)}")
            return None
    
    def setup_sidebar(self):
        """Setup the application sidebar"""
        title_container=st.container(horizontal=True)

        img_base64 = self.img_to_base64(self.img_path)
        if img_base64:
           #st.sidebar.markdown(
            title_container.image(
                self.img_path, width=400
                #f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
                
                #unsafe_allow_html=True,
            )
        title_container.markdown("""

            📋 **:red[Important INSTRUCTIONS]**:
                                 
            📹 Video Limit:
            You can upload a maximum of 3 videos at once.

            ⏱️ Video Duration:
            Each video MUST NOT exceed 60 seconds (1 minute).
            Longer videos will be automatically rejected.
                    
            ⏳ Processing Time:
            Analysis may take up to 2 minutes per video.
            Please wait patiently during processing.
                    
            ⚠️ Error Prevention:
            DO NOT interact with the application during processing.
            Any interaction may cause analysis errors.
            Keep the browser tab active and visible.
                    
            🎯 Recommended Format:
            Accepted formats: MP4, AVI, MOV
            Ensure videos clearly show the barbell and discs
            """)
        
    def process_videos(self, uploaded_videos: List, modelo_selected: str, delay: float, model_classes: Dict):
        """Process uploaded videos with YOLO model"""
            

        for video in uploaded_videos:
            video_name = os.path.join(self.video_processor.output_dir, video.name)
            excel_video = video_name.replace('.mp4', '.xlsx')
            
            # Save uploaded video to disk
            g = io.BytesIO(video.read())
            with open(video_name, "wb") as out:
                out.write(g.read())
            
            
            cap=cv2.VideoCapture(video_name)
            fps=cap.get(cv2.CAP_PROP_FPS)
            frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            seconds=frames/fps
            
            if seconds>60. :
                st.warning(f'No process {video.name}. The video is too long.\n Upload a video shorter than 60 seconds.')
                os.remove(video_name)
                continue

            # Process video
            if os.path.exists(excel_video):
                df_spline = pd.read_excel(excel_video)

                clases_faltantes = [clase for clase in df_spline['class'].unique() 
                                if clase not in self.modelos[self.mi]['classes'].values()]
                
                if clases_faltantes:
                    df_spline = self.video_processor.yolo_detection_videos(
                        model_selected=modelo_selected,
                        path_video=video_name,
                        model_classes=model_classes
                    )

            else:

                df_spline = self.video_processor.yolo_detection_videos(
                    model_selected=modelo_selected,
                    path_video=video_name,
                    model_classes=model_classes
                )

            # Create video with trajectory
            video_name_avi = video_name.replace('.mp4', '.avi')
            success = self.video_processor.write_video_with_trajectory(
                path_video=video_name,
                df_spline=df_spline,
                out_filename=video_name_avi,
                delay=delay
            )

            if success:
                # Save results and convert video format
                df_spline.to_excel(excel_video, index=False)
                self.video_processor.convert_avi_to_mp4(video_name_avi, video_name)
                os.remove(video_name_avi)
                st.success(f'Analysis Sucessful {video.name}',icon="✅", width=400)
        
    
    def display_results(self):
        """Display processed videos and download buttons"""
        video_names = [os.path.join(self.video_processor.output_dir, video) 
                      for video in os.listdir(self.video_processor.output_dir) 
                      if video.endswith('.mp4') and video in self.uploaded_videos]
        
      
        
        excel_names = [os.path.join(self.video_processor.output_dir, excel) 
                      for excel in os.listdir(self.video_processor.output_dir) 
                      if excel.endswith('.xlsx') and 
                      os.path.join(self.video_processor.output_dir, excel).replace('.xlsx','.mp4')  in video_names ] 

        
        # Create containers for display
        video_windows = st.container(horizontal=True,key='video_windows')
        download_video_buttons = st.container(horizontal=True, key='download_video_buttons')
        download_excel_buttons = st.container(horizontal=True,key='download_excel_buttons')

        # Display videos
        
        with video_windows:
            for video in video_names:
                video_windows.video(video, autoplay=True, loop=True, muted=True, width=400)
               
        # Download buttons for videos
        with download_video_buttons:
            for video in video_names:
                with open(video, "rb") as file:
                    download_video_buttons.download_button(
                        label=f"Download Video {os.path.basename(video)}",
                        data=file,
                        file_name='tracking_' + os.path.basename(video),
                        mime="video/mp4",
                        width=400
                    )
        
        # Download buttons for Excel files
        with download_excel_buttons:
            for  excel in excel_names:
                with open(excel, "rb") as file:
                    download_excel_buttons.download_button(
                        label=f"Download Trajectory (Excel) {os.path.basename(excel)}",
                        data=file,
                        file_name='trajectory_' + os.path.basename(excel),
                        mime="application/vnd.ms-excel",
                        width=400
                    )
    
    def run(self):
        """Run the main application"""
        self.setup_sidebar()
        
        # Create UI columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col2:
            self.mi = st.selectbox(
                label='Select model',
                options=self.modelos.keys(),
                index=0
            )
            self.model_selected=self.modelos[self.mi]
        
        with col1:
            try:
                uploaded_videos = st.file_uploader(
                    label=f"Upload Video, Maximum {self.number_maximum_videos} videos",
                    accept_multiple_files=True,
                    type=['mp4', 'avi', 'mov'],
                    key="unique_video_uploader_key",
                )
                
                # Remove duplicates by name

                uploaded_videos = {k.name: k for k in uploaded_videos}
                self.uploaded_videos=list(uploaded_videos.keys())

                if uploaded_videos:
                    
                    delete_videos=[video 
                                   for video in os.listdir(self.video_processor.output_dir)  
                                   if not video in uploaded_videos.keys() and 
                                   video.endswith('.mp4') ]
                    delete_excel=[excel 
                                   for excel in os.listdir(self.video_processor.output_dir)  
                                   if not excel.replace('.xlsx','.mp4') in uploaded_videos.keys() and 
                                   excel.endswith('.xlsx') ]
                    
                    for video in delete_videos+delete_excel:
                        os.remove(os.path.join(self.video_processor.output_dir,video))
                
                uploaded_videos = list(uploaded_videos.values())
                

            except Exception as e:
                st.error(f"⚠️ Unexpected error: {str(e)}")
                return
        
        with col3:
            delay = st.select_slider(
                label='Choose delay in seconds',
                options=np.arange(0, 10.5, 0.5),
                value=3.0
            )
        

            
        # Validate number of videos
        if len(uploaded_videos) > self.number_maximum_videos :
            st.warning(f'!!! ONLY {self.number_maximum_videos} VIDEOS ALLOWED', icon="🚨")
            st.stop()
        # Display uploaded videos
        if uploaded_videos:
            container1 = st.container(horizontal=True)
            with container1:
                for video in uploaded_videos:
                    container1.video(video, width=400, autoplay=True, loop=True, muted=True)
        
       
        if st.button('Process video with YOLO'):
            if not uploaded_videos:
                st.warning("Please upload at least one video first")
                return
            try:
                with st.spinner("Wait for it. The analysis could take minutes.", show_time=True):
                    self.process_videos(uploaded_videos=uploaded_videos, 
                                        modelo_selected=self.model_selected['path'], 
                                        delay=delay, 
                                        model_classes=self.model_selected['classes'])
            except :
                st.warning('No change parameters please')        
        # Display results
        self.display_results()


# Run the application
if __name__ == "__main__":
    app = WeightliftingApp()
    app.run()

