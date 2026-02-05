import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import yt_dlp
import os
from collections import defaultdict, Counter
import json
from datetime import datetime

class VideoEmotionPipeline:
    def __init__(self, model_path='../model.h5', cascade_path='../haarcascade_frontalface_default.xml'):
        self.face_classifier = cv2.CascadeClassifier(cascade_path)
        self.emotion_model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.frame_emotions = []
        self.face_count_per_frame = []
        
    def download_youtube_video(self, url, output_path='downloads'):
        """Download YouTube video"""
        os.makedirs(output_path, exist_ok=True)
        
        def progress_hook(d):
            if d['status'] == 'downloading':
                percent = d.get('_percent_str', 'N/A')
                speed = d.get('_speed_str', 'N/A')
                print(f"\rDownloading... {percent} at {speed}", end='', flush=True)
            elif d['status'] == 'finished':
                print(f"\nDownload completed: {d['filename']}")
        
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
            'progress_hooks': [progress_hook],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename, info['title']
    
    def detect_emotions_in_frame(self, frame):
        """Detect emotions in a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)
        
        frame_emotions = []
        
        for face_num, (x, y, w, h) in enumerate(faces, 1):
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
            
            # Draw face number
            cv2.rectangle(frame, (x, y), (x+80, y+30), (255, 0, 0), -1)
            cv2.putText(frame, f'Face {face_num}', (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Extract face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = self.emotion_model.predict(roi, verbose=0)[0]
                emotion_idx = prediction.argmax()
                emotion = self.emotion_labels[emotion_idx]
                confidence = prediction[emotion_idx]
                
                frame_emotions.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'bbox': (x, y, w, h),
                    'face_number': face_num
                })
                
                # Draw emotion label
                cv2.putText(frame, f'{emotion}: {confidence:.2f}', 
                           (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
        return frame, frame_emotions
    
    def process_video(self, video_path, output_path=None, progress_callback=None, realtime_callback=None):
        """Process entire video frame by frame"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.frame_emotions = []
        self.face_count_per_frame = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, emotions = self.detect_emotions_in_frame(frame.copy())
            
            # Store results
            self.frame_emotions.append(emotions)
            self.face_count_per_frame.append(len(emotions))
            
            # Write processed frame
            if output_path:
                out.write(processed_frame)
            
            # Show real-time processing (every 10th frame to avoid lag)
            if realtime_callback and frame_count % 10 == 0:
                realtime_callback(processed_frame)
            
            frame_count += 1
            if progress_callback:
                progress_callback(frame_count, total_frames)
        
        cap.release()
        if output_path:
            out.release()
            
        return self.generate_statistics()
    
    def generate_statistics(self):
        """Generate comprehensive statistics"""
        all_emotions = []
        emotion_timeline = []
        
        for frame_idx, frame_data in enumerate(self.frame_emotions):
            frame_emotion_counts = Counter()
            
            for face_data in frame_data:
                emotion = face_data['emotion']
                all_emotions.append(emotion)
                frame_emotion_counts[emotion] += 1
            
            emotion_timeline.append({
                'frame': frame_idx,
                'face_count': len(frame_data),
                **dict(frame_emotion_counts)
            })
        
        # Overall statistics
        emotion_counts = Counter(all_emotions)
        total_faces = len(all_emotions)
        
        stats = {
            'total_frames': len(self.frame_emotions),
            'total_faces_detected': total_faces,
            'avg_faces_per_frame': np.mean(self.face_count_per_frame) if self.face_count_per_frame else 0,
            'emotion_distribution': dict(emotion_counts),
            'emotion_percentages': {k: (v/total_faces)*100 for k, v in emotion_counts.items()} if total_faces > 0 else {},
            'timeline': emotion_timeline,
            'dominant_emotion': emotion_counts.most_common(1)[0][0] if emotion_counts else 'None'
        }
        
        return stats
    
    def create_visualizations(self, stats, save_path='plots'):
        """Create visualization plots"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Emotion Distribution Pie Chart
        if stats['emotion_distribution']:
            plt.figure(figsize=(10, 8))
            plt.pie(stats['emotion_distribution'].values(), 
                   labels=stats['emotion_distribution'].keys(), 
                   autopct='%1.1f%%', startangle=90)
            plt.title('Overall Emotion Distribution')
            plt.savefig(f'{save_path}/emotion_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Emotion Timeline
        if stats['timeline']:
            df = pd.DataFrame(stats['timeline']).fillna(0)
            emotion_cols = [col for col in df.columns if col not in ['frame', 'face_count']]
            
            plt.figure(figsize=(15, 8))
            for emotion in emotion_cols:
                plt.plot(df['frame'], df[emotion], label=emotion, marker='o', markersize=2)
            plt.xlabel('Frame Number')
            plt.ylabel('Number of Faces')
            plt.title('Emotion Timeline Throughout Video')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{save_path}/emotion_timeline.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Face Count per Frame
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(self.face_count_per_frame)), self.face_count_per_frame, 'b-', alpha=0.7)
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Faces Detected')
        plt.title('Face Detection Throughout Video')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_path}/face_count_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Emotion Heatmap
        if stats['timeline']:
            df = pd.DataFrame(stats['timeline']).fillna(0)
            emotion_cols = [col for col in df.columns if col not in ['frame', 'face_count']]
            
            if emotion_cols:
                # Sample frames for heatmap (every 30th frame)
                sample_df = df.iloc[::30][emotion_cols]
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(sample_df.T, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Face Count'})
                plt.xlabel('Frame Sample (every 30th frame)')
                plt.ylabel('Emotions')
                plt.title('Emotion Intensity Heatmap')
                plt.savefig(f'{save_path}/emotion_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def save_results(self, stats, output_file='video_emotion_results.json'):
        """Save results to JSON file"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean stats for JSON serialization
        clean_stats = json.loads(json.dumps(stats, default=convert_types))
        clean_stats['timestamp'] = datetime.now().isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(clean_stats, f, indent=2)
        
        return output_file