from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import subprocess
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import uuid
import json
import threading
import time
from datetime import datetime
import zipfile
import tempfile
import shutil

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

# Flask configuration for file uploads
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Store processing status
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_info(video_path):
    """Get video duration and other metadata using FFprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format',
            '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                duration = float(stream.get('duration', 0))
                width = int(stream.get('width', 0))
                height = int(stream.get('height', 0))
                return duration, width, height
    except Exception as e:
        print(f"Error getting video info: {e}")
    return 0, 0, 0

def detect_audience_segments(video_path, job_id):
    """Detect segments that likely contain audience shots using computer vision"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        audience_segments = []
        current_segment_start = None
        
        # Load face cascade for people detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        frame_count = 0
        sample_rate = int(fps * 2)  # Sample every 2 seconds
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                # Update progress
                progress = (frame_count / total_frames) * 30  # This takes 30% of total progress
                processing_status[job_id]['progress'] = 10 + progress
                processing_status[job_id]['status'] = 'Analyzing audience segments...'
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # If many faces detected (>3), likely audience shot
                timestamp = frame_count / fps
                if len(faces) > 3:
                    if current_segment_start is None:
                        current_segment_start = timestamp
                else:
                    if current_segment_start is not None:
                        audience_segments.append((current_segment_start, timestamp))
                        current_segment_start = None
            
            frame_count += 1
        
        # Close the last segment if needed
        if current_segment_start is not None:
            audience_segments.append((current_segment_start, frame_count / fps))
        
        cap.release()
        return audience_segments
        
    except Exception as e:
        print(f"Error in audience detection: {e}")
        return []

def extract_clips(video_path, output_dir, clip_duration, overlap, audience_segments, settings, job_id):
    """Extract video clips avoiding audience segments"""
    try:
        duration, width, height = get_video_info(video_path)
        if duration == 0:
            return []
        
        clips = []
        current_time = 0
        clip_number = 1
        
        while current_time + clip_duration <= duration:
            clip_end = current_time + clip_duration
            
            # Check if this clip overlaps with audience segments
            is_audience_heavy = False
            for aud_start, aud_end in audience_segments:
                overlap_start = max(current_time, aud_start)
                overlap_end = min(clip_end, aud_end)
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > clip_duration * 0.3:  # More than 30% overlap
                        is_audience_heavy = True
                        break
            
            if not is_audience_heavy:
                # Update progress
                progress = 40 + (current_time / duration) * 40
                processing_status[job_id]['progress'] = progress
                processing_status[job_id]['status'] = f'Creating clip {clip_number}...'
                
                # Generate clip
                output_file = os.path.join(output_dir, f'clip_{clip_number:03d}.mp4')
                
                # Build FFmpeg command
                cmd = [
                    'ffmpeg', '-y', '-i', video_path,
                    '-ss', str(current_time),
                    '-t', str(clip_duration),
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-preset', 'medium'
                ]
                
                # Add quality settings
                if settings['quality'] == '720p':
                    cmd.extend(['-vf', 'scale=1280:720'])
                elif settings['quality'] == '1080p':
                    cmd.extend(['-vf', 'scale=1920:1080'])
                elif settings['quality'] == '4k':
                    cmd.extend(['-vf', 'scale=3840:2160'])
                
                # Add audio enhancements if enabled
                if settings.get('enhance_audio', False):
                    cmd.extend(['-af', 'highpass=f=200,lowpass=f=3000,dynaudnorm'])
                
                # Add silence removal if enabled
                if settings.get('remove_silence', False):
                    cmd.extend(['-af', 'silenceremove=1:0:-50dB'])
                
                cmd.append(output_file)
                
                # Execute FFmpeg command
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    clips.append({
                        'filename': f'clip_{clip_number:03d}.mp4',
                        'start_time': current_time,
                        'duration': clip_duration,
                        'path': output_file
                    })
                    clip_number += 1
                else:
                    print(f"FFmpeg error: {result.stderr}")
            
            # Move to next position
            current_time += clip_duration - overlap
        
        return clips
        
    except Exception as e:
        print(f"Error extracting clips: {e}")
        return []

def process_video_job(video_path, job_id, settings):
    """Background job to process video"""
    try:
        processing_status[job_id]['status'] = 'Starting analysis...'
        processing_status[job_id]['progress'] = 5
        
        # Create output directory for this job
        output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect audience segments
        processing_status[job_id]['status'] = 'Detecting audience segments...'
        processing_status[job_id]['progress'] = 10
        
        audience_segments = detect_audience_segments(video_path, job_id)
        
        # Extract clips
        processing_status[job_id]['status'] = 'Extracting clips...'
        processing_status[job_id]['progress'] = 40
        
        clips = extract_clips(
            video_path, 
            output_dir, 
            settings['clip_duration'], 
            settings['overlap'], 
            audience_segments, 
            settings, 
            job_id
        )
        
        # Finalize
        processing_status[job_id]['status'] = 'Complete'
        processing_status[job_id]['progress'] = 100
        processing_status[job_id]['clips'] = clips
        processing_status[job_id]['completed_at'] = datetime.now().isoformat()
        
    except Exception as e:
        processing_status[job_id]['status'] = 'Error'
        processing_status[job_id]['error'] = str(e)
        processing_status[job_id]['progress'] = 0

@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        # Check if the post request has the file part
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided. Please select a video file.'}), 400
        
        file = request.files['video']
        if file.filename == '' or file.filename is None:
            return jsonify({'error': 'No file selected. Please choose a video file.'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not supported. Please use: {", ".join(ALLOWED_EXTENSIONS).upper()}'}), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            return jsonify({'error': f'File too large ({size_mb:.1f} MB). Maximum size is 2GB.'}), 400
        
        if file_size == 0:
            return jsonify({'error': 'File appears to be empty. Please select a valid video file.'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(filepath)
        
        # Get processing settings from form
        settings = {
            'clip_duration': int(request.form.get('clip_duration', 30)),
            'overlap': int(request.form.get('overlap', 5)),
            'quality': request.form.get('quality', '1080p'),
            'min_speaker_time': int(request.form.get('min_speaker_time', 70)),
            'remove_silence': request.form.get('remove_silence') == 'true',
            'enhance_audio': request.form.get('enhance_audio') == 'true',
            'add_subtitles': request.form.get('add_subtitles') == 'true',
            'smart_cropping': request.form.get('smart_cropping') == 'true'
        }
        
        # Initialize processing status
        processing_status[job_id] = {
            'status': 'Initializing...',
            'progress': 0,
            'started_at': datetime.now().isoformat(),
            'filename': filename,
            'settings': settings
        }
        
        # Start background processing
        thread = threading.Thread(target=process_video_job, args=(filepath, job_id, settings))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'Processing started',
            'message': 'Video upload successful. Processing has begun.'
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 2GB.'}), 413

@app.route('/api/status/<job_id>')
def get_status(job_id):
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_status[job_id])

@app.route('/api/download/<job_id>')
def download_clips(job_id):
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    if processing_status[job_id]['status'] != 'Complete':
        return jsonify({'error': 'Processing not complete'}), 400
    
    try:
        # Create ZIP file with all clips
        zip_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_clips.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            output_dir = os.path.join(OUTPUT_FOLDER, job_id)
            for clip in processing_status[job_id]['clips']:
                clip_path = clip['path']
                if os.path.exists(clip_path):
                    zipf.write(clip_path, clip['filename'])
        
        return send_file(zip_path, as_attachment=True, download_name=f'ted_talk_clips_{job_id}.zip')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<job_id>/<filename>')
def download_single_clip(job_id, filename):
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    clip_path = os.path.join(OUTPUT_FOLDER, job_id, filename)
    if not os.path.exists(clip_path):
        return jsonify({'error': 'Clip not found'}), 404
    
    return send_file(clip_path, as_attachment=True)

@app.route('/api/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id):
    """Clean up job files and status"""
    try:
        # Remove from status
        if job_id in processing_status:
            del processing_status[job_id]
        
        # Remove uploaded file
        upload_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(job_id)]
        for f in upload_files:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        
        # Remove output directory
        output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        # Remove ZIP file
        zip_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_clips.zip')
        if os.path.exists(zip_path):
            os.remove(zip_path)
        
        return jsonify({'message': 'Cleanup completed'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/test')
def test_page():
    """Serve the upload test page"""
    return send_from_directory('.', 'test_upload.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_jobs': len(processing_status)
    })

if __name__ == '__main__':
    print("Starting TED Talk Clip Generator API...")
    print("Make sure you have FFmpeg installed and available in PATH")
    print("Installing required packages: pip install flask flask-cors opencv-python")
    app.run(debug=True, host='0.0.0.0', port=5000)