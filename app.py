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
        if result.returncode != 0:
            print(f"FFprobe error: {result.stderr}")
            return 0, 0, 0
            
        data = json.loads(result.stdout)
        
        # Try to get duration from format first (more reliable)
        duration = 0
        if 'format' in data and 'duration' in data['format']:
            duration = float(data['format']['duration'])
        
        # Get video stream info
        width, height = 0, 0
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                # If duration not found in format, try stream
                if duration == 0 and 'duration' in stream:
                    duration = float(stream['duration'])
                width = int(stream.get('width', 0))
                height = int(stream.get('height', 0))
                break
                
        return duration, width, height
    except Exception as e:
        print(f"Error getting video info: {e}")
    return 0, 0, 0

def detect_audience_segments(video_path, job_id):
    """Detect segments that likely contain audience shots using improved computer vision"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or total_frames == 0:
            print("Warning: Could not get video properties for audience detection")
            return []
        
        audience_segments = []
        audience_detections = []  # Store detection history for smoothing
        
        # Load face cascade for people detection
        try:
            # Try modern OpenCV first
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except AttributeError:
            # Fallback for older OpenCV versions
            face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                print("Warning: Could not load face cascade classifier")
                return []
        
        frame_count = 0
        sample_rate = max(1, int(fps))  # Sample every second
        
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
                # Use different parameters for better detection
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                timestamp = frame_count / fps
                
                # More sophisticated audience detection
                is_audience_frame = False
                if len(faces) >= 4:  # Multiple faces indicate audience
                    # Check face size distribution
                    face_areas = [w * h for (x, y, w, h) in faces]
                    avg_face_area = sum(face_areas) / len(face_areas) if face_areas else 0
                    
                    # Small faces indicate distant audience shots
                    if avg_face_area < 5000:  # Typical speaker face is larger
                        is_audience_frame = True
                    # Many similar-sized faces also indicate audience
                    elif len(faces) >= 6:
                        is_audience_frame = True
                
                audience_detections.append((timestamp, is_audience_frame))
            
            frame_count += 1
        
        cap.release()
        
        # Apply temporal smoothing to reduce false positives
        if len(audience_detections) >= 3:
            smoothed_detections = []
            for i in range(len(audience_detections)):
                # Look at surrounding frames for consensus
                start_idx = max(0, i - 2)
                end_idx = min(len(audience_detections), i + 3)
                surrounding = audience_detections[start_idx:end_idx]
                
                # If majority of surrounding frames indicate audience, mark as audience
                audience_count = sum(1 for _, is_aud in surrounding if is_aud)
                is_audience = audience_count > len(surrounding) // 2
                smoothed_detections.append((audience_detections[i][0], is_audience))
            
            # Convert smoothed detections to segments
            current_segment_start = None
            for timestamp, is_audience in smoothed_detections:
                if is_audience:
                    if current_segment_start is None:
                        current_segment_start = timestamp
                else:
                    if current_segment_start is not None:
                        # Only add segments longer than 3 seconds
                        if timestamp - current_segment_start >= 3:
                            audience_segments.append((current_segment_start, timestamp))
                        current_segment_start = None
            
            # Close the last segment if needed
            if current_segment_start is not None:
                final_timestamp = audience_detections[-1][0]
                if final_timestamp - current_segment_start >= 3:
                    audience_segments.append((current_segment_start, final_timestamp))
        
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
        
        # Ensure we don't go beyond video duration
        while current_time + clip_duration <= duration:
            clip_end = current_time + clip_duration
            
            # Check if this clip overlaps with audience segments
            is_audience_heavy = False
            audience_overlap_duration = 0
            
            for aud_start, aud_end in audience_segments:
                overlap_start = max(current_time, aud_start)
                overlap_end = min(clip_end, aud_end)
                if overlap_end > overlap_start:
                    audience_overlap_duration += overlap_end - overlap_start
            
            # Calculate percentage of audience content
            audience_percentage = audience_overlap_duration / clip_duration
            
            # Skip clips with too much audience content
            min_speaker_percentage = settings.get('min_speaker_time', 70) / 100.0
            if audience_percentage > (1 - min_speaker_percentage):
                is_audience_heavy = True
            
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
                    '-preset', 'medium',
                    '-movflags', '+faststart'  # Optimize for web playback
                ]
                
                # Build video filters
                video_filters = []
                if settings['quality'] == '720p':
                    video_filters.append('scale=1280:720')
                elif settings['quality'] == '1080p':
                    video_filters.append('scale=1920:1080')
                elif settings['quality'] == '4k':
                    video_filters.append('scale=3840:2160')
                
                # Add smart cropping if enabled
                if settings.get('smart_cropping', False):
                    # Only crop if the video is wider than 16:9
                    video_filters.append('crop=min(iw\\,ih*16/9):ih:max(0\\,(iw-min(iw\\,ih*16/9))/2):0')
                
                if video_filters:
                    cmd.extend(['-vf', ','.join(video_filters)])
                
                # Build audio filters
                audio_filters = []
                if settings.get('enhance_audio', False):
                    audio_filters.extend(['highpass=f=200', 'lowpass=f=3000', 'dynaudnorm'])
                
                if settings.get('remove_silence', False):
                    audio_filters.append('silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse')
                
                if audio_filters:
                    cmd.extend(['-af', ','.join(audio_filters)])
                
                # Add quality settings
                cmd.extend(['-crf', '23', '-maxrate', '2M', '-bufsize', '4M'])
                
                cmd.append(output_file)
                
                # Execute FFmpeg command with timeout
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
                    
                    if result.returncode == 0 and os.path.exists(output_file):
                        # Verify the output file is valid
                        if os.path.getsize(output_file) > 1000:  # At least 1KB
                            clips.append({
                                'filename': f'clip_{clip_number:03d}.mp4',
                                'start_time': current_time,
                                'duration': clip_duration,
                                'path': output_file
                            })
                            clip_number += 1
                        else:
                            print(f"Warning: Generated clip is too small, skipping: {output_file}")
                            if os.path.exists(output_file):
                                os.remove(output_file)
                    else:
                        print(f"FFmpeg error (exit code {result.returncode}): {result.stderr}")
                        if os.path.exists(output_file):
                            os.remove(output_file)
                            
                except subprocess.TimeoutExpired:
                    print(f"FFmpeg timeout for clip at {current_time}s")
                    if os.path.exists(output_file):
                        os.remove(output_file)
                except Exception as e:
                    print(f"Error processing clip at {current_time}s: {e}")
                    if os.path.exists(output_file):
                        os.remove(output_file)
            
            # Move to next position with proper overlap handling
            if is_audience_heavy:
                # If this was audience-heavy, move forward by smaller increments to find good content
                current_time += 5  # Move in 5-second increments when skipping audience
            else:
                # Normal advancement with overlap
                step_size = clip_duration - overlap
                current_time += max(step_size, 1)  # Ensure we always move forward
        
        return clips
        
    except Exception as e:
        print(f"Error extracting clips: {e}")
        return []

def process_video_job(video_path, job_id, settings):
    """Background job to process video"""
    try:
        processing_status[job_id]['status'] = 'Starting analysis...'
        processing_status[job_id]['progress'] = 5
        
        # Validate video file exists and is readable
        if not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")
        
        if os.path.getsize(video_path) == 0:
            raise Exception("Video file is empty")
        
        # Create output directory for this job
        output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check video info first
        duration, width, height = get_video_info(video_path)
        if duration == 0:
            raise Exception("Could not determine video duration - file may be corrupted")
        
        if duration < settings['clip_duration']:
            raise Exception(f"Video is too short ({duration:.1f}s) for clip duration of {settings['clip_duration']}s")
        
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
        
        # Check if we got any clips
        if not clips:
            processing_status[job_id]['status'] = 'Warning'
            processing_status[job_id]['error'] = 'No suitable clips found. Try adjusting settings or use a different video.'
            processing_status[job_id]['progress'] = 100
            return
        
        # Finalize
        processing_status[job_id]['status'] = 'Complete'
        processing_status[job_id]['progress'] = 100
        processing_status[job_id]['clips'] = clips
        processing_status[job_id]['completed_at'] = datetime.now().isoformat()
        
        print(f"Successfully processed video: {len(clips)} clips generated")
        
    except Exception as e:
        print(f"Error processing video {job_id}: {str(e)}")
        processing_status[job_id]['status'] = 'Error'
        processing_status[job_id]['error'] = str(e)
        processing_status[job_id]['progress'] = 0
        
        # Clean up any partial files
        try:
            output_dir = os.path.join(OUTPUT_FOLDER, job_id)
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
        except:
            pass

@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
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
        return jsonify({'error': str(e)}), 500

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

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_jobs': len(processing_status)
    })

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        # Check FFmpeg
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("FFmpeg not found")
        
        # Check FFprobe
        result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("FFprobe not found")
            
        print("✓ FFmpeg and FFprobe are available")
        return True
    except FileNotFoundError:
        print("✗ FFmpeg/FFprobe not found in PATH")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"✗ Dependency check failed: {e}")
        return False

if __name__ == '__main__':
    print("Starting TED Talk Clip Generator API...")
    
    if not check_dependencies():
        print("Please install missing dependencies before running the application.")
        exit(1)
    
    print("✓ All dependencies available")
    print("Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)