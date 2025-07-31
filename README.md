# TED Talk Clip Generator

A modern web application that automatically processes TED talk videos into engaging 30-second clips while intelligently avoiding audience shots. Perfect for creating social media content from educational talks.

## Features

🎬 **Smart Video Clipping**: Automatically cuts TED talks into perfect 30-second clips
🚫 **Audience Detection**: AI-powered computer vision removes audience shots automatically  
🎨 **Quality Options**: Export in 720p, 1080p, or 4K resolution
🔊 **Audio Enhancement**: Optional audio processing and silence removal
📱 **Social Media Ready**: Smart cropping options for various platforms
⚡ **Fast Processing**: Efficient FFmpeg-based video processing
📦 **Batch Download**: Download all clips as a convenient ZIP file

## Prerequisites

Before running this application, ensure you have:

1. **Python 3.8+** installed
2. **FFmpeg** installed and available in your system PATH
3. **OpenCV** dependencies (usually installed automatically)

### Installing FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**macOS:**
```bash
# Using Homebrew
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

## Quick Start

1. **Clone or download** this repository
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server:**
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to `http://localhost:5000`

5. **Upload a TED talk video** and configure your processing settings

6. **Wait for processing** to complete and download your clips!

## How It Works

### Audience Detection Algorithm

The system uses OpenCV's Haar Cascade classifiers to detect faces in video frames:

- **Sampling**: Analyzes every 2 seconds of video
- **Face Detection**: Identifies multiple faces in each frame
- **Audience Classification**: Segments with >3 faces are marked as "audience shots"
- **Smart Filtering**: Clips with >30% audience content are skipped

### Video Processing Pipeline

1. **Upload & Analysis**: Video is uploaded and analyzed for duration and quality
2. **Audience Detection**: Computer vision identifies audience segments
3. **Intelligent Clipping**: Creates 30-second clips avoiding audience shots
4. **Enhancement**: Applies optional audio/video improvements
5. **Export**: Generates downloadable clips in your chosen format

## Configuration Options

### Basic Settings
- **Clip Duration**: 10-60 seconds (default: 30s)
- **Quality**: 720p, 1080p, or 4K
- **Overlap**: 0-15 seconds between clips

### Advanced Options
- ✅ **Remove Silence**: Automatically removes long pauses
- ✅ **Audio Enhancement**: Improves voice clarity and removes noise
- ⚪ **Subtitles**: Generate automatic captions (coming soon)
- ✅ **Smart Cropping**: Optimize aspect ratio for social media

### Speaker Focus
- **Minimum Speaker Time**: 50-100% (default: 70%)
- Ensures clips focus primarily on the speaker, not the audience

## API Endpoints

The backend provides a REST API for integration:

- `POST /api/upload` - Upload video and start processing
- `GET /api/status/<job_id>` - Check processing status
- `GET /api/download/<job_id>` - Download all clips as ZIP
- `GET /api/download/<job_id>/<filename>` - Download individual clip
- `DELETE /api/cleanup/<job_id>` - Clean up job files
- `GET /api/health` - Health check

## File Structure

```
ted-talk-clipper/
├── app.py                 # Flask backend API
├── index.html            # Frontend web interface
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── uploads/             # Temporary uploaded videos
└── outputs/             # Generated clips and ZIP files
```

## Troubleshooting

### Common Issues

**"FFmpeg not found"**
- Ensure FFmpeg is installed and in your system PATH
- Try running `ffmpeg -version` in terminal

**"OpenCV error"**
- Install OpenCV system dependencies:
  ```bash
  # Ubuntu/Debian
  sudo apt install python3-opencv libopencv-dev
  
  # macOS  
  brew install opencv
  ```

**Large file uploads failing**
- Check your available disk space
- Ensure video is under 2GB file size limit

**Processing takes too long**
- Try using lower quality settings (720p)
- Reduce clip duration or increase overlap
- Use a more powerful machine for 4K processing

### Performance Tips

- **For best performance**: Use SSD storage and at least 8GB RAM
- **Large videos**: Consider using 720p for faster processing
- **Server deployment**: Increase Flask timeout settings for production

## Browser Compatibility

- ✅ Chrome 80+
- ✅ Firefox 75+  
- ✅ Safari 13+
- ✅ Edge 80+

## Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

### Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Make your changes and test thoroughly
6. Submit a pull request

## License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with detailed information

---

**Happy clipping! 🎬✨**