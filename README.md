# Fire & Smoke Detection System Using Computer Vision

A Computer Vision-based system for detecting fire and smoke from images and videos.

The system uses image preprocessing, HSV color-space analysis, segmentation, morphological operations, contour-based region detection, feature extraction, rule-based classification, and temporal motion validation for video.

---

## Project Objective

The objective of this project is to develop an educational Computer Vision system that can analyze images and videos for visual patterns associated with fire and smoke.

The system:

- Detects fire-like regions.
- Detects smoke-like regions.
- Draws bounding boxes around detected regions.
- Classifies input as `FIRE`, `SMOKE`, `FIRE + SMOKE`, or `NORMAL`.
- Uses temporal confirmation for video detection.
- Uses motion validation to reduce false smoke detections in videos.
- Saves processed images and videos as output.

> **Note:** This project is developed for academic and educational purposes. It is not intended to replace certified fire detection or safety systems.

---

## Features

### Image Detection

The system can process individual images and detect:

- Fire
- Smoke
- Fire + Smoke
- Normal scenes

The processed image is saved with detection results and bounding boxes.

### Video Detection

The system processes videos frame by frame and provides:

- Fire region detection
- Smoke region detection
- Temporal confirmation
- Smoke motion validation
- Frame-level detection information
- Final video-level classification
- Processed output video

### Command-Line Interface

The complete system can be executed from the command line without requiring a graphical interface.

---

## Computer Vision Methodology

The project follows the following pipeline:

```text
Input Image / Video
        |
        v
Image Preprocessing
        |
        v
Gaussian Blur
        |
        v
HSV Color Conversion
        |
        +-------------------+
        |                   |
        v                   v
Fire Segmentation      Smoke Segmentation
        |                   |
        v                   v
Morphological         Morphological
Processing             Processing
        |                   |
        +---------+---------+
                  |
                  v
          Region Detection
                  |
                  v
          Feature Extraction
                  |
                  v
        Rule-Based Classification
                  |
                  v
      Video Temporal Validation
                  |
                  v
          Final Classification
                  |
                  v
        Output Image / Video
```

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Pytest

### Computer Vision Techniques

- Image resizing
- Gaussian blur
- HSV color-space conversion
- Color-based segmentation
- Binary masking
- Morphological opening
- Morphological closing
- Contour detection
- Bounding-box extraction
- Region feature extraction
- Rule-based classification
- Temporal validation
- Frame difference / motion analysis

---

## Project Structure

```text
Fire & Smoke Detection/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── morphology.py
│   ├── detection.py
│   ├── classification.py
│   └── video_detection.py
│
├── data/
│   ├── images/
│   └── videos/
│
├── results/
│   ├── images/
│   └── videos/
│
├── tests/
│   └── test_detection.py
│
├── main.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

The project requires:

- Python 3.10 or later
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Pytest

The exact Python packages and versions used in the project are listed in:

```text
requirements.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with the actual GitHub repository details.

---

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv venv
```

Activate the environment:

```powershell
venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, you can use:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

The project is executed through `main.py`.

The system supports both image and video input.

---

## Image Detection

### Fire Image

```bash
python main.py --input data/images/test_fire.jpeg --output results/images/fire_result.jpg
```

### Smoke Image

```bash
python main.py --input data/images/test_smoke.jpeg --output results/images/smoke_result.jpg
```

### Normal Image

```bash
python main.py --input data/images/normal.jpeg --output results/images/normal_result.jpg
```

The processed image is saved in the specified output location.

---

## Video Detection

### Fire Video

```bash
python main.py --input data/videos/test_fire.mp4 --type video --output results/videos/fire_result.mp4
```

### Smoke Video

```bash
python main.py --input data/videos/test_smoke.mp4 --type video --output results/videos/smoke_result.mp4
```

### Normal Video

```bash
python main.py --input data/videos/normal.mp4 --type video --output results/videos/normal_result.mp4
```

The processed video is saved in the specified output location.

---

## Optional Display Mode

The video can also be displayed while processing by using the `--show` option.

```bash
python main.py --input data/videos/test_fire.mp4 --type video --output results/videos/fire_result.mp4 --show
```

Press:

```text
Q
```

to stop video playback.

---

## Verbose Mode

Additional region-level information can be displayed during image processing using:

```bash
python main.py --input data/images/test_fire.jpeg --verbose
```

Verbose mode provides additional information about detected regions and extracted features.

---

## Detection Process

### 1. Image Preprocessing

The input image or video frame is resized to a standard width while maintaining its aspect ratio.

Gaussian blur is then applied to reduce image noise and small variations.

### 2. HSV Color Conversion

The preprocessed BGR image is converted to the HSV color space.

HSV separates:

- Hue
- Saturation
- Value

This makes it useful for identifying color characteristics associated with fire and smoke.

### 3. Fire Segmentation

A color-based HSV threshold is used to create a binary fire mask.

Pixels satisfying the defined fire-like HSV range are identified as candidate fire regions.

### 4. Smoke Segmentation

A separate HSV threshold identifies low-saturation regions with brightness characteristics associated with smoke.

These regions are treated as candidate smoke regions.

### 5. Morphological Processing

Morphological operations are applied to the binary masks.

The system uses:

- Morphological opening
- Morphological closing

These operations help remove small noise and close small gaps in detected regions.

### 6. Region Detection

Contours are extracted from the cleaned binary masks.

Regions below the minimum area threshold are discarded.

Bounding boxes and region properties are then calculated.

### 7. Feature Extraction

For detected fire regions, the system extracts features including:

- Region area
- Width
- Height
- Aspect ratio
- Mean hue
- Mean saturation
- Mean value
- Maximum value
- Strong fire-color fraction
- Bright-pixel fraction

For smoke regions, characteristics including:

- Mean saturation
- Mean brightness
- Brightness variation
- Region area
- Region extent

are analyzed.

### 8. Classification

The extracted features are evaluated using rule-based classification.

Possible classifications include:

```text
FIRE
NON-FIRE
SMOKE
NON-SMOKE
```

### 9. Temporal Validation for Video

Video detection uses consecutive-frame confirmation.

A detection must persist for at least five consecutive frames before it is counted as confirmed.

This helps reduce short-lived detections caused by individual frames.

### 10. Smoke Motion Validation

Smoke candidates are additionally checked for temporal changes between consecutive frames.

Frame-difference analysis is used to identify movement or visual change within the candidate region.

This helps reduce false smoke detections caused by static low-saturation background regions.

---

## Output Classes

The system produces one of four final statuses:

```text
FIRE
SMOKE
FIRE + SMOKE
NORMAL
```

For images, detected regions are highlighted with bounding boxes.

For videos, the detection status is displayed on the processed frames and the processed video is saved to the specified output path.

---

## Testing

The project includes automated tests using Pytest.

Run the complete test suite with:

```bash
pytest -v
```

### Current Test Result

```text
8 passed
```

The automated tests cover:

- Region detection
- Fire classification
- Non-fire classification
- Smoke classification
- FIRE status
- SMOKE status
- FIRE + SMOKE status
- NORMAL status

---

## Experimental Validation

The system was tested using separate fire, smoke, and normal image and video samples.

### Image Tests

| Input | Fire Regions | Smoke Regions | Final Result |
|---|---:|---:|---|
| Fire image | 2 | 0 | FIRE |
| Smoke image | 0 | 8 | SMOKE |
| Normal image | 0 | 0 | NORMAL |

### Video Tests

| Input | Fire Frames | Smoke Frames | Final Result |
|---|---:|---:|---|
| Fire video | 669 | 0 | FIRE |
| Smoke video | 0 | 10 | SMOKE |
| Normal video | 0 | 0 | NORMAL |

> These results are specific to the test samples used during development and should not be interpreted as a general accuracy benchmark.

---

## Example Output

For a fire image, the terminal reports information similar to:

```text
------------------------------------------------------------
                    FINAL RESULT
------------------------------------------------------------
Fire regions detected  : 2
Smoke regions detected : 0
Detection status       : FIRE
------------------------------------------------------------
Output saved           : results/images/fire_result.jpg
------------------------------------------------------------
```

For a smoke image:

```text
------------------------------------------------------------
                    FINAL RESULT
------------------------------------------------------------
Fire regions detected  : 0
Smoke regions detected : 8
Detection status       : SMOKE
------------------------------------------------------------
Output saved           : results/images/smoke_result.jpg
------------------------------------------------------------
```

For a normal image:

```text
------------------------------------------------------------
                    FINAL RESULT
------------------------------------------------------------
Fire regions detected  : 0
Smoke regions detected : 0
Detection status       : NORMAL
------------------------------------------------------------
Output saved           : results/images/normal_result.jpg
------------------------------------------------------------
```

---

## Limitations

The system uses traditional Computer Vision techniques and rule-based classification. Therefore, performance can be affected by:

- Lighting conditions
- Fire-like colors in objects
- Gray or low-saturation backgrounds
- Different smoke appearances
- Image and video quality
- Camera movement
- Occlusion
- Environmental conditions
- Unseen scenes and objects

The system may produce false positives or false negatives on data that differs from the development test samples.

The project is intended for academic and educational demonstration and is not a certified real-world fire safety system.

---

## Future Scope

Possible improvements include:

- Training a machine learning classifier using a larger labeled dataset.
- Using deep learning models for fire and smoke detection.
- Integrating object detection architectures such as YOLO.
- Improving smoke segmentation using motion and texture features.
- Adding confidence scores.
- Using larger and more diverse datasets.
- Evaluating precision, recall, F1-score, and confusion matrices.
- Adding real-time camera support.
- Adding email or SMS alert functionality.
- Deploying the system on edge devices such as Raspberry Pi.
- Developing a web-based monitoring dashboard.

---

## Academic Relevance

This project demonstrates several important Computer Vision concepts:

- Image preprocessing
- Image resizing
- Gaussian filtering
- Color-space transformation
- Image segmentation
- Binary masking
- Morphological image processing
- Contour detection
- Region analysis
- Feature extraction
- Rule-based classification
- Temporal validation
- Motion detection

These techniques are integrated into an end-to-end Computer Vision application for fire and smoke analysis.

---

## Conclusion

The Fire & Smoke Detection System demonstrates how traditional Computer Vision techniques can be combined to analyze images and videos for fire- and smoke-like visual patterns.

The system supports both image and video input, performs preprocessing and segmentation, detects relevant regions, extracts visual features, applies rule-based classification, and uses temporal and motion-based validation for video analysis.

The project provides a practical implementation of Computer Vision concepts while also demonstrating the limitations of rule-based detection systems.

---

## Author

**Tuba Gulfen Qureshi**

Computer Science / Artificial Intelligence & Machine Learning Student

---

## License

This project is developed for educational and academic purposes.