# Music Genre Classification

A simple music genre classifier built in Python.

The project extracts MFCC-based audio features from labeled song segments, stores them in a serialized dataset, and predicts genre labels using a nearest-neighbor classification strategy.

## Project Structure

- `app.py` - Streamlit app for uploading audio and predicting genre.
- `music_genre.py` - CLI utility to build the dataset and evaluate model accuracy.
- `test.py` - CLI script to classify a single audio file.
- `genre_utils.py` - Core feature extraction, dataset serialization, audio conversion, and nearest-neighbor prediction.
- `genres/genres` - Root dataset folder containing genre-specific subfolders and WAV samples.

## Features

- Builds a serialized dataset from local genre audio files.
- Computes MFCC features and covariance-based distance metrics.
- Uses k-nearest neighbors for genre prediction.
- Supports user audio uploads via Streamlit.
- Supports WAV, MP3, FLAC, OGG, M4A, AAC, and WMA input files.

## Requirements

- Python 3.8+ (or compatible)
- `numpy`
- `scipy`
- `streamlit`
- `python_speech_features`
- `imageio_ffmpeg`
- `ffmpeg` installed or available via `imageio_ffmpeg`

## Installation

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install numpy scipy streamlit python_speech_features imageio_ffmpeg
```

3. Ensure `ffmpeg` is installed and available on your system path, or that `imageio_ffmpeg` can locate its binary.

## Usage

### 1. Build the dataset

Before using the app, build the serialized dataset from the `genres/genres` folder:

```bash
python music_genre.py --rebuild
```

This creates `my.dat` and `label_map.pkl` in the repository root.

### 2. Run the Streamlit app

Launch the web interface:

```bash
streamlit run app.py
```

Upload an audio file and the app will return a predicted genre.

### 3. Predict using the CLI

Classify a single file from the command line:

```bash
python test.py path\to\audio.wav
```

### 4. Evaluate model accuracy

Run evaluation on the dataset using a train/test split:

```bash
python music_genre.py
```

Optional CLI arguments:

- `--k`: Number of neighbors (default: `5`)
- `--train-fraction`: Train split fraction (default: `0.66`)
- `--seed`: Random seed for splitting (default: `42`)

## Notes

- Audio files are converted to WAV format before feature extraction when needed.
- The prediction pipeline compares feature vectors and covariance matrices between the uploaded audio and dataset examples.
- The dataset loader automatically rebuilds the dataset if the serialized file is missing or invalid.

## License

This repository is provided as-is for educational and experimentation purposes.
