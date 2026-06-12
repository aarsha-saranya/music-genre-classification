import os
import pickle
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wav
from imageio_ffmpeg import get_ffmpeg_exe
from python_speech_features import mfcc

LABEL_MAP_FILENAME = "label_map.pkl"
DATASET_FILENAME = "my.dat"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
FeatureTuple = Tuple[np.ndarray, np.ndarray, int]


def get_workspace_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_default_paths() -> Tuple[str, str, str]:
    workspace_root = get_workspace_root()
    genres_root = os.path.join(workspace_root, "genres", "genres")
    dataset_path = os.path.join(workspace_root, DATASET_FILENAME)
    label_map_path = os.path.join(workspace_root, LABEL_MAP_FILENAME)
    return genres_root, dataset_path, label_map_path


def build_dataset(
    genres_root: Optional[str] = None,
    dataset_path: Optional[str] = None,
    label_map_path: Optional[str] = None,
) -> Dict[int, str]:
    if genres_root is None or dataset_path is None or label_map_path is None:
        genres_root, dataset_path, label_map_path = get_default_paths()

    if not os.path.isdir(genres_root):
        raise FileNotFoundError(f"Genres folder not found: {genres_root}")

    label_map: Dict[str, int] = {}
    current_label = 0

    with open(dataset_path, "wb") as f:
        for root, dirs, files in os.walk(genres_root):
            rel = os.path.relpath(root, genres_root)
            if rel == ".":
                continue

            genre = rel.split(os.sep)[0]
            if genre not in label_map:
                current_label += 1
                label_map[genre] = current_label
            label = label_map[genre]

            for file in sorted(files):
                if not file.lower().endswith(".wav"):
                    continue
                file_path = os.path.join(root, file)
                try:
                    rate, sig = wav.read(file_path)
                except ValueError as e:
                    print(f"Skipping invalid WAV file: {file_path} ({e})")
                    continue
                mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                covariance = np.cov(np.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(axis=0)
                pickle.dump((mean_matrix, covariance, label), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(label_map_path, "wb") as map_file:
        pickle.dump(label_map, map_file, protocol=pickle.HIGHEST_PROTOCOL)

    return {value: key for key, value in label_map.items()}


def load_dataset(dat_path: Optional[str] = None) -> List[FeatureTuple]:
    if dat_path is None:
        _, dat_path, _ = get_default_paths()

    if not os.path.isfile(dat_path):
        raise FileNotFoundError(f"Dataset file not found: {dat_path}")

    items: List[FeatureTuple] = []
    with open(dat_path, "rb") as f:
        while True:
            try:
                items.append(pickle.load(f))
            except EOFError:
                break
    return items


def validate_dataset(dataset: List[FeatureTuple]) -> bool:
    return len(dataset) > 0 and len({item[2] for item in dataset}) > 1


def ensure_dataset_ready(
    genres_root: Optional[str] = None,
    dataset_path: Optional[str] = None,
    label_map_path: Optional[str] = None,
) -> List[FeatureTuple]:
    if genres_root is None or dataset_path is None or label_map_path is None:
        genres_root, dataset_path, label_map_path = get_default_paths()

    try:
        dataset = load_dataset(dataset_path)
    except (FileNotFoundError, pickle.UnpicklingError):
        dataset = []

    if not validate_dataset(dataset):
        build_dataset(genres_root, dataset_path, label_map_path)
        dataset = load_dataset(dataset_path)
    return dataset


def load_label_map(
    label_map_path: Optional[str] = None,
    genres_root: Optional[str] = None,
) -> Dict[int, str]:
    workspace_root = get_workspace_root()
    if label_map_path is None:
        label_map_path = os.path.join(workspace_root, LABEL_MAP_FILENAME)
    if genres_root is None:
        genres_root = os.path.join(workspace_root, "genres", "genres")

    if os.path.isfile(label_map_path):
        with open(label_map_path, "rb") as f:
            label_map = pickle.load(f)
        if isinstance(label_map, dict):
            if all(isinstance(k, str) and isinstance(v, int) for k, v in label_map.items()):
                return {value: key for key, value in label_map.items()}
            if all(isinstance(k, int) and isinstance(v, str) for k, v in label_map.items()):
                return label_map

    if not os.path.isdir(genres_root):
        raise FileNotFoundError(f"Genres folder not found: {genres_root}")

    genre_names = [name for name in sorted(os.listdir(genres_root)) if os.path.isdir(os.path.join(genres_root, name))]
    if not genre_names:
        raise ValueError(f"No genre folders found in {genres_root}")

    return {idx + 1: genre for idx, genre in enumerate(genre_names)}


def convert_to_wav(input_path: str) -> Tuple[str, Optional[str]]:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path, None

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported audio extension. Please upload WAV, MP3, FLAC, OGG, M4A, AAC, or WMA files."
        )

    ffmpeg_exe = get_ffmpeg_exe()
    if not ffmpeg_exe or not os.path.isfile(ffmpeg_exe):
        raise RuntimeError("FFmpeg binary not found. Install ffmpeg or add it to PATH.")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_file.close()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
        "-ar",
        "22050",
        "-ac",
        "1",
        tmp_file.name,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.strip()}")

    return tmp_file.name, tmp_file.name


def compute_feature(audio_path: str) -> FeatureTuple:
    rate, sig = wav.read(audio_path)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(axis=0)
    return mean_matrix, covariance, 0


def _regularize_covariance(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return matrix + np.eye(matrix.shape[0], dtype=matrix.dtype) * eps


def distance(instance1: FeatureTuple, instance2: FeatureTuple, k: int = 1) -> float:
    mm1, cm1, _ = instance1
    mm2, cm2, _ = instance2
    cm1 = _regularize_covariance(cm1)
    cm2 = _regularize_covariance(cm2)
    inv_cm2 = np.linalg.inv(cm2)

    score = np.trace(np.dot(inv_cm2, cm1))
    delta = mm2 - mm1
    score += np.dot(np.dot(delta.transpose(), inv_cm2), delta)
    score += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    score -= k
    return float(score)


def get_neighbors(training_set: List[FeatureTuple], instance: FeatureTuple, k: int = 5) -> List[int]:
    distances = []
    for item in training_set:
        dist = distance(item, instance, k) + distance(instance, item, k)
        distances.append((item[2], dist))
    distances.sort(key=lambda pair: pair[1])
    return [label for label, _ in distances[:k]]


def nearest_class(neighbors: List[int]) -> int:
    votes: Dict[int, int] = {}
    for label in neighbors:
        votes[label] = votes.get(label, 0) + 1
    return max(votes.items(), key=lambda item: item[1])[0]


def predict_genre(dataset: List[FeatureTuple], feature: FeatureTuple, k: int = 5) -> int:
    neighbors = get_neighbors(dataset, feature, k)
    return nearest_class(neighbors)


def split_dataset(
    dataset: List[FeatureTuple],
    train_fraction: float = 0.66,
    seed: int = 42,
) -> Tuple[List[FeatureTuple], List[FeatureTuple]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    split_index = int(len(indices) * train_fraction)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return [dataset[i] for i in train_indices], [dataset[i] for i in test_indices]


def evaluate_dataset(
    dataset: List[FeatureTuple],
    k: int = 5,
    train_fraction: float = 0.66,
    seed: int = 42,
) -> float:
    train_set, test_set = split_dataset(dataset, train_fraction, seed)
    if not test_set:
        return 0.0
    predictions = [predict_genre(train_set, item, k) for item in test_set]
    correct = sum(1 for item, prediction in zip(test_set, predictions) if item[2] == prediction)
    return float(correct / len(test_set))
