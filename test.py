import argparse
import os

from genre_utils import (
    compute_feature,
    convert_to_wav,
    ensure_dataset_ready,
    get_default_paths,
    load_label_map,
    predict_genre,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the music genre for a single audio file."
    )
    parser.add_argument("audio_path", help="Path to the audio file to classify.")
    parser.add_argument(
        "--dataset-path",
        help="Path to the serialized dataset file.",
    )
    parser.add_argument(
        "--label-map-path",
        help="Path to the label map pickle file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of neighbors to use for prediction.",
    )

    args = parser.parse_args()
    if not os.path.isfile(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    genres_root, default_dataset_path, default_label_map_path = get_default_paths()
    dataset = ensure_dataset_ready(
        genres_root=genres_root,
        dataset_path=args.dataset_path or default_dataset_path,
        label_map_path=args.label_map_path or default_label_map_path,
    )
    label_map = load_label_map(
        label_map_path=args.label_map_path or default_label_map_path,
        genres_root=genres_root,
    )

    wav_path, temp_wav = convert_to_wav(args.audio_path)
    try:
        feature = compute_feature(wav_path)
        predicted_label = predict_genre(dataset, feature, k=args.k)
        print(label_map.get(predicted_label, "Unknown"))
    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)


if __name__ == "__main__":
    main()
