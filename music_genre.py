import argparse

from genre_utils import (
    build_dataset,
    ensure_dataset_ready,
    evaluate_dataset,
    get_default_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate the music genre dataset."
    )
    parser.add_argument(
        "--genres-root",
        type=str,
        help="Path to the genres dataset folder.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the serialized dataset file.",
    )
    parser.add_argument(
        "--label-map-path",
        type=str,
        help="Path to the label map pickle file.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the serialized dataset from the genres folder.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of neighbors used during evaluation.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.66,
        help="Fraction of examples used for training during evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/test splitting.",
    )

    args = parser.parse_args()
    genres_root, default_dataset_path, default_label_map_path = get_default_paths()
    dataset_path = args.dataset_path or default_dataset_path
    label_map_path = args.label_map_path or default_label_map_path
    genres_path = args.genres_root or genres_root

    if args.rebuild:
        print("Building dataset from genres folder...")
        build_dataset(
            genres_root=genres_path,
            dataset_path=dataset_path,
            label_map_path=label_map_path,
        )

    dataset = ensure_dataset_ready(
        genres_root=genres_path,
        dataset_path=dataset_path,
        label_map_path=label_map_path,
    )

    print(f"Dataset ready: {len(dataset)} examples")

    if dataset:
        accuracy = evaluate_dataset(
            dataset,
            k=args.k,
            train_fraction=args.train_fraction,
            seed=args.seed,
        )
        print(
            f"Evaluation: k={args.k}, train_fraction={args.train_fraction:.2f}, "
            f"accuracy={accuracy:.2%}"
        )


if __name__ == "__main__":
    main()
