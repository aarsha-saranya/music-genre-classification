import os
import tempfile
import streamlit as st

from genre_utils import (
    SUPPORTED_EXTENSIONS,
    compute_feature,
    convert_to_wav,
    ensure_dataset_ready,
    load_label_map,
    predict_genre,
)


def main() -> None:
    st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")
    st.title("Music Genre Classification")
    st.write(
        "Upload an audio file and the app will predict the music genre using a precomputed dataset. "
        "Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC, WMA."
    )

    uploaded_file = st.file_uploader("Choose an audio file", type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS])

    if uploaded_file is None:
        st.info("Upload a file to get started.")
        return

    dataset = None
    label_map = None
    try:
        dataset = ensure_dataset_ready()
        label_map = load_label_map()
    except Exception as exc:
        st.error(f"Failed to load model data: {exc}")
        return

    if uploaded_file is not None:
        with st.spinner("Processing audio..."):
            bytes_data = uploaded_file.read()
            original_name = uploaded_file.name or "upload"
            file_ext = os.path.splitext(original_name)[1].lower() or ".wav"
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            temp_input_path = temp_audio.name
            temp_audio.write(bytes_data)
            temp_audio.close()

            try:
                wav_path, tmp_wav = convert_to_wav(temp_input_path)
                feature = compute_feature(wav_path)
                predicted_label = predict_genre(dataset, feature, k=5)
                predicted_genre = label_map.get(predicted_label, "Unknown")

                st.success(f"Predicted genre: **{predicted_genre}**")
                st.write("The prediction is based on a nearest-neighbor comparison against the serialized dataset.")
            except Exception as exc:
                st.error(f"Audio processing failed: {exc}")
            finally:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if 'tmp_wav' in locals() and tmp_wav and os.path.exists(tmp_wav):
                    os.remove(tmp_wav)


if __name__ == "__main__":
    main()
