import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import whisper

# Set paths
input_audio = "data/Pray.wav"  # Path to your full audio file
output_folder = "data/chunks"    # Folder to save the split audio files
metadata_file = "data/metadata.list"  # Metadata file to save the transcriptions

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Step 1: Split the audio into chunks of 5 seconds
def split_audio(input_audio, output_folder, chunk_length_ms=3000):
    """
    Splits an audio file into chunks of specified length (default is 5 seconds).
    """
    audio = AudioSegment.from_file(input_audio)
    chunks = make_chunks(audio, chunk_length_ms)

    print(f"Splitting audio into {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk_{i:03d}.wav"
        chunk_path = os.path.join(output_folder, chunk_name)
        chunk.export(chunk_path, format="wav")
        print(f"Saved: {chunk_path}")

    return len(chunks)

# Step 2: Transcribe audio chunks using Whisper
def transcribe_audio_chunks(output_folder, metadata_file, speaker_name="Kumuyi", language_code="en"):
    """
    Transcribes audio chunks and saves the metadata file.
    """
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("small")  # Use 'small', 'medium', or 'large' for more accuracy if needed

    print("Transcribing audio chunks...")
    with open(metadata_file, "w") as f:
        for audio_file in sorted(os.listdir(output_folder)):
            if audio_file.endswith(".wav"):
                file_path = os.path.join(output_folder, audio_file)
                print(f"Transcribing: {file_path}")
                result = model.transcribe(file_path)
                transcription = result["text"].strip()
                f.write(f"{file_path} |{speaker_name}|{language_code}|{transcription}\n")

    print(f"Transcription complete. Metadata saved to {metadata_file}")

# Step 3: Main function to orchestrate the workflow
def main():
    # Split the audio into chunks of 5 seconds each
    num_chunks = split_audio(input_audio, output_folder, chunk_length_ms=3000)
    print(f"Total chunks created: {num_chunks}")

    # Transcribe the chunks and generate the metadata file
    transcribe_audio_chunks(output_folder, metadata_file)

if __name__ == "__main__":
    main()
