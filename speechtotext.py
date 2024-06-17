import streamlit as st
from pydub import AudioSegment
from docx import Document
from io import BytesIO
from tempfile import NamedTemporaryFile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import subprocess
import uuid
from vosk import Model, KaldiRecognizer
import wave
import json
import matplotlib.pyplot as plt

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []
        self.recording = False

    def start_recording(self):
        self.audio_frames = []
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def recv_queued(self, frames):
        if self.recording:
            for frame in frames:
                self.audio_frames.append(frame.to_ndarray().flatten())

    def is_recording(self):
        return self.recording

def save_audio_to_wav(audio_frames, output_file_path):
    if audio_frames:
        audio_data = np.concatenate(audio_frames, axis=0).astype(np.int16)
        audio_segment = AudioSegment(
            data=audio_data.tobytes(),
            sample_width=2,  # Assuming 16-bit audio
            frame_rate=16000,  # 16 kHz
            channels=1  # Mono
        )
        audio_segment.export(output_file_path, format="wav")

def ffmpeg_convert_to_wav(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ar', '16000',
        '-ac', '1',
        '-f', 'wav',
        output_file
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as e:
        st.error("FFmpeg is not installed or not found in PATH. Please install FFmpeg and try again.")
        raise e

def transcribe_audio(wav_file_path, model_path):
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    with wave.open(wav_file_path, 'rb') as wf:
        recognizer.AcceptWaveform(wf.readframes(wf.getnframes()))
    result = recognizer.FinalResult()
    text = json.loads(result).get('text', '')
    return text

def save_to_word(transcript):
    doc = Document()
    doc.add_paragraph(transcript)
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

def plot_waveform(audio_frames):
    if audio_frames:
        audio_data = np.concatenate(audio_frames, axis=0).astype(np.int16)
        plt.figure(figsize=(10, 4))
        plt.plot(audio_data)
        plt.title('Audio Waveform')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        st.pyplot(plt)

st.title("Multilingual Speech-to-Text App")

# Initialize Audio Recorder
audio_recorder = AudioRecorder()

# Recorder Section
st.header("Record from Microphone")

if st.checkbox("Start Recording", key="recording_checkbox"):
    if not audio_recorder.is_recording():
        audio_recorder.start_recording()
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=1024,
        audio_processor_factory=lambda: audio_recorder,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    if webrtc_ctx.state.playing:
        st.write("WebRTC is playing.")
    else:
        st.write("WebRTC is not playing. Please check the connection.")
        st.write("WebRTC context state:", webrtc_ctx.state)

    plot_waveform(audio_recorder.audio_frames)

else:
    if audio_recorder.is_recording():
        audio_recorder.stop_recording()
        if audio_recorder.audio_frames:
            unique_filename = str(uuid.uuid4())
            temp_wav_file_path = f"/tmp/{unique_filename}.wav"
            save_audio_to_wav(audio_recorder.audio_frames, temp_wav_file_path)
            st.session_state['recorded_audio_path'] = temp_wav_file_path

if st.button("Stop Recording"):
    if audio_recorder.is_recording():
        audio_recorder.stop_recording()
        if audio_recorder.audio_frames:
            unique_filename = str(uuid.uuid4())
            temp_wav_file_path = f"/tmp/{unique_filename}.wav"
            save_audio_to_wav(audio_recorder.audio_frames, temp_wav_file_path)
            st.session_state['recorded_audio_path'] = temp_wav_file_path

# Language Selection
st.header("Select Language")
language = st.selectbox("Choose a language for transcription:", ["English", "Persian"])


# Set model path based on language selection
if language == "English":
    model_path = "/Users/mahdi/Downloads/vosk-model-en-us-0.22"  # Replace with your English model path
elif language == "Persian":
    model_path = "/Users/mahdi/Downloads/vosk-model-fa-0.5"  # Replace with your Persian model path

# Upload Audio File Section
st.header("Upload Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    try:
        with NamedTemporaryFile(delete=False) as temp_audio_file:
            temp_audio_file.write(uploaded_file.read())
            temp_audio_file_path = temp_audio_file.name

        unique_filename = str(uuid.uuid4())
        temp_wav_file_path = f"/tmp/{unique_filename}.wav"

        ffmpeg_convert_to_wav(temp_audio_file_path, temp_wav_file_path)

        transcript = transcribe_audio(temp_wav_file_path, model_path)
        st.write(transcript)

        if transcript:
            word_file = save_to_word(transcript)
            st.download_button(
                label="Download Word File",
                data=word_file,
                file_name=f"uploaded_transcript_{unique_filename}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
    except Exception as e:
        st.error(f"Error processing uploaded audio file: {e}")

# Transcribe Recorded Audio Section
st.header("Transcribe Recorded Audio")

if 'recorded_audio_path' in st.session_state:
    if st.button("Transcribe Recorded Audio"):
        try:
            transcript = transcribe_audio(st.session_state['recorded_audio_path'], model_path)
            st.write(transcript)

            if transcript:
                word_file = save_to_word(transcript)
                st.download_button(
                    label="Download Word File",
                    data=word_file,
                    file_name=f"recorded_transcript_{uuid.uuid4()}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        except Exception as e:
            st.error(f"Error during transcription: {e}")


 

