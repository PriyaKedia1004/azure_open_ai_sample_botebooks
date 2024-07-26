import time
from dotenv import load_dotenv
import os

import azure.cognitiveservices.speech as speechsdk
import wave
import asyncio
from pydub import AudioSegment
import threading
load_dotenv()
global transcribed_result
transcribed_result = []

def chunk_audio(file_path, chunk_length_ms):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Calculate the number of chunks
    num_chunks = len(audio) // chunk_length_ms

    # Create a directory to save the chunks
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"{base_name}_chunks"
    os.makedirs(output_dir, exist_ok=True)

    # Split the audio into chunks and save them
    for i in range(num_chunks):
        start_time = i * chunk_length_ms
        end_time = start_time + chunk_length_ms
        chunk = audio[start_time:end_time]
        chunk.export(os.path.join(output_dir, f"{base_name}_chunk_{i}.wav"), format="wav")

    print(f"Audio has been split into {num_chunks} chunks and saved in '{output_dir}' directory.")

# # Example usage
chunk_audio("gujrati_tts.wav", 5000)  # Chunk length is 10 seconds (10000 ms)

async def speech_recognize_continuous_from_file(file):
    """performs continuous speech recognition with input from an audio file"""
    # <SpeechContinuousRecognitionWithFile>
    global transcribed_result
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("SUBSCRIPTION_KEY"), region="centralIndia")
    audio_config = speechsdk.audio.AudioConfig(filename=file)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    # speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: transcribed_result.append(evt.result.text))
    # speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    # speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    # speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # Stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    # speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    start = time.time()
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.0005)

    speech_recognizer.stop_continuous_recognition()

    end = time.time()
    print("Time taken: ", end-start)

async def recognize_files_in_parallel(file_list):
    """Run speech recognition on multiple files in parallel"""
    tasks = []
    for i,file in enumerate(file_list):
        tasks.append(asyncio.create_task(speech_recognize_continuous_from_file(file)))
    await asyncio.gather(*tasks)

audio_files = [os.path.join("gujrati_tts_chunks/", file) for file in os.listdir("gujrati_tts_chunks")]

start_transcription_time = time.time()
asyncio.run(recognize_files_in_parallel(audio_files))
end_transcription_time = time.time()

print("Time taken: ", end_transcription_time - start_transcription_time)
