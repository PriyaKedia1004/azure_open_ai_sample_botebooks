import time
from dotenv import load_dotenv
import os

import azure.cognitiveservices.speech as speechsdk
import wave
import asyncio
from pydub import AudioSegment
import threading
load_dotenv()

def read_wave_header(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        framerate = audio_file.getframerate()
        bits_per_sample = audio_file.getsampwidth() * 8
        num_channels = audio_file.getnchannels()
        return framerate, bits_per_sample, num_channels

def push_stream_writer(stream):
    # The number of bytes to push per buffer
    n_bytes = 10000
    wav_fh = wave.open("gujrati_tts.wav", 'rb') ## change the file name to the file you want to transcribe
    # Start pushing data until all data has been read from the file
    try:
        while True:
            frames = wav_fh.readframes(n_bytes // 2)
            print('read {} bytes'.format(len(frames)))
            if not frames:
                break
            stream.write(frames)
            time.sleep(.0001)
    finally:
        wav_fh.close()
        stream.close()  # must be done to signal the end of stream

start = time.time()

def speech_recognition_with_push_stream():
    """gives an example how to use a push audio stream to recognize speech from a custom audio
    source"""
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("SUBSCRIPTION_KEY"), region="centralIndia")
    speech_config.request_word_level_timestamps()
    # Setup the audio stream
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # Instantiate the speech recognizer with push stream input
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    recognition_done = threading.Event()

    start = time.time()

    def recognizing_cb(evt):
        nonlocal start
        print(time.time() - start)
        start = time.time()

    # Connect callbacks to the events fired by the speech recognizer
    def session_stopped_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('SESSION STOPPED: {}'.format(evt))
        recognition_done.set()

    start = time.time()
    # speech_recognizer.recognizing.connect(recognizing_cb)
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt.result)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(session_stopped_cb)
    # speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    push_stream_writer_thread = threading.Thread(target=push_stream_writer, args=[stream])

    start_time = time.time()
    speech_recognizer.start_continuous_recognition_async().get()

    # Start push stream writer thread
    # push_stream_writer_thread = threading.Thread(target=push_stream_writer, args=[stream])
    push_stream_writer_thread.start()
    # Start continuous speech recognition

    # Wait until all input processed
    recognition_done.wait()

    # Stop recognition and clean up
    speech_recognizer.stop_continuous_recognition_async()
    push_stream_writer_thread.join()
    end_time = time.time()
    print("Time taken : ", end_time - start_time)

speech_recognition_with_push_stream()
