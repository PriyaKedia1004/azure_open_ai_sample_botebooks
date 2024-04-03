using DotNetEnv;
using Microsoft.CognitiveServices.Speech;
using Microsoft.CognitiveServices.Speech.Audio;
using NAudio.Wave;
using System;
using System.IO;
using System.Threading.Tasks;

class Program
{
    static async Task Main(string[] args)
    {
        // Replace with your Azure Speech service subscription key and region
        // var speechKey = "63417bf8afd6407d97f129c2c8bd2906";
        // var speechRegion = "centralIndia";
        var speechKey = Environment.GetEnvironmentVariable("SUBSCRIPTION_KEY");
        var speechRegion = Environment.GetEnvironmentVariable("SPEECH_REGION");
        Console.WriteLine($"SpeechKey: {speechKey}");
        Console.WriteLine($"SpeechRegion: {speechRegion}");

        // Create a speech configuration
        var config = SpeechConfig.FromSubscription(speechKey, speechRegion);

        // URL of the audio stream
        // string url = "https://animalsvisionc5452967521.blob.core.windows.net/sample-audio-test/whatstheweatherlike.wav";
        string url = "https://animalsvisionc5452967521.blob.core.windows.net/sample-audio-test/Recording.wav";

        // Create a push stream
        var format = AudioStreamFormat.GetWaveFormatPCM(44100, 16, 1);
        var pushStream = AudioInputStream.CreatePushStream();

        // Create a speech recognizer using the push stream and the speech configuration
        using (var recognizer = new SpeechRecognizer(config, AudioConfig.FromStreamInput(pushStream)))
        {   
            // Subscribes to events.
            // recognizer.Recognizing += (s, e) =>
            // {
            //     Console.WriteLine($"RECOGNIZING: Text={e.Result.Text}");
            // };
            // Start recognizing and print the transcriptions to the console
            recognizer.Recognized += (s, e) =>
            {
                if (e.Result.Reason == ResultReason.RecognizedSpeech)
                {
                    Console.WriteLine($"Recognized: {e.Result.Text}");
                }
                else if (e.Result.Reason == ResultReason.NoMatch)
                {
                    Console.WriteLine($"No speech could be recognized.");
                }
            };

            // Start downloading the audio stream
            using (var httpClient = new HttpClient())
            using (var response = await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead))
            using (var stream = await response.Content.ReadAsStreamAsync())
            {

                // Start continuous recognition
                await recognizer.StartContinuousRecognitionAsync();

                var buffer = new byte[1024];
                int bytesRead;
                while ((bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length)) > 0)
                {
                    // Write the audio data to the push stream
                    pushStream.Write(buffer, bytesRead);
                
                }
            }

            // Wait for a key press
            Console.WriteLine("Press any key to stop transcription...");
            Console.ReadKey();

            // Stop continuous recognition
            await recognizer.StopContinuousRecognitionAsync();
        }
    }
}
