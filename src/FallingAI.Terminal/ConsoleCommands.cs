using System;
using FallingAI.Core;
using Microsoft.ML;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace FallingAI.Terminal
{
    public class ConsoleCommands
    {
        private readonly ImageClassifier _imageClassifier = new ImageClassifier(@"D:\SlipAndFall");

        public void Train(string datasetPath)
        {
            Console.WriteLine("Starting custom model training...");

            // Load image dataset and train.
            _imageClassifier.Train(datasetPath);

            Console.WriteLine("Custom model training done.");
        }

        public void Test()
        {
            // Starting Failing Detector.
            Console.WriteLine("Starting Falling Detector...");

            Console.WriteLine("Loading custom model...");

            // Loading custom model.
            _imageClassifier.LoadModel();

            // Start new video capture device.
            var capture = new VideoCapture(0) {Fps = 5};

            // Create a window for displaying camera stream.
            using (var window = new Window("Camera"))
            {
                using var image = new Mat();
                while (true)
                {
                    
                    // Capture image from the stream.
                    capture.Read(image);
                    
                    // Exit from cycle if there is no frame captured.
                    if (image.Empty()) break;
                    
                    // Show video frame on window.
                    window.ShowImage(image);
                    Console.Write("[Falling Detection AI] Detected: ");
                    
                    // Detect video frame.
                    var detectionResult = _imageClassifier.Test(image.ToBitmap());

                    // Check result and print to console.
                    switch (detectionResult.PredictedLabelValue)
                    {
                        case "standing":
                            Console.ForegroundColor = ConsoleColor.Green;
                            Console.Write("Standing\n");
                            break;
                        case "falling":
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.Write("Falling\n");
                            break;
                        case "empty":
                            Console.ForegroundColor = ConsoleColor.Yellow;
                            Console.Write("Empty\n");
                            break;
                        default:
                            Console.Write("Undetected\n");
                            break;
                    }

                    Console.ForegroundColor = ConsoleColor.White;

                    // Camera capture delay.
                    Cv2.WaitKey(30);
                }
            }

            Console.ReadKey();
        }

    }
}