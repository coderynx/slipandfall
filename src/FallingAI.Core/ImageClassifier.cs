using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace FallingAI.Core
{
    /// <summary>
    ///     Image Classifier with Artificial Intelligence.
    /// </summary>
    public class ImageClassifier
    {
        /// <summary>
        ///     Custom model path.
        /// </summary>
        private readonly string _customModelPath;

        /// <summary>
        ///     Inception TensorFlow model path.
        /// </summary>
        private readonly string _inceptionTensorFlowModelPath;

        /// <summary>
        ///     Machine Learning Context used.
        /// </summary>
        private readonly MLContext _mlContext = new MLContext(1);

        /// <summary>
        ///     Prediction engine.
        /// </summary>
        private PredictionEngine<ModelInput, ModelOutput> _predictionEngine;

        /// <summary>
        ///     Image classifier constructor.
        /// </summary>
        /// <param name="modelsDirectory">Models directory</param>
        public ImageClassifier(string modelsDirectory)
        {
            _inceptionTensorFlowModelPath = Path.Combine(modelsDirectory, "tensorflow_inception_graph.pb");
            _customModelPath = Path.Combine(modelsDirectory, "model.zip");
        }

        /// <summary>
        ///     Load the model from the working directory.
        /// </summary>
        public void LoadModel()
        {
            // Load model and initialize the prediction engine.
            _predictionEngine =
                _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(
                    _mlContext.Model.Load(_customModelPath, out _));
        }

        /// <summary>
        ///     Test a sample.
        /// </summary>
        /// <param name="bitmap">Bitmap to test.</param>
        /// <returns>Result.</returns>
        public ModelOutput Test(Bitmap bitmap)
        {
            // Predict image.
            return _predictionEngine.Predict(new ModelInput("", bitmap));
        }

        /// <summary>
        ///     Execute training.
        /// </summary>
        /// <param name="datasetPath">Training dataset path.</param>
        public void Train(string datasetPath)
        {
            // Load training data.
            var trainingData = LoadDataset(datasetPath);

            // Generate the model
            var model = GenerateModel(trainingData);

            // Save the model.
            _mlContext.Model.Save(model, trainingData.Schema, @"D:\SlipAndFall\model.zip");
        }

        /// <summary>
        ///     Load images (jpg) dataset from file system.
        /// </summary>
        /// <param name="datasetPath">Dataset path.</param>
        /// <returns>DataView dataset.</returns>
        private IDataView LoadDataset(string datasetPath)
        {
            // Load images from file system.
            var trainingSamples = (from directory in Directory.GetDirectories(datasetPath)
                from file in new DirectoryInfo(directory).GetFiles("*.jpg")
                let directoryInfo = file.Directory
                where directoryInfo != null
                select new ModelInput(directoryInfo.Name, new Bitmap(file.FullName))).ToList();

            // Return dataset DataView.
            return _mlContext.Data.LoadFromEnumerable(trainingSamples);
        }

        /// <summary>
        ///     Method to train the model.
        /// </summary>
        /// <param name="trainingData">Data to use for training.</param>
        /// <returns>Generated model.</returns>
        private ITransformer GenerateModel(IDataView trainingData)
        {
            // Defining the pipeline.
            var pipeline = _mlContext.Transforms.ResizeImages("Image_Resized",
                    InceptionModelSettings.ImageWidth, InceptionModelSettings.ImageHeight, "Image")

                // Convert the image to GrayScale.
                .Append(_mlContext.Transforms.ConvertToGrayscale("Image_GrayScale", "Image_Resized"))

                // Extract pixels from the image.
                .Append(_mlContext.Transforms.ExtractPixels("input", "Image_GrayScale",
                    interleavePixelColors: InceptionModelSettings.ChannelsLast,
                    offsetImage: InceptionModelSettings.Mean))

                // Load the Inception TensorFlow Model.
                .Append(_mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModelPath)
                    .ScoreTensorFlowModel(new[] { "softmax2_pre_activation" }, new[] { "input" }, true))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("LabelKey", "Label"))
                .Append(_mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy("LabelKey",
                    "softmax2_pre_activation"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(_mlContext);

            // Train the model.
            return pipeline.Fit(trainingData);
        }

        /// <summary>
        ///     Image inception model settings.
        /// </summary>
        private struct InceptionModelSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const bool ChannelsLast = true;
        }
    }
}