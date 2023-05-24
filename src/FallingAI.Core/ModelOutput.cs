using System.Drawing;
using Microsoft.ML.Transforms.Image;

namespace FallingAI.Core
{
    /// <summary>
    /// Model output.
    /// </summary>
    public class ModelOutput
    {
        /// <summary>
        /// Prediction score.
        /// </summary>
        public float[] Score;

        /// <summary>
        /// Prediction result.
        /// </summary>
        public string PredictedLabelValue { get; set; }
    }
}