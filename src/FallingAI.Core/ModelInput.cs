using System.Drawing;
using Microsoft.ML.Transforms.Image;

namespace FallingAI.Core
{
    /// <summary>
    ///     ML Model input sample.
    /// </summary>
    public class ModelInput
    {
        [ImageType(800, 600)] public Bitmap Image;
        public string Label;

        public ModelInput(string label, Bitmap image)
        {
            Label = label;
            Image = image;
        }
    }
}