using System.Drawing;
using Microsoft.ML.Transforms.Image;

namespace FallingAI.Core
{
    /// <summary>
    /// ML Model input sample.
    /// </summary>
    public class ModelInput
    {
        public string Label;

        [ImageType(800, 600)] public Bitmap Image;

        public ModelInput(string label, Bitmap image)
        {
            Label = label;
            Image = image;
        }
    }
}