namespace ConsoleApp1;

/// <summary>
/// Interface for object detection providers that analyze images and return descriptions
/// </summary>
public interface IObjectDetectionProvider
{
    /// <summary>
    /// Analyzes an image and returns a natural language description
    /// </summary>
    /// <param name="imageBytes">The image data as a byte array (JPEG format)</param>
    /// <returns>A description of what's in the image</returns>
    Task<string> AnalyzeImageAsync(byte[] imageBytes);
}
