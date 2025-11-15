namespace ConsoleApp1;

/// <summary>
/// Manages object detection requests and coordinates with the configured provider
/// </summary>
public class ObjectDetectionManager
{
    private readonly IObjectDetectionProvider _provider;
    private readonly SemaphoreSlim _semaphore;

    /// <summary>
    /// Creates a new ObjectDetectionManager with the specified provider
    /// </summary>
    /// <param name="provider">The detection provider to use</param>
    /// <param name="maxConcurrentRequests">Maximum number of concurrent detection requests</param>
    public ObjectDetectionManager(IObjectDetectionProvider provider, int maxConcurrentRequests = 1)
    {
        _provider = provider ?? throw new ArgumentNullException(nameof(provider));
        _semaphore = new SemaphoreSlim(maxConcurrentRequests, maxConcurrentRequests);
    }

    /// <summary>
    /// Detects objects in the provided image and returns a description
    /// </summary>
    /// <param name="imageBytes">The image data as a byte array</param>
    /// <returns>A natural language description of the detected objects</returns>
    public async Task<string> DetectObjectsAsync(byte[] imageBytes)
    {
        if (imageBytes == null || imageBytes.Length == 0)
        {
            throw new ArgumentException("Image bytes cannot be null or empty", nameof(imageBytes));
        }

        // Use semaphore to control concurrent requests and reduce latency spikes
        await _semaphore.WaitAsync();
        try
        {
            var startTime = DateTime.UtcNow;
            Console.WriteLine($"[{startTime:HH:mm:ss}] Starting image analysis...");

            var description = await _provider.AnalyzeImageAsync(imageBytes);

            var duration = (DateTime.UtcNow - startTime).TotalMilliseconds;
            Console.WriteLine($"[{DateTime.UtcNow:HH:mm:ss}] Analysis completed in {duration:F0}ms");

            return description;
        }
        finally
        {
            _semaphore.Release();
        }
    }
}
