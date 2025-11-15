using Emgu.CV;
using Emgu.CV.Structure;
using System.Speech.Synthesis;
using System;
using System.Threading.Tasks;
using System.IO;
using Microsoft.Extensions.Configuration;
using ConsoleApp1;

// Load configuration
var configuration = new ConfigurationBuilder()
    .SetBasePath(Directory.GetCurrentDirectory())
    .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
    .AddEnvironmentVariables()
    .Build();

// Get configuration values
var providerType = configuration["SceneDescriber:Provider"] ?? "OpenAI";
var updateInterval = int.Parse(configuration["SceneDescriber:Detection:UpdateIntervalMs"] ?? "6000");
var changeThreshold = double.Parse(configuration["SceneDescriber:Detection:ChangeThreshold"] ?? "0.1");
var cameraIndex = int.Parse(configuration["SceneDescriber:Detection:CameraIndex"] ?? "0");

// Variables to track state
Mat? previousFrame = null;
DateTime lastUpdate = DateTime.MinValue;

// Initialize LangChain provider with flexibility for multiple providers
IObjectDetectionProvider provider = providerType.ToUpperInvariant() switch
{
    "ANTHROPIC" => LangChainObjectDetectionProvider.CreateAnthropic(
        configuration["SceneDescriber:Anthropic:ApiKey"] ?? throw new InvalidOperationException("Anthropic API key not configured"),
        configuration["SceneDescriber:Anthropic:Model"] ?? "claude-3-sonnet-20240229"
    ),
    "OPENAI" or _ => LangChainObjectDetectionProvider.CreateOpenAI(
        configuration["SceneDescriber:OpenAI:ApiKey"] ?? throw new InvalidOperationException("OpenAI API key not configured"),
        configuration["SceneDescriber:OpenAI:Model"] ?? "gpt-4-vision-preview"
    )
};

Console.WriteLine($"Using provider: {providerType}");
ObjectDetectionManager detectionManager = new ObjectDetectionManager(provider);

// Set up the camera capture
using VideoCapture capture = new VideoCapture(cameraIndex);
SpeechSynthesizer synth = new SpeechSynthesizer();
synth.SelectVoiceByHints(VoiceGender.Male, VoiceAge.Adult);

if (!capture.IsOpened)
{
    Console.WriteLine("Unable to access the camera");
    return;
}

Console.WriteLine("Press 'Esc' to stop the application...");

while (true)
{
    using Mat frame = new Mat();
    capture.Read(frame);

    if (!frame.IsEmpty)
    {
        // Only update if the scene has changed significantly and it's been more than 6 seconds
        if (SceneChangedSignificantly(frame) && EnoughTimePassed())
        {
            byte[] imageBytes = MatToBytes(frame);
            string description = await detectionManager.DetectObjectsAsync(imageBytes);
            synth.Speak(description);

            lastUpdate = DateTime.Now;
        }

        CvInvoke.Imshow("Camera", frame);

        if (CvInvoke.WaitKey(30) == 27) // Press 'Esc' to stop
            break;
    }
}

// Function to convert Mat to byte array for object detection
byte[] MatToBytes(Mat mat)
{
    using MemoryStream ms = new MemoryStream();
    mat.Bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Jpeg);
    return ms.ToArray();
}

// Check if enough time has passed for the next update
bool EnoughTimePassed()
{
    return (DateTime.Now - lastUpdate).TotalMilliseconds >= updateInterval;
}

// Check if the scene has changed significantly compared to the previous frame
bool SceneChangedSignificantly(Mat currentFrame)
{
    if (previousFrame == null)
    {
        previousFrame = currentFrame.Clone();
        return true; // First frame always triggers detection
    }

    // Calculate the difference between the current frame and the previous one
    Mat diff = new Mat();
    CvInvoke.AbsDiff(currentFrame, previousFrame, diff);
    previousFrame = currentFrame.Clone();

    // Calculate the percentage of changed pixels
    double nonZeroCount = CvInvoke.CountNonZero(diff);
    double totalPixels = diff.Rows * diff.Cols;
    double changeRatio = nonZeroCount / totalPixels;

    return changeRatio >= changeThreshold;
}
