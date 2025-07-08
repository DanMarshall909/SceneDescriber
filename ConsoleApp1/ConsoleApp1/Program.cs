using Emgu.CV;
using Emgu.CV.Structure;
using System.Speech.Synthesis;
using System;
using System.Threading.Tasks;
using System.IO;

// Variables to track state
Mat? previousFrame = null;
DateTime lastUpdate = DateTime.MinValue;
const int UpdateInterval = 6000; // 6 seconds
const double ChangeThreshold = 0.1; // 10% change threshold

// Initialize OpenAI provider (can switch later)
IObjectDetectionProvider provider = new OpenAIObjectDetectionProvider("your-openai-api-key");
ObjectDetectionManager detectionManager = new ObjectDetectionManager(provider);

// Set up the camera capture
using VideoCapture capture = new VideoCapture(0);
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
    return (DateTime.Now - lastUpdate).TotalMilliseconds >= UpdateInterval;
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

    return changeRatio >= ChangeThreshold;
}
