using LangChain.Providers;
using LangChain.Providers.OpenAI;
using LangChain.Providers.Anthropic;

namespace ConsoleApp1;

/// <summary>
/// LangChain-based object detection provider that supports multiple LLM providers
/// for improved flexibility and latency optimization
/// </summary>
public class LangChainObjectDetectionProvider : IObjectDetectionProvider
{
    private readonly IChatModel _chatModel;
    private readonly string _systemPrompt;

    /// <summary>
    /// Creates a new LangChain provider with OpenAI GPT-4 Vision
    /// </summary>
    /// <param name="apiKey">OpenAI API key</param>
    /// <param name="modelName">Model name (default: gpt-4-vision-preview)</param>
    public static LangChainObjectDetectionProvider CreateOpenAI(
        string apiKey,
        string modelName = "gpt-4-vision-preview")
    {
        var provider = new OpenAiProvider(apiKey);
        var model = new OpenAiChatModel(provider, modelName);
        return new LangChainObjectDetectionProvider(model);
    }

    /// <summary>
    /// Creates a new LangChain provider with Anthropic Claude
    /// </summary>
    /// <param name="apiKey">Anthropic API key</param>
    /// <param name="modelName">Model name (default: claude-3-sonnet-20240229)</param>
    public static LangChainObjectDetectionProvider CreateAnthropic(
        string apiKey,
        string modelName = "claude-3-sonnet-20240229")
    {
        var provider = new AnthropicProvider(apiKey);
        var model = new AnthropicChatModel(provider, modelName);
        return new LangChainObjectDetectionProvider(model);
    }

    /// <summary>
    /// Creates a provider with a custom chat model
    /// </summary>
    /// <param name="chatModel">Any LangChain-compatible chat model</param>
    public LangChainObjectDetectionProvider(IChatModel chatModel)
    {
        _chatModel = chatModel ?? throw new ArgumentNullException(nameof(chatModel));
        _systemPrompt = @"You are a helpful assistant that describes scenes in images.
Provide concise, natural descriptions focusing on:
- Main subjects and their actions
- Important objects and their relationships
- The overall scene context
Keep descriptions brief (1-2 sentences) and conversational.";
    }

    /// <summary>
    /// Analyzes an image and returns a natural language description
    /// </summary>
    /// <param name="imageBytes">The image data as a byte array (JPEG format)</param>
    /// <returns>A description of what's in the image</returns>
    public async Task<string> AnalyzeImageAsync(byte[] imageBytes)
    {
        try
        {
            // Convert image bytes to base64 for LangChain
            string base64Image = Convert.ToBase64String(imageBytes);
            string imageDataUri = $"data:image/jpeg;base64,{base64Image}";

            // Create the message with image content
            var messages = new[]
            {
                new Message
                {
                    Role = MessageRole.System,
                    Content = _systemPrompt
                },
                new Message
                {
                    Role = MessageRole.User,
                    Content = new[]
                    {
                        new MessageContent
                        {
                            Type = MessageContentType.Text,
                            Text = "Describe what you see in this image."
                        },
                        new MessageContent
                        {
                            Type = MessageContentType.ImageUrl,
                            ImageUrl = new ImageUrl
                            {
                                Url = imageDataUri
                            }
                        }
                    }
                }
            };

            // Get response from the model
            var response = await _chatModel.GenerateAsync(
                messages,
                new ChatSettings
                {
                    MaxTokens = 150,
                    Temperature = 0.7
                });

            return response.Messages.LastOrDefault()?.Content.Text
                   ?? "Unable to generate description";
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error analyzing image: {ex.Message}");
            return "Error processing image";
        }
    }
}
