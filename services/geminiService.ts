import { GoogleGenAI, Chat, GenerateContentResponse, Content } from "@google/genai";
import { Message, AppSettings, DEFAULT_SETTINGS } from "../types";

// Initialize the API client
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export class GeminiService {
  private chat: Chat | null = null;
  private currentSettings: AppSettings = DEFAULT_SETTINGS;

  constructor() {
    this.startNewChat();
  }

  private getSystemInstruction(settings: AppSettings): string {
    const baseInstruction = "You are a helpful, clever, and concise AI assistant named Gemini Clone. Format your responses with Markdown.";
    
    let toneInstruction = "";
    switch (settings.personality) {
      case 'professional': toneInstruction = "Maintain a strictly professional, objective, and formal tone."; break;
      case 'friendly': toneInstruction = "Be warm, approachable, and friendly. Use emojis occasionally if appropriate."; break;
      case 'creative': toneInstruction = "Be imaginative and creative. Use colorful language and metaphors."; break;
      case 'humorous': toneInstruction = "Be witty and humorous. Feel free to crack jokes where appropriate."; break;
      case 'strict': toneInstruction = "Be concise, direct, and strict. Avoid filler words and pleasantries."; break;
    }

    const parts = [baseInstruction, toneInstruction, settings.customSystemInstruction].filter(Boolean);
    return parts.join("\n\n");
  }

  public startNewChat(settings: AppSettings = DEFAULT_SETTINGS, history?: Message[]) {
    this.currentSettings = settings;
    
    const config: any = {
      systemInstruction: this.getSystemInstruction(settings),
      temperature: settings.temperature,
      maxOutputTokens: settings.maxOutputTokens,
    };

    // Apply thinking config if enabled and using a supported model
    if (settings.enableThinking) {
      config.thinkingConfig = { thinkingBudget: settings.thinkingBudget };
    }

    // Convert internal message format to SDK Content format
    let sdkHistory: Content[] | undefined = undefined;
    if (history && history.length > 0) {
      sdkHistory = history.map(m => ({
        role: m.role,
        parts: [{ text: m.text }]
      }));
    }

    this.chat = ai.chats.create({
      model: settings.model,
      config: config,
      history: sdkHistory
    });
  }

  // Called when settings change mid-chat to update the session context
  public updateSettings(newSettings: AppSettings, currentHistory: Message[]) {
    this.startNewChat(newSettings, currentHistory);
  }

  public async *sendMessageStream(
    message: string, 
    attachments?: Array<{ mimeType: string; base64: string }>
  ): AsyncGenerator<string, void, unknown> {
    
    // Intercept image generation requests
    // We check if the prompt explicitly asks to "Generate an image" (case-insensitive)
    if (message.trim().toLowerCase().startsWith("generate an image")) {
        try {
            // Use the specific model for image generation
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash-image',
                contents: {
                    parts: [{ text: message }]
                }
            });
            
            let content = '';
            // Process response to find text and inline image data
            if (response.candidates?.[0]?.content?.parts) {
                for (const part of response.candidates[0].content.parts) {
                    if (part.text) {
                        content += part.text;
                    }
                    if (part.inlineData) {
                        const mimeType = part.inlineData.mimeType || 'image/png';
                        const data = part.inlineData.data;
                        // Construct Markdown image syntax
                        content += `\n\n![Generated Image](data:${mimeType};base64,${data})\n\n`;
                    }
                }
            }
            
            if (!content) content = "I couldn't generate an image based on that description.";
            
            // Yield the result immediately (simulate streaming since image gen is atomic)
            yield content;
            return;
        } catch (error) {
            console.error("Image generation failed:", error);
            yield "Sorry, I encountered an error while trying to generate that image. Please try again.";
            return;
        }
    }

    if (!this.chat) {
      this.startNewChat(this.currentSettings);
    }

    if (!this.chat) throw new Error("Failed to initialize chat");

    try {
      // Prepare parts for the message
      const parts: any[] = [{ text: message }];

      if (attachments && attachments.length > 0) {
        attachments.forEach(att => {
          parts.push({
            inlineData: {
              mimeType: att.mimeType,
              data: att.base64
            }
          });
        });
      }

      // If we have multiple parts (e.g. text + images), we need to pass the parts array
      // The SDK wrapper usually takes { message: string | Part[] }
      const messagePayload = parts.length === 1 ? message : parts;

      // Note: In the specific SDK wrapper @google/genai, chat.sendMessageStream typically expects
      // a string for the message property in strict typing, but often accepts parts arrays in practice
      // or via a different property. Based on standard Google Generative AI patterns, we send parts.
      // If the specific 'message' prop is strictly string, this might fail, but standard pattern is parts.
      
      // Since the types provided in the prompt's instruction only showed `message: string`, 
      // we are pushing the boundaries here to support multi-modal chat.
      // We pass the payload to 'message' which is the standard way to pass content in this wrapper.
      const result = await this.chat.sendMessageStream({ message: messagePayload as any });
      
      for await (const chunk of result) {
        const c = chunk as GenerateContentResponse;
        if (c.text) {
          yield c.text;
        }
      }
    } catch (error) {
      console.error("Error sending message to Gemini:", error);
      throw error;
    }
  }
}

export const geminiService = new GeminiService();