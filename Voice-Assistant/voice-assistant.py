import os
import webbrowser
import requests
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS
import pygame

# Load environment variables
load_dotenv()

# Initialize recognizer and pygame mixer globally
recognizer = sr.Recognizer()
pygame.mixer.init()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # put your key in .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenAI client once
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def main():
    print(">>> main() started")   # debug
    speak("Initializing Jarvis...")
    ...

# -------------------- SPEAK FUNCTIONS -------------------- #
def speak(text):
    print(f"[Jarvis says]: {text}")  # debug
    try:
        tts = gTTS(text)
        filename = "temp.mp3"
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
        os.remove(filename)
    except Exception as e:
        print(f"[Speak Error] {e}")
# -------------------- AI PROCESSOR -------------------- #
def ai_process(command: str) -> str:
    """Send user command to LLM and return response."""
    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528:free",
            messages=[
                {"role": "system", "content": "You are Jarvis. Keep responses short and clear."},
                {"role": "user", "content": command}
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {e}"

# -------------------- COMMAND HANDLER -------------------- #
def process_command(command: str):
    """Handle user commands with rules or AI fallback."""
    cmd = command.lower()

    if "open google" in cmd:
        webbrowser.open("https://google.com")
    elif "open facebook" in cmd:
        webbrowser.open("https://facebook.com")
    elif "open youtube" in cmd:
        webbrowser.open("https://youtube.com")
    elif "open linkedin" in cmd:
        webbrowser.open("https://linkedin.com")
    elif "news" in cmd:
        fetch_news()
    else:
        response = ai_process(command)
        speak(response)

def fetch_news():
    """Fetch latest headlines from NewsAPI."""
    if not NEWS_API_KEY:
        speak("No news API key configured.")
        return
    try:
        r = requests.get(
            f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWS_API_KEY}"
        )
        if r.status_code == 200:
            articles = r.json().get("articles", [])
            for article in articles[:5]:  # only first 5 to avoid endless talking
                speak(article.get("title", "No title"))
        else:
            speak("Failed to fetch news.")
    except Exception as e:
        speak(f"News error: {e}")

# -------------------- MAIN LOOP -------------------- #
def main():
    speak("Initializing Jarvis...")
    while True:
        try:
            with sr.Microphone() as source:
                print("Listening for wake word...")
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=2)

            try:
                wake_word = recognizer.recognize_google(audio).lower()
            except sr.UnknownValueError:
                # Nothing understandable said — just keep looping
                continue

            if wake_word == "jarvis":
                speak("Yes?")
                with sr.Microphone() as source:
                    print("Jarvis active, listening for command...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)

                try:
                    command = recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    speak("I didn't catch that.")
                    continue

                print(f"Command: {command}")
                process_command(command)

        except sr.WaitTimeoutError:
            # Just means silence — loop continues
            continue
        except Exception as e:
            print(f"[Loop Error] {e}")
            continue  # Don’t break the loop

if __name__ == "__main__":
    main()
