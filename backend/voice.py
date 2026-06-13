import pyttsx3
import threading

class VoiceCoach:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        self.last_message = ""
    
    def speak(self, message, force=False):
        if message == self.last_message and not force:
            return
        self.last_message = message
        # Run in thread to avoid blocking
        thread = threading.Thread(target=self._say, args=(message,))
        thread.daemon = True
        thread.start()
    
    def _say(self, message):
        self.engine.say(message)
        self.engine.runAndWait()