import threading

class VoiceCoach:
    def __init__(self):
        self.enabled = False
        self.engine = None
        self.last_message = ""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            self.enabled = True
        except Exception:
            pass

    def speak(self, message, force=False):
        if not self.enabled or self.engine is None:
            return
        if message == self.last_message and not force:
            return
        self.last_message = message
        thread = threading.Thread(target=self._say, args=(message,))
        thread.daemon = True
        thread.start()

    def _say(self, message):
        try:
            self.engine.say(message)
            self.engine.runAndWait()
        except Exception:
            pass