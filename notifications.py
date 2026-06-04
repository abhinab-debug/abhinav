import requests
import json
import time
import queue
import threading

class DiscordNotificationManager:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.payload_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        while True:
            payload = self.payload_queue.get()
            if payload is None:
                break

            self._send_to_discord(payload)
            self.payload_queue.task_done()
            time.sleep(0.5) # Proactive 500ms delay

    def _send_to_discord(self, payload):
        try:
            response = requests.post(self.webhook_url, json=payload)
            if response.status_code == 429:
                # Rate Limit Handler
                retry_after = response.json().get('retry_after', 1)
                time.sleep(retry_after)
                self._send_to_discord(payload) # Retry
            elif response.status_code != 204:
                print(f"Discord error: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Exception sending to Discord: {e}")

    def send_alert(self, title: str, description: str, color: int):
        """
        Emerald Green (#00FA9A) -> 64154
        Crimson Red (#FF0000) -> 16711680
        """
        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }]
        }
        self.payload_queue.put(payload)

    def stop(self):
        self.payload_queue.put(None)
        self.worker_thread.join()
