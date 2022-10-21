import os
import requests
from dotenv import load_dotenv

load_dotenv()
slack_webhook_url = os.environ.get('SLACK_NOTIFICATION_URL', '')
slack_webhook_channel = os.environ.get('SLACK_NITIFICATION_CHANNEL', '')


def send_notification_api(data):
    sms_data = {'text': data}
    try:
        r = requests.post(url=slack_webhook_url, json=sms_data)
        if r.status_code == 200:
            return "Slack notification successfully sent."
    except Exception as e:
        print(repr(e))
        pass
    return "Slack notification failed."
    # return r.text
    

if __name__ == '__main__':
    send_notification_api("Hello, notification testing.")
    