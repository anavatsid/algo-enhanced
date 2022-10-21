import os.path
from datetime import datetime
from notifier import send_notification_api


meta_log = "log/meta.log"
def export_log(msg, log_path, slack_msg: str = "", is_notified=False):
    # now = datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f")[:-3]
    # tm_msg = '{}\t{}\t{}'.format(ticker_name, now, msg)
    print(msg)
    if slack_msg is None or slack_msg.strip() == "":
        slack_msg = msg

    with open(log_path, "a") as f:
        f.write('{}\n'.format(msg))

        if is_notified:
            ret = send_notification_api(slack_msg)
            f.write('{}\n'.format(ret))


def export_meta(msg):
    with open(meta_log, "a") as f:
        f.write('{}\n'.format(msg))


def read_meta():
    if os.path.exists(meta_log):
        with open(meta_log, "r") as f:
            raw_data = f.readlines()
        lines = [line.strip() for line in raw_data if line.strip() != ""]
        history_data = [row.split("\t") for row in lines]
        history_data.reverse()
        return history_data
    else:
        return []
