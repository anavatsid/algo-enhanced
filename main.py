import os
import argparse
from datetime import datetime
import time

import cv2
from order_utils.create_order import get_cfg
from order_utils.read_status import read_positions
from process import LSC_Detector
from rect_input import rectangle_select
from capture import ScreenCap
from trade import process_trade
from log_utils import export_log
import threading
import configparser


log_folder = "log"
os.makedirs(log_folder, exist_ok=True)
# from main import *
trade_config_dir = "order_utils/cfg"
os.makedirs(trade_config_dir, exist_ok=True)
global_trade_file = "order_utils/TRADE_FLAG.cfg"


def get_config_file_list():

    files = [file for file in os.listdir(trade_config_dir) if file.endswith(".cfg")]
    print(files)
    return files


def get_global_flag():
    if os.path.exists(global_trade_file):
        try:
            config = configparser.ConfigParser()
            config.read([global_trade_file])
            return config.getboolean("global", "global")
        except:
            return False
    else:
        print("Global Trade File not exists.")
        return False


def order_parser(data):
    cfg_file = os.path.join(trade_config_dir, data["ticker"])
    # target_position = 

    signal = data["signal"]
    if not os.path.exists(cfg_file):
        res_data = ["Config File not Exists."]
        print(res_data[0])
    else:
        original_config_dict = {}
        try:
            original_config_dict = get_cfg(cfg_file)
        except:
            res_data = ["Invalid Config File."]
            print(res_data[0])

        print("original_config_dict=", original_config_dict)

        if original_config_dict != {}:
            try:
                if signal == "LG":
                    action = "BUY"
                elif signal == "ST":
                    action = "SELL"
                else:
                    action = "CLOSE"
                # target_symbol = original_config_dict["contract"]["symbol"]
                # if len(current_positions) != 0:
                #     try:
                #         target_position = int(current_positions.Quantity[current_positions["Symbol"] == target_symbol].tolist()[0])
                #         # print(f"{target_pos=}")
                #     except:
                #         target_position = 0
                # else:
                #     target_position = 0
                cfg_dict = {
                    "contract": {},
                    "order": {
                        "action": action,
                        # "position": target_position                    
                    }
                }
                ticker_name = original_config_dict["contract"]["symbol"]
                log_file = ticker_name + "_" + datetime.now().strftime("%m_%d_%Y") + ".log"
                log_path = os.path.join(log_folder, log_file)
                # print("HI")
                
                if get_global_flag():
                    response_data = process_trade(cfg_dict, cfg_file, is_auto=True)
                    print("response_data[success]: ", response_data["success"])
                    if response_data["success"]:
                        tm_msg = response_data["description"]
                        slack_msg = response_data["slack_msg"]
                        export_log(tm_msg, log_path, slack_msg, True)
                        # self.response_data["status"] = "success"
                        cur_pos = response_data["current_pos"]
                        # res_data = ["{} Order Placed Successfully.".format(command), cur_pos]
                        res_data = [tm_msg]
                    else:
                        tm_msg = response_data["description"]
                        if response_data["slack_msg"] != "":
                            slack_msg = response_data["slack_msg"]
                        else:
                            slack_msg = tm_msg
                        export_log(tm_msg, log_path, slack_msg, is_notified=True)
                        # res_data = ["{} Order: {}".format(command, tm_msg.split("\t")[-1])]
                        res_data = [tm_msg]
                else:
                    print("The Current Auto Trade Status is Disabled.")

            except Exception as err:
                print("Failed to place order: {}".format(repr(err)))
                res_data = ["Failed to place order: {}".format(repr(err))]
    # print(res_data[0])


class OrderUtils():
    def __init__(self) -> None:
        self.order_buffer = []
        self.trader_thread = None
        self.is_start = False
        # self.position_parsed = False
        # self.positions = None
        # self.position_thread = None
        # self.is_getting_pos = True
        # self.position_loop()
    
    def position_loop(self):
        self.is_start =True
        self.position_thread = threading.Thread(target=self.update_positions, args=(2,))
        self.position_thread.setDaemon(True)
        if self.is_start:
            self.position_thread.start()

    def update_positions(self, id):
        while True:
            # if len(self.order_buffer) == 0:
            self.is_getting_pos = True
            try:
                self.positions = read_positions()
                print("Current Positions:\n", self.positions)
                self.position_parsed = True
                self.is_getting_pos = False
            except Exception as error:
                print(repr(error))
            finally:
                time.sleep(5)
            # else:
            #     time.sleep(5)

    def run_loop(self):
        self.is_start =True
        self.trader_thread = threading.Thread(target=self.read_order_request, args=(1,))
        self.trader_thread.setDaemon(True)
        # while True:
        #     if self.position_parsed:
        #         break
        #     else:
        #         time.sleep(1)
        if self.is_start:
            self.trader_thread.start()

    def read_order_request(self, id):
        while self.is_start:
            if len(self.order_buffer) != 0:
                # try:
                order_dict = self.order_buffer[0]
                order_parser(order_dict)
                del self.order_buffer[0]
                # except:
                #     time.sleep(1)
            else:
                time.sleep(1)

    def stop(self):
        self.is_start = False
        self.trader_thread.join()
        # self.position_thread.join()


def ls_detect(cap, trade_config_path, is_show, is_trade=False):
    # print("hello world!!!!")

    now = datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f")[:-3]
    msg = "{}\t{}\tScreen Capture Started...".format(now, trade_config_path)

    export_log(msg, trade_config_path, is_notified=True)
    
    ls_processor = LSC_Detector(weights="model/detects_final.weights", config="model/config")

    cur_signal = None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 15
        cap.set(cv2.CAP_PROP_FPS, fps)
    signal_num = 0
    frame_count = 0

    order_processor = OrderUtils()
    order_processor.run_loop()
    while cap.isOpened():
        
        frame_count += 1
        ret, frame = cap.read()
        if frame_count % fps != 0:
            continue
        if ret:
            response_data = ls_processor.infer(frame)
            if response_data["success"]:
                if cur_signal != response_data["signal"]:
                    if response_data["signal"] is None:
                        pass
                    else:
                        cur_signal = response_data["signal"]
                        if cur_signal == "LG":
                            full_signal = "LONG"
                            signal_num += 1
                        elif cur_signal == "ST":
                            full_signal = "SHORT"
                            signal_num += 1
                        else:
                            full_signal = "CLOSE"
                            signal_num += 1
                        msg = "{}\t{}".format(full_signal, signal_num)
                        now = datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f")[:-3]
                        tm_msg = '{}\t{}\t{}'.format(trade_config_path, now, msg)
                        export_log(tm_msg, trade_config_path, is_notified=True)
                        if is_trade:
                            order_data = {
                                "signal": cur_signal,
                                "ticker": trade_config_path
                            }
                            order_processor.order_buffer.append(order_data)

                if is_show:
                    cv2.imshow('frame', response_data["frame"])
            else:
                msg = "Failed to Process: {}".format(response_data['descript'])
                now = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
                tm_msg = '{}\t{}\t{}'.format(trade_config_path, now, msg)
                export_log(tm_msg, trade_config_path, is_notified=True)

                break
        else:
            msg = "Done process..."
            now = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
            tm_msg = '{}\t{}\t{}'.format(trade_config_path, now, msg)
            export_log(tm_msg, trade_config_path, is_notified=True)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    order_processor.stop()
    # Destroy all the windows
    cv2.destroyAllWindows()


def main(args):
    default_types = ["capture"]
    input_type = args.input_type
    # video_file = args.video
    # ticker_config_file = args.config
    is_show = args.show
    is_trade = args.trade

    assert input_type in default_types
    pre_defined_cfg_files = get_config_file_list()

    if pre_defined_cfg_files == []:
        print("Config file list is empty. Please check it.")
        return

    if input_type == "capture":
            # with open(ticker_config_file, "r") as t:
            #     pre_defined_tickers = [tick.strip() for tick in t.readlines() if tick.strip() != ""]
            # if len(pre_defined_tickers) == 0:
            #     print("Empty config file. Please confirm that.")
            #     return
        print("The existing ticker names are follow:")
        for i, tick in enumerate(pre_defined_cfg_files):
            print("\t{}:\t{}".format(i, tick))
        
        while True:
            ticker_idx = input("Please select ticker index as integer: ")
            try:
                ticker_idx = int(ticker_idx)
                if ticker_idx in range(len(pre_defined_cfg_files)):
                    ticker_config_file = pre_defined_cfg_files[ticker_idx]
                    # print()
                    break
                else:
                    print("Error: invalid ticker index. Please select again.")
            except ValueError:
                print("This is invalid index type. Input integer number.")
        if ticker_config_file is None:
            return

        print("Capture screen.. \nPlease confirm box boundaries and labels")

        time.sleep(1)
        coordinates = rectangle_select()
        if coordinates is None:
            return
        else:
            ticker_name = ticker_config_file.split(".")[0]
            ticker_info = "Ticker Name: {}\tTicker Coordinates: {}".format(ticker_name, coordinates)
            
            cap = ScreenCap(coordinates)
            # log_file = ticker_name + "_" + datetime.now().strftime("%m_%d_%Y") + ".log"

            # log_path = os.path.join(log_folder, log_file)
            # export_log(ticker_info, log_path)
            # trade_config_path = os.path.join(trade_config_dir, ticker_config_file)
            ls_detect(cap, ticker_config_file, is_show, is_trade)


if __name__ == "__main__":

    default="chart line light Right to Left.mov"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_type", default="capture", help="input type to be processed.")
    parser.add_argument("--show", action="store_true", help="Showing process and result frame.")
    parser.add_argument("--trade", action="store_true", help="Decide if auto trade is enable or not.")
    args = parser.parse_args()
    main(args)
