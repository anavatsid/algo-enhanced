# CREDITS https://tradersacademy.online/trading-topics/trader-workstation-tws/placing-orders
# WORKING CODE VV APR 2022
import os.path
import sys
import time
from datetime import datetime

import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
from threading import Timer, Thread
import configparser
from dotenv import load_dotenv
load_dotenv()
ibgw_ip_address = os.environ.get('IB_GATEWAY_IP', '127.0.0.1')
ibgw_port_num = int(os.environ.get('IB_GATEWAY_PORT', '4002'))
ibgw_client_num = int(os.environ.get('IB_GATEWAY_CLIENTS', '8'))

print(ibgw_ip_address, ibgw_port_num, ibgw_client_num)

def get_cfg(config_path):
    config = configparser.ConfigParser()
    config.read([config_path])
    ticker_name = os.path.basename(config_path)[:-4]
    _dict_ = {
        "contract": {
            **dict(config.items("contract")),
            'ticker_name': dict(config.items("contract"))['symbol'] # ticker_name
        },
        "order": dict(config.items("order"))
    }
    return _dict_


class OrderApp(EWrapper, EClient):
    def __init__(self, contract_config: dict = None, order_config: dict = None):
        EClient.__init__(self, self)
        self.is_working = False
        self.contract_config = contract_config
        self.order_config = order_config
        self.error_msg = [] #  pd.DataFrame([], columns=["ReqId", "ErrorCode", "ErrorMsg"])
        self.orderStatus_msg = []  # pd.DataFrame([], columns=["Id", "Status", "Filled", "Remaining", "LastFillPrice"])
        self.openOrder_msg = []  # pd.DataFrame([], columns=["Id", "Symbol", "Sec Type", "Exchange", "Action", "Order Type",
                                                       # "Quantity", "Status"])
        self.execDetails_msg = []  # pd.DataFrame([], columns=["ReqId", "Symbol", "Sec Type", "Currency", "Exec Id",
                                                         # "Exec OrderId", "Exec Shares", "Last Liquidity"])
        self.success = False
        self.final_status = "Failed"
        # self.init_para()
 
    def nextValidId(self, orderId):
        self.nextOrderId = orderId
        self.start()
 
    def error(self, reqId , errorCode, errorString):
        msg = "Error: {}, {}, {}".format(reqId, errorCode, errorString)
        # length = len(self.error_msg)
        now = datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f")[:-3]
        self.error_msg.append([now, msg])

    def orderStatus(self, orderId , status, filled, remaining, avgFillPrice, permId,
                    parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        msg = "OrderStatus. Id: {} Status: {} Filled: {} Remaining: {} LastFillPrice: {}".format(orderId, status, filled, remaining, lastFillPrice)
        # length = len(self.orderStatus_msg)
        now = datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f")[:-3]

        self.orderStatus_msg.append([now, msg])
        self.final_status = status

        # print("msg: ", msg)
        if status == "Filled":
            self.success = True
            self.stop()

    def openOrder(self, orderId, contract, order, orderState):
        msg = "OpenOrder. ID: {}, {}, {} @ {} : {} {} {} {}".format(orderId, contract.symbol, contract.secType,
                                                                    contract.exchange, order.action, order.orderType,
                                                                    order.totalQuantity, orderState.status)
        # length = len(self.openOrder_msg)
        now = datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f")[:-3]

        self.openOrder_msg.append([now, msg])

    def execDetails(self, reqId, contract, execution):
        msg = "ExecDetails. {}, {}, {}, {}, {}, {}, {}, {}".format(reqId, contract.symbol, contract.secType,
                                                                   contract.currency, execution.execId,
                                                                   execution.orderId, execution.shares,
                                                                   execution.lastLiquidity)
        # length = len(self.execDetails_msg)
        now = datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f")[:-3]

        self.execDetails_msg.append([now, msg])

    def start(self):
        contract = Contract()
        # print("contract config = ", self.contract_config)
        contract.symbol = self.contract_config["ticker_name"]
        contract.secType = self.contract_config["sectype"]
        contract.exchange = self.contract_config["exchange"]
        contract.currency = self.contract_config["currency"]
        contract.primaryExchange = self.contract_config["primaryexchange"]
        contract.lastTradeDateOrContractMonth = self.contract_config["lasttradedateorcontractmonth"]
        # contract.waitToPlaceTrade = self.contract_config["waittoplacetrade"]

        order = Order()
        # print("order config = ", self.order_config)
        order.action = self.order_config["action"]
        order.totalQuantity = self.order_config["qty"]
        if "ordertype" in self.order_config:
            order.orderType = self.order_config["ordertype"]
        else:
            order.orderType = "MKT"
        order.lmtPrice = 50.0
        order.transmit = True

        self.placeOrder(self.nextOrderId, contract, order)


    def stop(self):
        self.done = True
        self.disconnect()


def place_order(contract_dict: dict, order_dict: dict):

    app = OrderApp(contract_dict, order_dict)
    app.nextOrderId = 0
    app.connect(ibgw_ip_address, ibgw_port_num, ibgw_client_num)
    # Start the socket in a thread
    # api_thread = Thread(target=app.run, daemon=True)
    # api_thread.start()
    # delay_time = int(contract_dict["waittoplacetrade"])
    # time.sleep(5)  # Sleep interval to allow time for connection to server
    Timer(5, app.stop).start()
    app.run()

    order_status = app.orderStatus_msg  # .drop_duplicates(ignore_index=True)
    open_order = app.openOrder_msg
    exec_detail = app.execDetails_msg
    errors = app.error_msg
    app.disconnect()

    return app.success, open_order, order_status, exec_detail, errors, app.final_status


if __name__ == "__main__":

    order_config = {}
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print("No exists such file: {}".format(config_file))
    else:
        contract_config = get_cfg(config_file)
        sys.exit(place_order(contract_config, order_config))
