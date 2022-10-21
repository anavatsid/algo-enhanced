
def read_positions(target_symbol=None):  # read all accounts positions and return DataFrame with information

    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.common import TickerId
    from threading import Thread, Timer

    import pandas as pd
    import time

    class ib_class(EWrapper, EClient):

        def __init__(self):
            EClient.__init__(self, self)
            self.success = False
            self.all_positions = pd.DataFrame([], columns=['Account', 'Symbol', 'Quantity', 'Average Cost', 'Sec Type'])

        def error(self, reqId: TickerId, errorCode: int, errorString: str):
            if reqId > -1:
                print("Error. Id: ", reqId, " Code: ", errorCode, " Msg: ", errorString)

        def position(self, account, contract, pos, avgCost):
            index = str(account) + str(contract.symbol)
            self.all_positions.loc[index] = account, contract.symbol, pos, avgCost, contract.secType
            if target_symbol == contract.symbol:
                self.stop()
                self.success = True
                # print("\nSAmm============\n")

        def pnl(self, reqId: int, dailyPnL: float,
                      unrealizedPnL: float, realizedPnL: float):
            super().pnl(reqId, dailyPnL, unrealizedPnL, realizedPnL)
            print("Daily PnL Single. ReqId:", reqId, "Position:", "DailyPnL:", dailyPnL, "UnrealizedPnL:",
                  unrealizedPnL, "RealizedPnL:", realizedPnL, "Value:", )
        def stop(self):
            self.done = True
            self.disconnect()
    
    def run_loop():
        app.run()

    app = ib_class()
    app.connect('127.0.0.1', 4002, 8)
    # Start the socket in a thread
    api_thread = Thread(target=run_loop, daemon=True)
    api_thread.start()
    time.sleep(1)  # Sleep interval to allow time for connection to server
    # Timer(5, app.stop).start()
    # app.run()
    app.reqPositions()  # associated callback: position
    print("Waiting for IB's API response for accounts positions requests...\n")
    # time.sleep(3)
    for _ in range(6):
        if app.success:
            time.sleep(0.5)
            break
        time.sleep(0.5)
    
    current_positions = app.all_positions
    # print(current_positions, "\n")
    current_positions.set_index('Account', inplace=True, drop=True)  # set all_positions DataFrame index to "Account"
    app.disconnect()

    return current_positions


if __name__ == "__main__":
    print("Testing IB's API as an imported library:")
    for _ in range(1):
        
        all_positions = read_positions("AAPL")
        print(all_positions)
        print()
    # all_navs = read_navs()
    # print(all_navs)
