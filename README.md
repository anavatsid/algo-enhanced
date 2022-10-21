# LS Detection from Stock Graph

## Install dependencies
    opencv-python
    requests
    mss
    pillow
    pyqt5
<!--  
`python-dotenv` is for **_slack-notification_**. For slack-notification, please create `.env` file. -->

## How to run
First, you must run api server. After then, please run main.py.

Here, argument **_show_** is for displaying the detection result.

    usage: main.py [-h] [-i INPUT_TYPE] [--show] [--trade]
    
    optional arguments:
        -h, --help            show this help message and exit
        -i INPUT_TYPE, --input_type INPUT_TYPE
                                input type to be processed. (default: capture)
        --show                Showing process and result frame. (default: False)
        --trade               Decide if auto trade is enable or not. (default: False)


### Screen Capture

In case of screen capture, Please use the below command. If so, the available ticker names will be displayed from config file. From there, select one of them and select the specific area on screeen.
```commandline
python main.py -i capture --show --trade
```

https://user-images.githubusercontent.com/96384530/184595640-47954005-1199-4cfd-bf85-2fa601dcf57a.mp4

### UI Manual 
```
python main_app.py
```
By using combobox, please select the target Ticker config file. 
Click _**FLATTEN**_, **_REVERSE_**, **_BUY_**, **_SELL_** button and place order action. 