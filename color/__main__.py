import sys
from color.trained import get_color_in_shell
from color.download_data import get_data
from color.run import train_model

try:
    script_to_run = sys.argv[1]
    print("getting script_to_run", script_to_run)
except ValueError:
    print("something went wrong")
    sys.exit(1)

if script_to_run == "trained":
    while True:
        get_color_in_shell()

elif script_to_run == "run":
    print("Starting training!")
    train_model()

elif script_to_run == "download":
    print("Getting data")
    get_data()
