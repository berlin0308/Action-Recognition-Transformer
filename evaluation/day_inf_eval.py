import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def plot_time_series(df, classes=['Active\nLying','Active\nStanding','Drinking','Feeding','Non-active\nLying','Non-active\nStanding','Ruminating','X']):

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H-%M-%S", errors='coerce')

    total_duration = pd.Timedelta(hours=24)
    time_interval = total_duration / (len(df) - 1)
    df["Timestamp"] = pd.date_range(start=df["Timestamp"].min(), periods=len(df), freq=time_interval)
    
    print(df["Timestamp"].min(),df["Timestamp"].max())

    df["Timestamp"] = df["Timestamp"].dt.strftime("%H:%M:%S")
    df.set_index("Timestamp", inplace=True)

    y = df["Action"]
    z = df["Truth"]

    # custom_order = [0, 2, , 1, 5, 3]  # 这是一个示例顺序，你可以根据需要进行修改
    # df["Action"] = df["Action"].map({v: i for i, v in enumerate(custom_order)})
    # y = [classes[i] for i in df["Action"]]

    plt.figure(figsize=(18, 6))
    plt.plot(y, marker='^', linestyle='', markersize=1, color='red')
    plt.plot(z, marker='o', linestyle='', markersize=1, color='blue')
    plt.xlabel("Timestamp")
    plt.ylabel("Action")
    plt.title("A Day of a Dairy Calf")
    plt.grid(True)

    x_ticks = [f"{i:02d}:00" for i in range(0, 25, 3)]  # 0, 3, 6, 9, 12, 15, 18, 21, 24
    x_positions = np.linspace(0, len(df) - 1, len(x_ticks))
    plt.xticks(x_positions, x_ticks, rotation=0)
    plt.yticks(range(8), classes)
    plt.show()
    plt.savefig("V7-2_1112_plot.png")



if __name__ == "__main__":

    csv_file_path = "inf_day_20231112.csv"
    # csv_file_path = "inf_day_2023" + date + ".csv"
    df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf-8')

    print(df.info)
    plot_time_series(df)
