import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def plot_truth_predict(df, classes=['Active\nLying','Active\nStanding','Drinking','Feeding','Non-active\nLying','Non-active\nStanding','Ruminating','X']):

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H-%M-%S", errors='coerce')

    total_duration = pd.Timedelta(hours=24)
    time_interval = total_duration / (len(df) - 1)
    df["Timestamp"] = pd.date_range(start=df["Timestamp"].min(), periods=len(df), freq=time_interval)
    
    print(df["Timestamp"].min(),df["Timestamp"].max())

    df["Timestamp"] = df["Timestamp"].dt.strftime("%H:%M:%S")
    df.set_index("Timestamp", inplace=True)

    y = df["Predict"]
    z = df["Truth"]

    plt.figure(figsize=(18, 6))
    plt.plot(y, marker='^', linestyle='', markersize=1, color='red')
    plt.plot(z, marker='o', linestyle='', markersize=1, color='blue')
    plt.xlabel("Timestamp")
    plt.ylabel("Predicted and Actual Behavior")
    plt.title("Daily Assessment - 1112")
    plt.grid(True)
    x_ticks = [f"{i:02d}:00" for i in range(0, 25, 3)]  # 0, 3, 6, 9, 12, 15, 18, 21, 24
    x_positions = np.linspace(0, len(df) - 1, len(x_ticks))
    plt.xticks(x_positions, x_ticks, rotation=0)
    plt.yticks(range(8), classes)
    plt.savefig("daily_assess/V9_1112_predict_truth.png")
    plt.show()


def plot_softmax_probs(df):

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H-%M-%S", errors='coerce')

    total_duration = pd.Timedelta(hours=24)
    time_interval = total_duration / (len(df) - 1)
    df["Timestamp"] = pd.date_range(start=df["Timestamp"].min(), periods=len(df), freq=time_interval)
    
    print(df["Timestamp"].min(),df["Timestamp"].max())

    df["Timestamp"] = df["Timestamp"].dt.strftime("%H:%M:%S")
    df.set_index("Timestamp", inplace=True)

    # Plotting
    plt.figure(figsize=(32, 4))

    # Plot each x0 to x6
    for i in range(7):
        plt.plot(df[f'x{i}'], label=f'x{i}')

    plt.xlabel('Time')
    plt.ylabel('Probabilties')
    plt.title('Plot of x0, x1, x2, x3, x4, x5, x6 over Time')
    plt.legend()

    x_ticks = [f"{i:02d}:00" for i in range(0, 25, 3)]  # 0, 3, 6, 9, 12, 15, 18, 21, 24
    x_positions = np.linspace(0, len(df) - 1, len(x_ticks))
    plt.xticks(x_positions, x_ticks, rotation=0)

    plt.tight_layout()
    plt.savefig("daily_assess/V9_1112_probs.png")
    plt.show()



if __name__ == "__main__":

    csv_file_path = "daily_assess/1112.csv"
    # csv_file_path = "inf_day_2023" + date + ".csv"
    df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf-8')

    print(df.info)
    plot_truth_predict(df)
    # plot_softmax_probs(df)
