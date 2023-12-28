import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def analyze_csv(df, classes=['AL','AS','DR','FD','NL','NS','RM','X']):


    # print(df['Date'].value_counts())
    print("Average Score:",df['Score'].mean())
    # print(behavior_counts)

    print("\nBehavior Occurrences:")    
    behavior_counts = df["Truth"].value_counts()
    occur_counts = [0,0,0,0,0,0,0,0]
    for tup in behavior_counts.items():
        occur_counts[tup[0]] = tup[1]

    # print(occur_counts)
    for behavior, count in enumerate(occur_counts):
        print(f"class : {classes[behavior]}, count: {count} [{count} min] [{round(count/60,1)} hr]")


    print("\nBehavior Bouts:")
    bout_counts = [0,0,0,0,0,0,0,0]
    current_behavior = 8

    for _, row in df.iterrows():
        behavior = row["Truth"]
        
        if behavior != current_behavior:
            bout_counts[behavior] += 1
            current_behavior = behavior

    # print(bout_counts)
    for behavior, count in enumerate(bout_counts):
        print(f"class : {classes[behavior]}, bout: {count}")

    print("\nAverage Duration per Bout")
    ADPBs = []
    for i in range(8):
        count = occur_counts[i]
        bouts = bout_counts[i]
        ADPB = round(float(count / bouts), 2)
        ADPBs.append(ADPB)

    for behavior, count in enumerate(ADPBs):
        print(f"class : {classes[behavior]}, ADPB: {count} [{count} min]")

    return occur_counts, bout_counts, ADPBs

def write_statistics_csv(input_files, output_path):

    all_data = pd.DataFrame()

    classes = ['AL','AS','DR','FD','NL','NS','RM','X']
    columns = ['Date'] + [f'Occurs-{cls}' for cls in classes] + [f'Bouts-{cls}' for cls in classes] + [f'ADPB-{cls}' for cls in classes]
    all_data = pd.DataFrame(columns=columns)

    for file in input_files:
        df = pd.read_csv(file, delimiter=',', encoding='utf-8')
        occurs, bouts, ADPBs = analyze_csv(df)
        data_row = [file] + occurs + bouts + ADPBs
        all_data.loc[len(all_data)] = data_row

    all_data.to_csv(output_path, index=False)

if __name__ == "__main__":

    csv_file_path = "daily_assess/1112.csv"
    # csv_file_path = "inf_day_2023" + date + ".csv"
    df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf-8')

    occurs, bouts, ADPBs = analyze_csv(df)
    print(occurs, bouts, ADPBs)

    input_files = ["daily_assess/1112.csv","daily_assess/1103.csv"]
    write_statistics_csv(input_files, "output.csv")

