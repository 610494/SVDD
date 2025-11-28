import json
import numpy as np

# 定義要處理的 JSON 檔案清單
json_files = ['log_LJ_vctk_libriTTS_100/LJSpeech.json', 'log_LJ_vctk_libriTTS_100/LJ_random.json', 'log_LJ_vctk_libriTTS_100/aishell3_train.json',
              'log_LJ_vctk_libriTTS_100_aishell3/LJSpeech.json', 'log_LJ_vctk_libriTTS_100_aishell3/LJ_random.json', 'log_LJ_vctk_libriTTS_100_aishell3/aishell3_train.json',
              'log_LJ_vctk_libriTTS_100_aishell3_hakkaradio_news/LJSpeech.json', 'log_LJ_vctk_libriTTS_100_aishell3_hakkaradio_news/LJ_random.json', 'log_LJ_vctk_libriTTS_100_aishell3_hakkaradio_news/aishell3_train.json',
              'log_LJ_vctk_libriTTS_100_hakkaradio_news/LJSpeech.json', 'log_LJ_vctk_libriTTS_100_hakkaradio_news/LJ_random.json', 'log_LJ_vctk_libriTTS_100_hakkaradio_news/aishell3_train.json',
              'log_LJ/LJSpeech.json', 'log_LJ/LJ_random.json', 'log_LJ/aishell3_train.json'
              ]  # 替換成你的檔案名稱

# 儲存所有結果的清單
results = []

# 逐一處理每個 JSON 檔案
for json_file in json_files:
    # 讀取 JSON 檔案
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # 提取分數數據
    scores = [score[0] for score in data['test_scores']]
    
    # 計算平均值
    mean_score = np.mean(scores)
    
    # 計算標準差
    std_dev = np.std(scores)
    
    # 計算四分位數
    q1 = np.percentile(scores, 25)
    q2 = np.percentile(scores, 50)  # Q2 也就是中位數
    q3 = np.percentile(scores, 75)
    
    # 將結果儲存為字典
    result = {
        'file': json_file,
        'mean': mean_score,
        'std_dev': std_dev,
        'q1': q1,
        'median': q2,
        'q3': q3
    }
    
    # 將結果添加到清單中
    results.append(result)

# 將結果寫入新的 JSON 檔案
with open('ana_output_results.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)

print("所有檔案處理完成，結果已儲存為 output_results.json")
