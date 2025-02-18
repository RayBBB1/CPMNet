import random

inFile = "train.txt"
outFile = "train300.txt"
outNumber = 300

patient_ids = []
try:
    with open(inFile,"r") as f :
        for line in f :
            patient_id = line.strip() # 移除行尾換行與空白符號
            if patient_id:
                patient_ids.append(patient_id)
except FileNotFoundError:
    print("錯誤: 找不到 train.txt 檔案。請確認檔案是否存在於相同目錄下。")
    exit()

selected_ids = random.sample(patient_ids,outNumber)

try:
    with open (outFile,"w") as f:
        for patient_id in selected_ids:
            f.write(patient_id + '\n')
except IOError:
    print("錯誤: 寫入 train300.txt 檔案時發生錯誤。請檢查檔案權限或磁碟空間。")
except Exception as e:
    print(f"發生未預期的錯誤: {e}")