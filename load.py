import torch
import os
import sys
sys.path.append('/root/notebooks/automl/CPMNet/networks')
def inspect_pth_file(file_path):
    """
    載入 .pth 檔案並嘗試輸出其內容。

    Args:
        file_path (str): .pth 檔案的路徑。
    """
    print(f"--- 開始檢查檔案: {file_path} ---")

    if not os.path.exists(file_path):
        print(f"錯誤：檔案不存在於 '{file_path}'")
        print("--- 檢查結束 ---")
        return

    try:
        # 嘗試載入檔案
        # map_location='cpu' 確保即使模型是在 GPU 上保存的，也能在只有 CPU 的環境下載入
        # 如果您確定有匹配的 GPU 且需要載入到 GPU，可以移除或修改 map_location
        loaded_object = torch.load(file_path, map_location=torch.device('cpu'))
        print(f"成功載入檔案。載入的物件類型: {type(loaded_object)}")
        print("-" * 30)

        # --- 根據載入物件的類型進行檢查 ---

        if isinstance(loaded_object, dict):
            print("物件是一個字典 (可能是一個 state_dict 或 checkpoint):")
            print(f"字典包含 {len(loaded_object)} 個鍵:")
            for key, value in loaded_object.items():
                print(f"\n  鍵: '{key}'")
                print(f"  值的類型: {type(value)}")
                # 如果值是張量，顯示其形狀和數據類型
                if isinstance(value, torch.Tensor):
                    print(f"  值的形狀 (Shape): {value.shape}")
                    print(f"  值的數據類型 (Dtype): {value.dtype}")
                    # 可以取消註解下面這行來查看張量的前幾個元素，但對於大張量可能會輸出很多內容
                    # print(f"  值的前幾個元素 (範例): {value.flatten()[:5]}...")
                # 如果值是另一個字典，顯示其鍵
                elif isinstance(value, dict):
                     print(f"  值是一個字典，包含鍵: {list(value.keys())}")
                # 否則，直接打印值 (適用於簡單類型如 int, float, str)
                else:
                    # 限制輸出長度以防過長
                    value_str = str(value)
                    if len(value_str) > 100:
                         value_str = value_str[:100] + "..."
                    print(f"  值: {value_str}")

        elif isinstance(loaded_object, torch.Tensor):
            print("物件是一個 PyTorch 張量 (Tensor):")
            print(f"  形狀 (Shape): {loaded_object.shape}")
            print(f"  數據類型 (Dtype): {loaded_object.dtype}")
            print(f"  設備 (Device): {loaded_object.device}") # 顯示張量所在的設備
            # 可以取消註解下面這行來查看張量的前幾個元素
            # print(f"  前幾個元素 (範例): {loaded_object.flatten()[:10]}...")

        elif isinstance(loaded_object, torch.nn.Module):
            print("物件是一個完整的 PyTorch 模型 (nn.Module):")
            print("模型結構:")
            print(loaded_object)
            # 您也可以選擇查看模型的 state_dict
            # print("\n模型的 state_dict 鍵:")
            # print(list(loaded_object.state_dict().keys()))

        else:
            print("物件是一個無法詳細分類的 Python 物件:")
            # 嘗試打印物件，但要注意可能會很長
            obj_str = str(loaded_object)
            if len(obj_str) > 500:
                obj_str = obj_str[:500] + "\n... (內容過長，已截斷)"
            print(obj_str)

    except FileNotFoundError:
        print(f"錯誤：檔案 '{file_path}' 未找到。")
    except Exception as e:
        print(f"載入或檢查檔案時發生錯誤: {e}")
        import traceback
        traceback.print_exc() # 打印詳細的錯誤堆疊信息

    finally:
        print("\n--- 檢查結束 ---")

# --- 如何使用 ---
if __name__ == "__main__":
    # *** 將 'your_model.pth' 替換為您 .pth 檔案的實際路徑 ***
    pth_file_to_inspect = '/root/notebooks/automl/CPMNet/save/[2025-04-17-1224]_ME_db1/model/epoch_5.pth'

    inspect_pth_file(pth_file_to_inspect)
