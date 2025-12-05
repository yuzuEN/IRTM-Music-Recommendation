import os

root_dir = "Data"  # 你的 dataset 根目錄
output_file = "all_captions.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)

        # 只處理資料夾
        if not os.path.isdir(folder_path):
            continue
        
        # 搜尋資料夾內所有 txt
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read().strip()
                
                # 寫入大檔案
                # outfile.write(f"### {folder}/{filename} ###\n")
                outfile.write(content + "\n\n")   # 每篇貼文後加空行

print(f"Done! All txt merged into: {output_file}")
