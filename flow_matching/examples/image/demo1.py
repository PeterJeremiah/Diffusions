#import h5py
#import sys
#
## 检查是否提供了文件路径
#if len(sys.argv) != 2:
#    #print("请提供 H5 文件的路径")
#    #print("用法: python check_h5.py /path/to/your/file.h5")
#    sys.exit(1)
#
#file_path = sys.argv[1]
#print(f"--- 正在检查 H5 文件: {file_path} ---")
#
#try:
#    with h5py.File(file_path, 'r') as f:
#        print("文件中的顶层键 (Top-level keys):")
#        # 打印所有顶层键名
#        keys = list(f.keys())
#        print(keys)
#        
#        # (可选) 打印所有键名（包括嵌套的）
#        # print("\n所有键名 (All keys):")
#        # f.visit(print)
#
#except Exception as e:
#    print(f"打开文件失败: {e}")
#
#print("-----------------------------------")




import h5py
import sys
import numpy as np

# 检查是否提供了文件路径
if len(sys.argv) != 2:
    print("请提供 H5 文件的路径")
    print("用法: python check_h5_count.py /path/to/your/file.h5")
    sys.exit(1)

file_path = sys.argv[1]
print(f"--- 正在检查 H5 文件: {file_path} ---")

try:
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        print("文件中的顶层键 (Top-level keys):")
        print(keys)
        print("\n每个顶层键的数据量:")

        for key in keys:
            dataset = f[key]
            shape = dataset.shape
            dtype = dataset.dtype
            # 数据量（元素总数）
            num_elements = np.prod(shape)
            print(f"  - {key}: shape={shape}, dtype={dtype}, 元素总数={num_elements}")

except Exception as e:
    print(f"打开文件失败: {e}")

print("-----------------------------------")
