import torch
a= torch.arange(6).reshape(2,3)
# print(a)
# print('b:',a.repeat(2,2))
# print('c:',a.repeat(1, 10))    # 每列扩展到10倍
# print("d:", a.repeat(10, 1))    # 每行扩展到10倍

annoList = ['D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\154446334_5d41cd1375_b.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\1297451346_5b92bdac08_b.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\17156759330_5af4f5a5b8_k.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\321888854_3723b6f10b_b.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\3342804367_f43682bb80_b.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\3927754171_9011487133_b.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\5178670692_63a4365c9c_b.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\7178882742_f090f3ce56_k.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\2354829160_3f65a6bf6f_b.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\9210739293_2b0e0d991e_k.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\14666848163_8be8e37562_k.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\5013250607_26359229b6_b.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\15331928994_d5b82eb368_k.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\2311771643_f46392fcc0_b.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\126700562_8e27720147_b.txt',
'D:\\workspace\\workspace_python\\YOLO-v5\\datasets\\balloon\\labels\\train\\12288355124_5e340d3de3_k.txt']


annoContent = []
for file in annoList:
    with open(file, 'r') as fo:
        for line in fo.readlines():
            line = line.strip("\n")
            # if len(line) > 0:
            annoContent.append(line)
            print(file, line)

print(len(annoContent))
print(annoContent)