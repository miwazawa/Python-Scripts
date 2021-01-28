# -*- coding: utf-8 -*-

# 最近傍法（プロトタイプが複数個）

import sys
import os
import numpy as np
from PIL import Image

# プロトタイプを格納する配列
train_num = 100
train_img = np.zeros((10,train_num,28,28), dtype=np.float32)

# プロトタイプの読み込み
for i in range(10):
    for j in range(1,train_num+1):
        train_file = "mnist/train/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
        train_img[i][j-1] = np.asarray(Image.open(train_file).convert('L')).astype(np.float32)

# 混同行列
result = np.zeros((10,10), dtype=np.int32)
for i in range(10):
    for j in range(1,101):
        # 未知パターンの読み込み
        pat_file = "mnist/test/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
        pat_img = np.asarray(Image.open(pat_file).convert('L')).astype(np.float32)

        min_val = float('inf')
        ans = 0
        # 最近傍法
        for k in range(10):
            for l in range(1,train_num+1):
                # SSDの計算
                t = train_img[k][l-1].flatten()
                p = pat_img.flatten()
                dist = np.dot( (t-p).T , (t-p) )

                # 最小値の探索
                if dist < min_val:
                    min_val = dist
                    ans = k

        # 結果の出力
        result[i][ans] +=1
        print( i , j , "->" , ans )
        
print( "\n [ 混同行列 ]" )
print( result )
print( "\n 正解数 -> " , np.trace(result) )

