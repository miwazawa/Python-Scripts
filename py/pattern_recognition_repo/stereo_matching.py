# -*- coding: utf-8 -*-

# ステレオマッチング

# python の場合，インストールして下さい
# > pip install numpy
# > pip install pillow

import sys
import os
import numpy as np
from PIL import Image, ImageDraw

def get_brightness(search, template, x, y):
    sum = 0.0
    ans_x = 0
    ans_y = 0
    min_val = float( 'inf' )   
    for x in range(0,search.shape[1],1):
            for y in range(0,search.shape[0],1):

                s = right[y:y+template.shape[1], \
                    x:x+template.shape[0]].flatten()
                t = cropped.flatten()
                sum = np.dot( (t-s).T , (t-s) )

                # 最小値を記憶
                if min_val > sum:
                    min_val = sum
                    ans_x = x
                    ans_y = y
    return ans_x, ans_y

# 左画像の読み込み
left_file = "left1.jpg"
left_img = Image.open(left_file).convert('L')

# numpyに変換 -> (Y,X,channel)
left = np.asarray(left_img).astype(np.float32)
print( left.shape )
left_width = left.shape[1]
left_height = left.shape[0]
print( left_width , left_height )

# 右画像の読み込み
right_file = "right1.jpg"
right_img = Image.open(right_file).convert('L')

# numpyに変換 -> (Y,X,channel)
right = np.asarray(right_img).astype(np.float32)
print( right.shape )
right_width = right.shape[1]
right_height = right.shape[0]
print( right_width , right_height )

# 視差マップ
result = np.ones((left_height, left_width))

# 探索領域の大きさ
search_size = 21 // 2



# テンプレートの大きさ
template_size = 21 // 2 
for y in range(0,left_height,1):
    print(y)
    for x in range(0,left_width,1):

        

        # 左画像の座標(x,y）を中心としたテンプレートに類似した領域を
        # 右画像から検索し，その座標（ans_x,ans_y）を求めなさい
        
        #cropped = left[y-template_size:y+template_size, x-template_size:x+template_size]
        cropped = left[y:y+template_size*2+1, \
            x:x+template_size*2+1]
        cropped_height, cropped_width = cropped.shape[:2]

        #print(cropped_height, cropped_width)
        
        #s = right[y-template_size-bias:y+template_size+bias, \
        #    x-template_size-bias:x+template_size+bias].flatten()
        
        #print(sum)
        #for yy in range(0,cropped_height,1):
        #    for xx in range(0,cropped_width,1):
        #        sum = 0.0
        #        sum += ( right[y+yy][x+xx] - cropped[yy][xx] )**2
        ans_x, ans_y = get_brightness(right, cropped, x, y)

        #cropped_img = Image.fromarray(np.uint8(cropped))
        #cropped_img.save('ponpon/ponpon' + str(x) + str(y) + '.jpg')

        result[y,x]=(x-ans_x)**2+(y-ans_y)**2
        #result[y,x]=x+y

        # 枠の描画
        
        #draw.rectangle((x, y, x+template_size, y+template_size), outline=(255,0,0))


min = np.min( result )
max = np.max( result )
result = (result-min)/(max-min) * 255
print(result)

result_img = Image.fromarray(np.uint8(result))
result_img.save('result.jpg')

cropped_img = Image.fromarray(np.uint8(left_img_r))
cropped_img.save('q.jpg')
                         



