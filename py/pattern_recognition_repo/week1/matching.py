# -*- coding: utf-8 -*-

# テンプレートマッチング
#
# python の場合 pillow をインストールして下さい
# > pip install pillow

import sys
import os
from PIL import Image, ImageDraw

# 探索画像の読み込み
search_file = "search.jpg"
search_img = Image.open(search_file).convert('RGB')

search_width , search_height = search_img.size
print( search_width , search_height )

# テンプレートの読み込み
template_file = "template.jpg"
template_img = Image.open(template_file).convert('RGB')

template_width , template_height = template_img.size
print( template_width , template_height )

# テンプレートマッチング
min_val = float('inf')
ans_x = 0
ans_y = 0
for y in range(0,search_height-template_height,10):
    print(y)
    for x in range(0,search_width-template_width,10):
        sum = 0.0
        # SSDの計算
        for yy in range(template_height):
            for xx in range(template_width):
                s = search_img.getpixel((x+xx,y+yy))
                t = template_img.getpixel((xx,yy))
                for i in range(3):
                    sum += ( s[i] - t[i] ) * ( s[i] - t[i] )
        
        # 最小値を記憶
        if min_val > sum:
            min_val = sum
            ans_x = x
            ans_y = y

# 枠の描画
draw = ImageDraw.Draw(search_img)
draw.rectangle((ans_x, ans_y, ans_x+template_width, ans_y+template_height), outline=(255,0,0))

# 結果の保存
search_img.save("result.jpg")

# 結果の表示
search_img.show()


                    



