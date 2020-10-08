#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import cv2
import os
import glob
import subprocess
import pandas as pd
import shutil
import yaml

#yamlファイルを読み込む関数
def load_yaml(yaml_path, encoding="utf-8"):
    with open(yaml_path) as file:
        obj = yaml.safe_load(file)
    return obj

#ディレクトリ内のファイル全てを読み込む関数
def load_file(folder, fmt="png"):
    images = []
    files = sorted(glob.glob(folder + '/*.' + fmt))
    #print(files)
    if fmt == "png" or fmt == "bmp":
        for filename in files:
            #print(filename)
            img = cv2.imread(filename)
            if img is not None:
                images.append(img) 
        print("{} is loaded.\n".format(folder))
        return images
    else:
        for filename in files:
            if filename is not None:
                images.append(filename)
        print("{} is loaded.\n".format(folder))
        return images
     
#重み付き平均化画像の生成関数
def make_average_image(weight_list, image_type_num, first_image_dir, first_output_dir, label_list, for_yama=False):
    len_weight_list = len(weight_list)
    if first_output_dir[-1] != "/":
        first_output_dir += "/"
    if len_weight_list==0:
        print("わーにんぐ！：重みリストが空です。\n")
        return
    for label in label_list: 
        cnt = 0
        average_image = 0
        image_dir  = first_image_dir + label + "/"
        output_dir = first_output_dir + label + "/"
        image_list = load_file(image_dir)
        for w_i in range(len_weight_list):
            image_num = 0
            for i,img in enumerate(image_list):
                if i%image_type_num is not image_type_num-1:
                    average_image += img*weight_list[w_i][cnt] 
                    cnt += 1
                else:
                    average_image += img*weight_list[w_i][cnt]  
                    if for_yama == True:
                        cv2.imwrite(output_dir + '/average_image' + str(image_num).zfill(3) + "_" + label + '.png', average_image)
                    else:
                        cv2.imwrite(output_dir + '/average_image' + str(image_num).zfill(3) + '.png', average_image)
                    cnt,average_image = 0,0
                    image_num += 1
            print("重み[{}]の{}ラベル平均画像が生成されました。\n" .format(weight_list[w_i],label))


#出力されたcsvファイルを全て読み込み、上位top_num個のデータをまとめる関数
def save_good_eval_asCSV(csv_files,top_num, header_rows, output_dir):
    once=True

    #一番左の行に無駄なindex行が生成されるため、以下全てでiloc[1:,:]としている
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if once is True:
            #headerだけのdf
            header_df = df.iloc[:header_rows, 1:]
            #データだけのdf
            only_good_eval_df = df.iloc[header_rows:header_rows + top_num + 1, 1:]
            once = False
        else:
            #header個所を除去する
            temp_df = df.iloc[header_rows:, 1:]

            #データ部分のみのdfを結合していく
            only_good_eval_df = pd.concat([only_good_eval_df, temp_df])

    #evaluation_valueを降順にソートしたデータフレームの生成
    evaluation_values_df = only_good_eval_df.iloc[:, :].sort_values(by='c01', ascending=False)

    #header個所のみのdfと結合
    good_eval_df = pd.concat([header_df, evaluation_values_df])

    good_eval_df.to_csv(output_dir + 'good_evaluation_values.csv', header=False, index=False, encoding="s-jis")


#4つの全パターンの重み値を返す関数
#これは4つのパターンに限定されているが、image_type_numを変えても対応するようにしたい→get_weight_listで実装完了
#使ってないから消してもいい
def get_all_4_weight_values(image_type_num, resolution):
    WEIGHT_LIST = []

    list = [num for num in range(0,11,int(resolution*10))]

    for i in list:
        for j in list:
            for p in list:
                for q in list:
                    w = [i, j, p, q]
                    if sum(w)==10:
                        WEIGHT_LIST.append(w)

    normalize_list = [[num/10 for num in WEIGHT_LIST[i]] for i in range(len(WEIGHT_LIST))]
    #print(normalize_list)
    
    return normalize_list

#5つの全パターンの重み値を返す関数
#使ってないから消してもいい
def get_all_5_weight_values(image_type_num, resolution):
    WEIGHT_LIST = []

    list = [num for num in range(0,11,int(resolution*10))]

    for i in list:
        for j in list:
            for p in list:
                for q in list:
                    for r in list:
                        w = [i, j, p, q, r]
                        if sum(w)==10:
                            WEIGHT_LIST.append(w)

    normalize_list = [[num/10 for num in WEIGHT_LIST[i]] for i in range(len(WEIGHT_LIST))]
    #print(normalize_list)
    
    return normalize_list

#一番強いやつ
def get_best_weight(weight_list, label_list, image_type_num, yaml_dir, output_dir, split, header_rows, header_cols, top_num, csv_output_dir):
    
    once = True
    temp_dir_name = "temp_ave"
    temp_dir = "./" + temp_dir_name + "/"

    image_dir, image_label_list, argo_path = summarize_images(yaml_dir, label_list, image_type_num)

    if temp_dir_name in os.listdir(path='./'):
        shutil.rmtree(temp_dir)

    for progress,weight in enumerate(weight_list):
        for i,label in enumerate(label_list):
            if i is 0:
                os.makedirs(temp_dir)
            os.makedirs(temp_dir + label)

        make_average_image([weight], image_type_num, image_dir, temp_dir, label_list)

        #まずNG画像を読み込む
        image_list = load_file(temp_dir + "NG/")

        #NG画像をargoのフォルダに移動させる。
        for i,img in enumerate(image_list):
            if i == len(image_list):
                    break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #ここngmasksとちゃんと同じ名前にするように注意する
            cv2.imwrite(argo_path + 'Image/test_ng/average_image' + str(i).zfill(3) + '.png', img)
        
        print("test_ng is ready.\n")

        image_list = load_file(temp_dir + "OK/")
        
        #OK画像をargoのフォルダに移動させる。
        for i,img in enumerate(image_list):
            border_index = len(image_list)*split
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if i >= border_index:
                cv2.imwrite(argo_path + 'Image/test_ok/average_image' + str(i).zfill(3) + '.png', img)
            else:
                cv2.imwrite(argo_path + 'Image/learn/average_image' + str(i).zfill(3) + '.png', img)
        print("learn and test_ok are ready.\n")
        
        #argoの実行
        print("learning...")
        subprocess.run('run.bat', cwd=argo_path, shell=True)
        
        #以下数行で重み情報を付する
        if once is True:
            col_names = [ 'c{0:02d}'.format(i) for i in range(header_cols) ]
            weight_label = 'weight(['
            for i,image_label in enumerate(image_label_list):
                if i != len(image_label_list)-1:
                    weight_label += image_label + ', '
                else:
                    weight_label += image_label + '])'

        #argo.yamlで指定したcsvの出力先ディレクトリと対応させる
        df = pd.read_csv(argo_path + 'evaluation_values.csv', names=col_names)
        df['new_col'] = ''
        df.iat[2, -1] = weight_label
        df.iloc[3:, -1] = str(weight)
        
        print("csv file is made correctly!\n")

        #一番左の行に無駄なindex行が生成されるため、以下でiloc[1:,:]としている
        if once is True:
            #headerだけのdf
            header_df = df.iloc[:header_rows, 1:]
            #データだけのdf
            only_good_eval_df = df.iloc[header_rows:header_rows + top_num + 1, 1:]
            once = False
        else:
            #header個所を除去する
            temp_df = df.iloc[header_rows:, 1:]
            #データ部分のみのdfを結合していく
            only_good_eval_df = pd.concat([only_good_eval_df, temp_df]) 
        shutil.rmtree(temp_dir_name)
        del image_list
        print("progress : {}/{}\n" .format(progress+1,len(weight_list)))
    
    shutil.rmtree(image_dir)
    #evaluation_valueを降順にソートしたデータフレームの生成
    evaluation_values_df = only_good_eval_df.iloc[:, :].sort_values(by='c01', ascending=False)
    #header個所のみのdfと結合
    good_eval_df = pd.concat([header_df, evaluation_values_df])
    good_eval_df.to_csv(csv_output_dir + 'good_evaluation_values.csv', header=False, index=False, encoding="s-jis")
    #求められた最適重み画像を画像出力フォルダに生成する。
    best_weight = evaluation_values_df.iloc[0,-1]
    best_weight = best_weight.replace('[', '').replace(']', '').split()
    make_average_image([np.array([float(w) for w in best_weight],dtype=np.float64)], image_type_num, image_dir, output_dir, label_list)


#再帰的に全重み値を導出する関数
def func(weight, resolution, n=0):
    global count
    for i,w in enumerate(weight):
        if np.allclose(w, 0):
            weight[i] = 0.0       
    if np.allclose(weight.sum(), 1):
        #print(weight)
        for w in weight:
            temp_list.append(w)        
        weight[n] -= resolution
        count = count+1
        #print(weight_list)
        return
    else:
        a=n
        for i in range(a,len(weight)):
            weight[i] += resolution
            func(weight, resolution, i)
        weight[a] -= resolution

#小数が雑(仕様上しょうがないらしい)なリストをキレイにする関数
def convert_correct_num(num_list):
    correct_list = []
    for num in num_list:
        correct_list.append(float(format(num, '.2f')))
    return correct_list
    
#1次元リストを適切な2次元の行列の形に直す関数
def conver_weight_list(num_list, image_type_num):
    length = len(num_list)/image_type_num
    length = int(length)
    weight_list = np.zeros([length, image_type_num])
    c = 0
    for i in range(length):
        for j in range(image_type_num):
            weight_list[i][j] = num_list[c]
            c += 1
    return weight_list

#分解能と画像種類数を入力するとそれに対する重み値のリストを返す関数
def get_weight_list(resolution, image_type_num):
    global temp_list
    global count
    temp_list = []
    count = 0

    weight = np.zeros([image_type_num])
    weight_list = []

    func(weight,resolution)

    return conver_weight_list(convert_correct_num(temp_list), image_type_num)

#全条件の画像を一つのフォルダにまとめる関数
def summarize_images(yaml_dir, label_list, image_type_num):
    temp_dir_name = "temp_ori"
    temp_dir = "./" + temp_dir_name + "/"
    if temp_dir_name in os.listdir(path='./'):
        shutil.rmtree(temp_dir)
    for i,label in enumerate(label_list):
        if i is 0:
            os.makedirs(temp_dir)
        os.makedirs(temp_dir + label)
    once = True
    for label in label_list:
        yaml_dict = load_yaml(yaml_dir)
        if once:
            #argoのパスを取得する
            argo_path = yaml_dict.pop('argo_path')
            image_label_list = []
        for i in range(len(yaml_dict)):
            image_num = 0
            #適当に辞書の中身を取得する
            popped = yaml_dict.popitem()
            image_label = get_label_from_path(popped[1])
            image_label_list.append(image_label)
            images = load_file(popped[1] + label)
            for img in images:
                cv2.imwrite(temp_dir + label + "/" + str(i+image_num).zfill(4) + image_label + '.png', img)
                image_num += image_type_num
    return temp_dir, image_label_list, argo_path

#最下層のディレクトリの名前をとってくる関数
def get_label_from_path(path):
    label=""
    first_flag=True
    for c in reversed(path):
        if first_flag:
            if c=="\\":
                continue
            else:
                label += c
            first_flag=False
        else:
            if c=="\\":
                return ''.join(list(reversed(label)))
            else:
                label += c

def main(YAML_DIR=None, OUTPUT_DIR=None, IMAGE_TYPE_NUM=None, RESOLUTION=None, YAMA_FLAG=False):
    if YAML_DIR is None:
        YAML_DIR  = "./half_aligned/" #入力画像フォルダ(○○/OK or NG/image.pngみたいなディレクトリ構成)
    if OUTPUT_DIR is None:
        OUTPUT_DIR = "./hoge/" #最適重み画像の出力先(ラベルのフォルダも自分で作成する)
    if IMAGE_TYPE_NUM is None:
        IMAGE_TYPE_NUM = 4 #画像の種類数
    if RESOLUTION is None:
        RESOLUTION = 0.1 #重み値を全パターン受け取るときの分解能的なやつ、勿論小さいほうが精度は良くなるが時間がかかる(1を割り切る必要があるため,0.1 OR 0.2 OR 0.5 OR 1.0)
    LABEL_LIST = ["OK","NG"] #ラベルのリスト

    SPLIT = 0.7 #(train:test = SPLIT:(1-SPLIT))
    TOP_NUM = 1 #各重み画像csvからトップ${TOP_NUM}のevaluation valueを持ってくるか。gridサーチの範囲内にしないと学習した結果が全て消滅するので細心の注意を払って定義すること
    HEADER_ROWS = 3 #今回使用するcsvファイルのheaderの行数は３
    HEADER_COLS = 31 #今回使用するcsvファイルのheaderの列数は３
    CSV_OUTPUT_DIR = "./" #出力されたevaluation_value.csvの評価値トップ${TOP_NUM}をまとめたcsvの出力先

    if YAMA_FLAG:
        input_weight = []
        WEIGHT_LIST = []
        image_dir, image_label_list = summarize_images(YAML_DIR, LABEL_LIST, IMAGE_TYPE_NUM)[:2]
        for i in range(IMAGE_TYPE_NUM):
            input_weight.append(float(input("条件:{} の画像の重みを入力してください。\n".format(image_label_list[i]))))
        make_average_image([np.array([float(w) for w in input_weight],dtype=np.float64)], IMAGE_TYPE_NUM, image_dir, OUTPUT_DIR, LABEL_LIST, YAMA_FLAG)
        shutil.rmtree(image_dir)
        return

    #手動で重みを設定したいならココ！！
    '''
    WEIGHT_LIST = np.array([
        [1.0, 0.0, 0.0, 0.0],
        #[0.80, 0.05, 0.05, 0.05, 0.05],
        #[0.05, 0.80, 0.05, 0.05, 0.05],
        #[0.05, 0.05, 0.85, 0.05],
        #[0.05, 0.05, 0.05, 0.85],
        ],dtype=np.float64)
    '''

    #get_all_X_weight_valuesは汎用性がないためもう使わない（計算部分も冗長な部分があるので0.4～0.5秒ほど余計にかかる。）
    #やってることはget_weight_list関数と同じで上記関数の方が読みやすいので理解のため、一応残しておく
    #WEIGHT_LIST = np.array(get_all_4_weight_values(IMAGE_TYPE_NUM,RESOLUTION))
    WEIGHT_LIST = get_weight_list(RESOLUTION, IMAGE_TYPE_NUM)
    print(WEIGHT_LIST)

    #get_best_weight(WEIGHT_LIST, LABEL_LIST, IMAGE_TYPE_NUM, YAML_DIR, OUTPUT_DIR, SPLIT, HEADER_ROWS, HEADER_COLS, TOP_NUM, CSV_OUTPUT_DIR)

if __name__ == "__main__":
    params = sys.argv
    if len(params) is 1:
        main()
    elif len(params) is 3:
        main(params[1], params[2])
    elif len(params) is 5:
        main(params[1], params[2], int(params[3]), float(params[4]))
    elif len(params) is 6:
        if params[5] == "True":
            main(params[1], params[2], int(params[3]), float(params[4]), True)
        elif params[5] == "False":
            main(params[1], params[2], int(params[3]), float(params[4]), False)
        else:
            print("無効な引数です。")
    else:
        print("無効な引数です。")
    