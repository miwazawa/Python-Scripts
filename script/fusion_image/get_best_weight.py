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
import copy
import itertools

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
def make_average_image(weight_list, image_type_num, first_image_dir, first_output_dir, label_list, yama_flag=False):
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
                    if yama_flag:
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


#一番強いやつ
def get_best_weight(argo_path, weight_list, label_list, image_type_num, yaml_dict, output_dir, split, header_rows, header_cols, top_num, csv_output_dir):
    
    once = True
    temp_dir_name = "temp_ave"
    temp_dir = "./" + temp_dir_name + "/"

    image_dir, image_label_list = summarize_images(yaml_dict, label_list, image_type_num)
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
        print("learning...\n")
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
    
    #evaluation_valueを降順にソートしたデータフレームの生成
    evaluation_values_df = only_good_eval_df.iloc[:, :].sort_values(by='c01', ascending=False)
    #header個所のみのdfと結合
    good_eval_df = pd.concat([header_df, evaluation_values_df])
    good_eval_df.to_csv(csv_output_dir + 'good_evaluation_values.csv', header=False, index=False, encoding="s-jis")
    #求められた最適重み画像を画像出力フォルダに生成する。
    best_weight = evaluation_values_df.iloc[0,-1]
    best_weight = best_weight.replace('(', '').replace(')', '').replace(',', '').split()
    #各ラベルフォルダがなかった時は生成する
    if not os.listdir(path=output_dir):
        for label in label_list:
            os.makedirs(output_dir + label)
    make_average_image([np.array([float(w) for w in best_weight],dtype=np.float64)], image_type_num, image_dir, output_dir, label_list)
    shutil.rmtree(image_dir)

#小数が雑(仕様上しょうがないらしい)なリストをキレイにする関数
def convert_correct_num(num_list):
    correct_list = []
    for num in num_list:
        correct_list.append(float(format(num, '.2f')))
    return correct_list
    
#yamlファイルから必要な情報をとってくる関数
def get_data_from_yaml(yaml_dir):
    yaml_dict = load_yaml(yaml_dir)
    
    argo_path = yaml_dict.pop('argo_path')
    minmax_values = yaml_dict.pop('minmax_values')
    if type(minmax_values[0]['min_value']) is float and type(minmax_values[0]['max_value']) is float:
        if minmax_values[0]['min_value'] > minmax_values[0]['max_value']:
            print("'minmax_values'の大小関係が不正です。")
            return
    else:
        print("'minmax_values'にはFloat型を渡してください。")
        return
    image_type_num = yaml_dict.pop('image_type_num')
    if type(image_type_num) is int:
        pass
    else:
        print("'image_type_num'にはInt型を渡してください。")
        return
    output_dir = yaml_dict.pop('best_image_output_path') 
    yama_flag = yaml_dict.pop('for_yama')
    if yama_flag:
        pass
    elif not yama_flag:
        pass
    else:
        print("'for_yama'にはBool値を渡してください。")
        return
    label_list = yaml_dict.pop('label_list')
    if type(label_list) is list:
        pass
    else:
        print("'label_list'にはList型を渡してください。")
        return
    top_num = yaml_dict.pop('top_num')
    if type(top_num) is int:
        pass
    else:
        print("'top_num'にはInt型を渡してください。")
        return
    header_rows = yaml_dict.pop('header_rows')
    if type(header_rows) is int:
        pass
    else:
        print("'header_rows'にはInt型を渡してください。")
        return
    header_cols = yaml_dict.pop('header_cols')
    if type(header_cols) is int:
        pass
    else:
        print("'header_cols'にはInt型を渡してください。")
        return
    train_data_rate = yaml_dict.pop('train_data_rate')
    if type(train_data_rate) is float:
        pass
    else:
        print("'train_data_rate'にはFloat型を渡してください。")
        return
    csv_output_path = yaml_dict.pop('csv_output_path')
    
    return yaml_dict, argo_path, minmax_values, image_type_num, output_dir, yama_flag, label_list, top_num, header_rows, header_cols, header_cols, train_data_rate, csv_output_path

#全条件の画像を一つのフォルダにまとめる関数
def summarize_images(yaml_dict, label_list, image_type_num):
    temp_dir_name = "temp_ori"
    temp_dir = "./" + temp_dir_name + "/"
    if temp_dir_name in os.listdir(path='./'):
        shutil.rmtree(temp_dir)
    for i,label in enumerate(label_list):
        if i is 0:
            os.makedirs(temp_dir)
        os.makedirs(temp_dir + label)
    image_label_list = []
    once = True
    for label in label_list:
        temp_yaml_dict = copy.deepcopy(yaml_dict)
        for i in range(len(yaml_dict)):
            image_num = 0
            #適当に辞書の中身を取得する
            popped = temp_yaml_dict.popitem()
            image_path = popped[1][0].pop('path')
            if once:
                image_label = get_label_from_path(image_path)
                image_label_list.append(image_label)
            images = load_file(image_path + label)
            for img in images:
                cv2.imwrite(temp_dir + label + "/" + str(i+image_num).zfill(4) + image_label + '.png', img)
                image_num += image_type_num
        once = False
    return temp_dir, image_label_list

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

#小数を与えると最大値が１の等差数列を作る関数
def get_list_from_resolution(resolution, max_val=1.0):
    weight_list = []
    weight_list.append(0.0)
    sum_num = resolution
    
    while sum_num <= max_val:
        weight_list.append(sum_num)
        sum_num += resolution
    
    return weight_list

#重みのリストを返す関数
def get_weight_list(yaml_dict, minmax_dict):
    min_value = minmax_dict[0].pop('min_value')
    max_value = minmax_dict[0].pop('max_value')
    image_weight_list = []
    weight_list = []
    temp_yaml_dict = copy.deepcopy(yaml_dict)
    for i in range(len(yaml_dict)):
        popped = temp_yaml_dict.popitem()
        popped[1][0].pop('path')
        image_weight_list.append(convert_correct_num(get_list_from_resolution(popped[1][0].pop('resolution'))))
        all_weight_pattern = list(itertools.product(*image_weight_list))
    weight_list = []
    for weight_pattern in all_weight_pattern:
        if min_value <= float(format(sum(weight_pattern), '.1f')) <= max_value:
            weight_list.append(weight_pattern) 
    return weight_list

def main(yaml_dir):

    yaml_dict, argo_path, minmax_dict, image_type_num, output_dir, yama_flag, label_list, top_num, header_rows, header_cols, header_cols, train_data_rate, csv_output_dir = get_data_from_yaml(yaml_dir)

    if yama_flag:
        input_weight = []
        WEIGHT_LIST = []
        image_dir, image_label_list = summarize_images(yaml_dict, label_list, image_type_num)
        for i in range(image_type_num):
            input_weight.append(float(input("条件:{} の画像の重みを入力してください。\n".format(image_label_list[i]))))
        #各ラベルフォルダがなかった時は生成する
        if not os.listdir(path=output_dir):
            for label in label_list:
                os.makedirs(output_dir + label)
        make_average_image([np.array([float(w) for w in input_weight],dtype=np.float64)], image_type_num, image_dir, output_dir, label_list, yama_flag)
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
    WEIGHT_LIST = get_weight_list(yaml_dict, minmax_dict)

    get_best_weight(argo_path, WEIGHT_LIST, label_list, image_type_num, yaml_dict, output_dir, train_data_rate, header_rows, header_cols, top_num, csv_output_dir)

if __name__ == "__main__":
    params = sys.argv
    if len(params) is 2:
        main(params[1])
    else:
        print("不正な引数です。")
    