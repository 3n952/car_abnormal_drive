# 한영상에 하나의 라벨(주행 종류)이 포함되어야 한다.(가정)

# 정상 주행파트의 프레임과 객체별 주행을 기록
# 나머지 주행 파트의 frame과 객체별 주행을 기록


import os
import json
from glob import glob
import shutil
from tqdm import tqdm
from traffic_datasplit import random_split
import re

def make_dataset(source_dir, src_img_dir, dst_img_dir, dst_label_dir):
    abdvtype = {'방향지시등 불이행':0,'실선구간 차로변경':1,'동시 차로 변경':2,'차선 물기':3,'2개 차로 연속 변경':4,'정체구간 차선변경':5,'안전거리 미확보 차선 변경':6}
    # root_dir = '비정상'에서 시작
    # d1 = 01.방향지시등 불이행 ... 등등
    for d1 in tqdm(os.listdir(source_dir)):
        class_num = int(str(d1)[:2]) - 1
        data_root_dir = os.path.join(source_dir, d1)

        # d2 = p01~~~_03의 영상 단위 폴더
        for d2 in os.listdir(data_root_dir):
            data_root_dir2 = os.path.join(data_root_dir, d2)

            # d3 = .png / .json 단위 파일 
            for d3 in os.listdir(data_root_dir2):
                data_root_dir3 = os.path.join(data_root_dir2, d3)

                # print(data_root_dir3)
                src_img_file = os.path.join(src_img_dir, d1,d2,d3[:-4]+'png')
        
                with open(data_root_dir3, 'rb') as jsfile:
                    json_data = json.load(jsfile)
                    bbox_list= []
                    driving_type_list = []

                    for i in range(len(json_data["annotation"])):
                        if json_data["annotation"][i]["DrivingType"] != '정상':
                            
                            imgmake_dir = os.path.join(dst_img_dir, str(class_num), d2)
                            labelmake_dir = os.path.join(dst_label_dir, str(class_num), d2)
                            try:
                                os.makedirs(imgmake_dir)
                                os.makedirs(labelmake_dir)
                            except FileExistsError:
                                pass
                            
                            #이미지 파일 데이터 셋 구성
                            shutil.copy2(src_img_file, imgmake_dir)

                            bbox_list.append(json_data["annotation"][i]["bbox"])
                            driving_type_list.append(json_data["annotation"][i]["DrivingType"])

                            dst_label_file = os.path.join(labelmake_dir, d3[:-4]+'txt')

                    #라벨 파일 데이터 셋 구성
                    try:
                        with open(dst_label_file, 'w') as tf:
                                for j in range(len(bbox_list)):
                                    bbox_list[j][2] = bbox_list[j][0] + bbox_list[j][2]
                                    bbox_list[j][3] = bbox_list[j][1] + bbox_list[j][3]

                                    try:
                                        tf.write(f'{abdvtype[driving_type_list[j]]} {float(bbox_list[j][0])} {float(bbox_list[j][1])} {float(bbox_list[j][2])} {float(bbox_list[j][3])}\n')
                                    except KeyError:
                                        print(f'Key Error: {driving_type_list[j]}' )
                                
                    except UnboundLocalError:
                        continue
                                    
def mk_splitfiles(root_dir, split_dir, is_train = True):

    # d1 = class num
    for d1 in tqdm(os.listdir(root_dir)):
        data_root_dir = os.path.join(root_dir, d1)

        # 영상단위
        for d2 in os.listdir(data_root_dir):
            data_root_dir2 = os.path.join(data_root_dir, d2)
            if not is_train:
                 with open(os.path.join(split_dir,'testlist_video.txt'), 'a') as f:
                     f.writelines(f'{d1}/{d2}\n')
            # 프레임단위
            for d3 in os.listdir(data_root_dir2):
                if is_train:
                    with open(os.path.join(split_dir,'trainlist.txt'), 'a') as v:
                        v.writelines(f'{d1}/{d2}/{d3[:-3]}txt\n')
                else:
                    with open(os.path.join(split_dir,'testlist.txt'), 'a') as v:
                        v.writelines(f'{d1}/{d2}/{d3[:-3]}txt\n')

# def rename_files_in_directory(directory):
#     # 파일명에서 숫자를 추출하기 위한 정규 표현식
#     pattern = re.compile(r'p02_(\d{4})\.png')
    
#     # 디렉토리 안의 파일 목록 가져오기
#     files = os.listdir(directory)
#     last_part = files.split('_')[-1]

#     # 이미지 파일 목록에서 숫자 추출 및 정렬
#     file_numbers = []
#     for file in files:
#         match = pattern.match(file)
#         if match:
#             file_numbers.append(int(match.group(1)))

#     # 숫자 순으로 정렬
#     file_numbers.sort()

#     # 연속적인지 확인
#     if all(file_numbers[i] + 1 == file_numbers[i + 1] for i in range(len(file_numbers) - 1)):
#         print("숫자가 연속적입니다.")
#         # 파일명을 순서대로 0001, 0002, ... 형식으로 변경
#         for i, number in enumerate(file_numbers):
#             old_name = f"p02_{number:04d}.png"
#             new_name = f"p02_{i + 1:04d}.png"
#             os.rename(os.path.join(directory, old_name), os.path.join(directory, new_name))
#             print(f"Renamed {old_name} to {new_name}")
#     else:
#         print("숫자가 연속적이지 않습니다.")

# # 디렉토리 경로 지정
# directory_path = "/path/to/your/directory"
# rename_files_in_directory(directory_path)

                    

if __name__ == '__main__':
    source_dir = r'D:\task2\289.국도 CCTV 영상을 통한 비정상주행 판별 데이터\01-1.정식개방데이터\Validation\라벨링데이터\비정상'
    src_img_dir = r'D:\task2\289.국도 CCTV 영상을 통한 비정상주행 판별 데이터\01-1.정식개방데이터\Validation\원천데이터\비정상'

    dst_img_dir = r'D:\last_dataset\rgb-images'
    dst_label_dir = r'D:\last_dataset\labels'

    make_dataset(source_dir, src_img_dir, dst_img_dir, dst_label_dir)

    root_dir = r'D:\last_dataset\rgb-images'
    split_dir = r'D:\last_dataset'

    mk_splitfiles(root_dir=root_dir, split_dir=split_dir)

    train_list = r'D:\last_dataset\trainlist.txt'
    test_list = r'D:\last_dataset\testlist.txt'
    random_split(train_list, train_list, test_list)