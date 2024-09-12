# train test split

# 전체이미지에 대한 txt를 묶기
import tqdm
from glob import glob
import json
import os 
import shutil

def mk_splitfiles(root_dir, split_dir, is_train = True):

    for d1 in tqdm.tqdm(os.listdir(root_dir)):
        data_root_dir = os.path.join(root_dir, d1)

        for d2 in os.listdir(data_root_dir):
            data_root_dir2 = os.path.join(data_root_dir, d2)
            if d1 == '정상':
                class_num = 0
            else:
                class_num = int(d2[:2])
        
            for d3 in os.listdir(data_root_dir2):
                data_root_dir3 = os.path.join(data_root_dir2, d3)

                if is_train:
                    with open(os.path.join(split_dir,'trainlist_video.txt'), 'a') as f:
                                f.writelines(f'{class_num}/{d3}\n')
                else:
                    with open(os.path.join(split_dir,'testlist_video.txt'), 'a') as f:
                                f.writelines(f'{class_num}/{d3}\n')

                for d4 in os.listdir(data_root_dir3):
                    if is_train:
                        with open(os.path.join(split_dir,'trainlist.txt'), 'a') as v:
                            v.writelines(f'{class_num}/{d3}/{d4[:-3]}txt\n')
                    else:
                        with open(os.path.join(split_dir,'testlist.txt'), 'a') as v:
                            v.writelines(f'{class_num}/{d3}/{d4[:-3]}txt\n')


# 원본 데이터셋에서 yowo학습을 위한 rgb-image, label 구성
def mk_img_(root_dir, rgb_dir):
    for d1 in tqdm.tqdm(os.listdir(root_dir)):
        data_root_dir = os.path.join(root_dir, d1)

        for d2 in os.listdir(data_root_dir):
            data_root_dir2 = os.path.join(data_root_dir, d2)
            if d1 == '정상':
                class_num = 0
            else:
                class_num = int(d2[:2])
            
            # target dir 존재 안하면 만들기
            dst_dir = os.path.join(rgb_dir, str(class_num))
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            for d3 in os.listdir(data_root_dir2):
                data_root_dir3 = os.path.join(data_root_dir2, d3)
                dst_dir2 = os.path.join(dst_dir, d3)

                if not os.path.exists(dst_dir2):
                    os.makedirs(dst_dir2)
                
                for file in os.listdir(data_root_dir3):
                    # dst_file = os.path.join(dst_dir, file)
                    
                    src_file = os.path.join(data_root_dir3,file)

                    if os.path.isfile(src_file):
                        shutil.move(src_file, dst_dir2)
                        print(f"Moved {file} to {dst_dir2}")



def mk_label_(root_dir, label_dir):
    abdvtype = {'정상':0,'방향지시등 불이행':1,'실선구간 차로변경':2,'동시 차로 변경':3,'차선 물기':4,'2개 차로 연속 변경':5,'정체구간 차선변경':6,'안전거리 미확보 차선 변경':7}
    for d1 in tqdm.tqdm(os.listdir(root_dir)):
        data_root_dir = os.path.join(root_dir, d1)

        for d2 in os.listdir(data_root_dir):
            data_root_dir2 = os.path.join(data_root_dir, d2)
            if d1 == '정상':
                class_num = 0
            else:
                class_num = int(d2[:2])
            
            # target dir 존재 안하면 만들기
            dst_dir = os.path.join(label_dir, str(class_num))
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            for d3 in os.listdir(data_root_dir2):
                data_root_dir3 = os.path.join(data_root_dir2, d3)
                dst_dir2 = os.path.join(dst_dir, d3)

                if not os.path.exists(dst_dir2):
                    os.makedirs(dst_dir2)
                
                for file in os.listdir(data_root_dir3):
                    print(file)
                    txt_file = os.path.join(dst_dir2, file[:-4]+'txt')
                    json_file = os.path.join(data_root_dir3,file)

                    with open(json_file, 'rb') as jsfile:
                        json_data = json.load(jsfile)
                        bbox_list= []
                        driving_type_list = []

                        for i in range(len(json_data["annotation"])):
                            bbox_list.append(json_data["annotation"][i]["bbox"])
                            driving_type_list.append(json_data["annotation"][i]["DrivingType"])
                        

                        assert len(bbox_list) == len(driving_type_list)

                        with open(txt_file, 'w') as f:
                            for j in range(len(bbox_list)):
                                bbox_list[j][2] = bbox_list[j][0] + bbox_list[j][2]
                                bbox_list[j][3] = bbox_list[j][1] + bbox_list[j][3]

                                if class_num == 0:
                                    f.write(f'{0} {bbox_list[j][0]} {bbox_list[j][1]} {bbox_list[j][2]} {bbox_list[j][3]}\n')
                                else:
                                    f.write(f'{abdvtype[driving_type_list[j]]} {bbox_list[j][0]} {bbox_list[j][1]} {bbox_list[j][2]} {bbox_list[j][3]}\n')
                                    



if __name__ == '__main__':

    # 예시 디렉토리 경로

    img_root_dir = r'C:\Users\QBIC\Desktop\workspace\vc_datasets\01.원천데이터'
    label_root_dir = r'C:\Users\QBIC\Desktop\workspace\vc_datasets\02.라벨링데이터'

    split_dir = r'C:\Users\QBIC\Desktop\workspace\yowo_dataset_test'
    rgb_dir = r'C:\Users\QBIC\Desktop\workspace\yowo_dataset_test\rgb-images'
    label_dir = r'C:\Users\QBIC\Desktop\workspace\yowo_dataset_test\labels'

    try:
        os.mkdir(rgb_dir)
    except:
        print('already exist')

    try:
        os.mkdir(label_dir)
    except:
        print('already exist')

    # dataloader에서 참고하기 위한 txt파일 만들기
    print('make splitfiles')
    mk_splitfiles(img_root_dir,split_dir, True)

    # img dataset 구성
    print('make img dataset')
    mk_img_(img_root_dir, rgb_dir)

    # label dataset 구성
    print('make label dataset')
    mk_label_(label_root_dir, label_dir)
