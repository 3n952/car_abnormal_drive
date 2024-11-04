import os
import tqdm

def label_change(label_dir, output_dir):

    except_label = [0, 5]

    for dvtype in tqdm.tqdm(os.listdir(label_dir)):
        dvtype_dir = os.path.join(label_dir, dvtype)
        if int(dvtype) in except_label:
            print(f'\noriginal class {dvtype} is deleted')
            continue
        print(f"\ndriving type {dvtype} start")

        for video in os.listdir(dvtype_dir):
            frame_dir = os.path.join(dvtype_dir, video)
            for fname in os.listdir(frame_dir):
                input_file = os.path.join(frame_dir, fname)
            
                # 파일 읽고 클래스 변경 후 저장
                with open(input_file, 'r') as t:
                    lines = t.readlines()
            
                if dvtype == '1':
                    dvtype_ch = 0
                elif dvtype == '2':
                    dvtype_ch = 1
                elif dvtype == '3':
                    dvtype_ch = 2
                elif dvtype == '4':
                    dvtype_ch = 3
                elif dvtype == '6':
                    dvtype_ch = 4
                else: 
                    print('filtering except label')

                try:
                    os.makedirs(os.path.join(output_dir, str(dvtype_ch), video))
                except:
                    pass

                ouput_loc = os.path.join(output_dir, str(dvtype_ch), video, fname)
                with open(ouput_loc, 'w') as f:
                    for line in lines:
                        parts = line.rstrip().split()
                        
                        # 정상 주행을 제외한 라벨(0~7).txt 파일에서 필요한 라벨만 학습하기 위함 
                        # 0: 실선구간 차선변경
                        # 1: 동시 차로변경
                        # 2: 차선물기
                        # 3: 2개 차로 연속 변경
                        # 4: 안전거리 미확보 차선변경
                    
                        if parts[0] == '1':
                            parts[0] = '0'
                        elif parts[0] == '2':
                            parts[0] = '1'
                        elif parts[0] == '3':
                            parts[0] = '2'
                        elif parts[0] == '4':
                            parts[0] = '3'
                        elif parts[0] == '6':
                            parts[0] = '4'
                        else:
                            continue

                        f.writelines(f'{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n')

def train_test_rebuild(train_txt, target_txt):
    # train_txt, test_txt는 학습을 위한 경로 지정 파일
    # 파일 읽고 클래스 변경 후 저장
    path_list = []

    with open(train_txt, 'r') as t:
        lines = t.readlines()
        for line in lines:
            parts = line.split('/')
            if parts[0] == '0' or parts[0] == '5':
                continue

            elif parts[0] == '1':
                parts[0] = '0'
            elif parts[0] == '2':
                parts[0] = '1'
            elif parts[0] == '3':
                parts[0] = '2'
            elif parts[0] == '4':
                parts[0] = '3'
            elif parts[0] == '6':
                parts[0] = '4'

            updated_path = '/'.join(parts)
            path_list.append(updated_path)
    
    with open(target_txt, 'w') as f:
        for path in path_list:
            f.writelines(path)
    

if __name__ == '__main__':

    #label_dir = r'D:\singlelabel_dataset\labels'
    #output_dir = r'D:\singlelabel_dataset\new_labels'
    #label_change(label_dir, output_dir)

    train_txt = r'D:\singlelabel_dataset\testlist.txt'
    target_txt = r'D:\singlelabel_dataset\new_testlist.txt'
    train_test_rebuild(train_txt, target_txt)