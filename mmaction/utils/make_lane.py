from PIL import Image, ImageDraw
import json
import os
import random


# 랜덤하게 2개 시각화

def draw_lane(img_dir, json_dir):
    # CSV 파일에 저장할 데이터 초기화
    dir_path = ['비정상', '정상']
    #dir_path2 = ['01. 방향지시등 불이행', '02. 실선구간 차선변경', '03. 동시 차로변경', '04. 차선 물기', '05. 2개 차로 연속 변경', '06. 정체구간 차선변경', '07. 안전거리 미확보 차선 변경']
    count = 0
    try:

        for path in dir_path:
            # path = 정상 or 비정상
            for dirname in os.listdir(os.path.join(json_dir, path)):
                # dirname = 01. 방향지시등 불이행 ... 07. 안전거리 미확보 차선 변경
                for filename in os.listdir(os.path.join(json_dir, path, dirname)):
                    # p01_20221103_072002_an1_036_03/p01_20221103_072002_an1_036_03_0001.png
                    for lastname in random.sample(os.listdir(os.path.join(json_dir, path, dirname, filename)), 1):
                        if lastname.endswith('.json'):  # JSON 파일만 처리
                            json_path = os.path.join(json_dir, path, dirname, filename, lastname)
                            # 이미지 경로
                            img_path = os.path.join(img_dir, path, dirname, filename, lastname[:-4]+'png')
                    

                            with open(json_path, 'r', encoding='utf-8') as json_file:
                                count += 1
                                total_lane = []
                                data = json.load(json_file)
                                for i in range(len(data['annotationImage'])):
                                    # annotation의 annotation image의 polyline 추출
                                    # 한 세트에 대한 coord
                                    lane_coord = data['annotationImage'][i]['polyline']

                                    # 리스트에 x,y 좌표 튜플 형식으로 저장
                                    points = [(point['x'], point['y']) for point in lane_coord]
                                    # 한 이미지에 대한 coord 좌표 저장
                                    total_lane.append(points)
                                
                                img = Image.open(img_path)
                                draw = ImageDraw.Draw(img)

                                for points in total_lane:
                                    draw.line(points, fill = 'red', width = 2)
                                
                                # 이미지에 차선 그리기 
                                img.show()
                                
                                if count >= 2:
                                    raise StopIteration
                            
    except StopIteration:
        pass

    return total_lane
                        
def extract_lane_coord(json_dir):
    # json_dir = ../datasets\02.라벨링데이터\비정상\01. 방향지시등 불이행\p01_20221103_072002_an1_036_03 와 같은 꼴로 입력되어야함.
    try:

        for path in os.listdir(json_dir):
            if path.endswith('.json'):  # JSON 파일만 처리
                json_path = os.path.join(json_dir, path)                                                                
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    total_lane = []
                    data = json.load(json_file)
                    for i in range(len(data['annotationImage'])):
                        # annotation의 annotation image의 polyline 추출
                        # 한 세트에 대한 coord
                        lane_coord = data['annotationImage'][i]['polyline']

                        # 리스트에 x,y 좌표 튜플 형식으로 저장
                        points = [(point['x'], point['y']) for point in lane_coord]
                        total_lane.append(points)

                    raise StopIteration
                    
    except StopIteration:
            pass
    
    # 같은 이미지에 대한 lane coord return
    return total_lane      



if __name__ == '__main__':

    img_dir = r'C:\Users\QBIC\Desktop\workspace\car_abnormal_driving\datasets\01.원천데이터'
    json_dir = r'C:\Users\QBIC\Desktop\workspace\car_abnormal_driving\datasets\02.라벨링데이터'

#     json_detail_dir = '../datasets/02.라벨링데이터/비정상/01. 방향지시등 불이행/p01_20221103_072002_an1_036_03'

#     coord = extract_lane_coord(json_detail_dir)
#     print(coord)
    draw_lane(img_dir, json_dir )
