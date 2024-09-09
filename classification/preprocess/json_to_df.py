import pandas as pd
import os
from glob import glob
import json
import csv


# 디렉토리 내의 모든 파일을 탐색
def json_to_csv(root_directory, output_csv):
    # CSV 파일에 저장할 데이터 초기화
    csv_data = []
    dir_path = ['01. 방향지시등 불이행', '02. 실선구간 차선변경', '03. 동시 차로변경', '04. 차선 물기', '05. 2개 차로 연속 변경', '06. 정체구간 차선변경', '07. 안전거리 미확보 차선 변경']

    for path in dir_path:
        for dirname in os.listdir(os.path.join(root_directory, path)):
            for filename in os.listdir(os.path.join(root_directory, path, dirname)):
                # print(os.path.join(root_directory, path, dirname, filename))

                if filename.endswith('.json'):  # JSON 파일만 처리
                    json_path = os.path.join(root_directory, path, dirname, filename)
                    with open(json_path, 'r', encoding='utf-8') as json_file:
                        data = json.load(json_file)

                        # annotation의 driving type과 imageInfo의 filename 추출
                        image_filename = data.get('imageInfo', {}).get('fileName', '')
                        for i in range(len(data['annotation'])):
                            driving_type = data['annotation'][i]['DrivingType']
                            # CSV 데이터에 추가
                            csv_data.append([image_filename, driving_type])

    # CSV 파일에 데이터 저장
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 헤더 추가
        csvwriter.writerow(['filename', 'drivingtype'])
        # 데이터 추가
        csvwriter.writerows(csv_data)

    print(f'{output_csv} 파일에 데이터가 저장되었습니다.')
    



# JSON 파일들이 있는 디렉토리 경로 설정
ndirectory = r'C:\Users\QBIC\Desktop\workspace\car_abnormal_driving\Sample\02.라벨링데이터\정상'
andirectory = r'C:\Users\QBIC\Desktop\workspace\car_abnormal_driving\Sample\02.라벨링데이터\비정상'

# 결과를 저장할 CSV 파일 이름 설정
normal_output_csv = 'normal_output.csv'
abnormal_output_csv = 'abnormal_output.csv'

json_to_csv(ndirectory, normal_output_csv)
json_to_csv(andirectory, abnormal_output_csv)
