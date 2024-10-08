import cv2
import matplotlib.pyplot as plt

def load_bbox_annotation(file_path):
    """
    txt 파일에서 bbox 어노테이션을 로드하는 함수
    """
    bboxes = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            # 공백을 기준으로 나눠서 bbox 좌표와 클래스를 읽음
            parts = line.split()
            class_id = int(parts[0])

            #anchor 확인용
            #x_min, y_min, x_max, y_max = map(float, parts[1:])

            _, x_min, y_min, x_max, y_max = map(float, parts[1:])
            
            bboxes.append((class_id, int(x_min), int(y_min), int(x_max), int(y_max)))

    return bboxes

def draw_bboxes(image_path, bbox_path):
    """
    이미지에 bbox를 그려서 시각화하는 함수
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # bbox 로드
    bboxes = load_bbox_annotation(bbox_path)
    
    # bbox 그리기
    for bbox in bboxes:

        ##anchor 확인용
        class_id, x_min, y_min, x_max, y_max = bbox

        #class_id, x_min, y_min, x_max, y_max = bbox

        # 사각형 그리기 (초록색)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        # 클래스 텍스트 표시
        cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
    
    # 이미지 출력
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':

    # 예시 경로
    image_path = r'C:\Users\QBIC\Desktop\workspace\car_abnormal_driving\recognition\custom_dataset\rgb-images\2\p02_20221223_123007_an2_011_03\p02_20221223_123007_an2_011_03_0002.png'   
    # 실제 이미지 경로
    image_path2 = r'C:\Users\QBIC\Desktop\workspace\car_abnormal_driving\recognition\custom_dataset\rgb-images\7\p01_20230107_141213_an7_065_04\p01_20230107_141213_an7_065_04_0020.png'


    #bbox_path1 = r'C:\Users\QBIC\Desktop\workspace\car_abnormal_driving\recognition\yowo_dataset_test\labels\0\p01_20221026_183003_n1_003_03\p01_20221026_183003_n1_003_03_0001.txt'   # 실제 어노테이션 txt 파일 경로
    
    # anchor
    # ('p10_20221103_065004_n2_270_04_0022.txt',
    # 'p10_20221103_065004_n2_196_04_0004.txt', 
    # 'p01_20221227_175001_n5_062_04_0015.txt', 
    # 'p08_20221004_143003_n6_004_05_0022.txt', 
    # 'p01_20221214_173005_n5_089_04_0012.txt', 
    # 'p01_20221224_161010_n5_055_05_0016.txt', 
    # 'p01_20230106_071002_an3_036_07_0004.txt', 
    # 'p01_20221213_105002_n3_007_06_0021.txt')

    bbox_path = r'C:\Users\QBIC\Desktop\workspace\car_abnormal_driving\recognition\custom_detections\test_2\detections_2\p01_20221026_183003_n1_003_03_0001.txt'
    bbox_path2 = r'C:\Users\QBIC\Desktop\workspace\car_abnormal_driving\recognition\custom_detections\detections_1\p02_20221223_123007_an2_011_03_0002.txt'

    #draw_bboxes(image_path, bbox_path)
    draw_bboxes(image_path, bbox_path2)