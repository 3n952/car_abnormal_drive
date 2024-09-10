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
            x_min, y_min, x_max, y_max = map(float, parts[1:])
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
        class_id, x_min, y_min, x_max, y_max = bbox
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
    image_path = r'C:\Users\QBIC\Desktop\ucf24\rgb-images\Basketball\v_Basketball_g01_c01\00009.jpg'   # 실제 이미지 경로
    bbox_path = r'C:\Users\QBIC\Desktop\ucf24\labels\Basketball\v_Basketball_g01_c01\00009.txt'   # 실제 어노테이션 txt 파일 경로


    draw_bboxes(image_path, bbox_path)