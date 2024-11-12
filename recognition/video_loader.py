import requests
import json
import cv2
'''
its 국도 / 고속도로 cctv 데이터 불러오기
회원가입 후 api 키를 받아야 api를 불러올 수 있다.
'''
# CCTV URL에 접근하여 영상 데이터를 다운로드 
def cctv_downloader(cctv_url, cctv_name):
    try:
        response = requests.get(cctv_url)
        if response.status_code == 200:
            # CCTV 영상을 파일로 저장 (예: cctv_video.mp4)
            with open(f"assets/{cctv_name}.mp4", 'wb') as f:
                f.write(response.content)
            print(f"{cctv_name} 영상을 성공적으로 다운로드했습니다.")
        else:
            print(f"{cctv_name} 영상에 접근할 수 없습니다. 상태 코드: {response.status_code}")
    except Exception as e:
        print(f"오류 발생: {e}")

def cctv_player(cctv_url):
    cap = cv2.VideoCapture(cctv_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('CCTV Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    cctv_url = "http://cctvsec.ktict.co.kr/4305/TeZ2/qvSZaEmP5HWSg7jBylf8A0v19wsoghUgQpNjpD+4ZrrEQfRVL8un57Y+VLT8cb4Y4Y6GbvgCFxHifrmag=="
    cctv_downloader(cctv_url, 'test_video4')
    print('complete to downloading cctv video')
    cctv_player(cctv_url)
    
