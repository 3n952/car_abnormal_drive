import cv2
import os
import tqdm

def make_video(img_dir, output_dir, fps = 15.0):

    for dvtype in os.listdir(img_dir):
        dvtype_dir = os.path.join(img_dir, dvtype)
        for video in os.listdir(dvtype_dir):

            save_dir = os.path.join(output_dir, video+'.mp4')
            frame_dir = os.path.join(dvtype_dir, video)

            # 프레임 목록 가져오기 및 정렬
            frames = sorted([img for img in os.listdir(frame_dir) if img.endswith(".png")])

            # 첫 번째 프레임을 사용하여 영상의 크기 설정
            frame_path = os.path.join(frame_dir, frames[0])
            frame = cv2.imread(frame_path)
            height, width, _ = frame.shape
            size = (width, height)

            # VideoWriter 객체 생성
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 포맷 사용
            video = cv2.VideoWriter(save_dir, fourcc, fps, size)

            # 모든 프레임을 읽어와서 동영상에 추가
            for frame_name in frames:
                frame_path = os.path.join(frame_dir, frame_name)
                frame = cv2.imread(frame_path)
                video.write(frame)

            # VideoWriter 객체 해제
            video.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':

    img_dir = r'D:\singlelabel_dataset\rgb-images'
    output_dir = r'D:\singlelabel_dataset\video'
    tqdm.tqdm(make_video(img_dir, output_dir))
