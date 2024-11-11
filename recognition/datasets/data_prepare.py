# raw dataset에서 multi , single .. dataset으로 변환 및 파싱하여 모델의 입력으로 처리
import to_multi
import to_single
import os
from glob import glob
import os 


'''
raw dataset은 ai hub의 국도 cctv 데이터의 출처 
sample 폴더를 3개를 준비해야함.
'''

# rawdataset for multi
img_root_dir = r'D:\Sample\01.원천데이터'
label_root_dir = r'D:\Sample\02.라벨링데이터'

# rawdataset for single
source_dir = r'D:\Sample2\02.라벨링데이터\비정상'
src_img_dir = r'D:\Sample2\01.원천데이터\비정상'

# multidataset의 test용 영상 frame 구성하기 ===============================================

multi_split_dir = r'D:\multilabel_test'
multi_rgb_dir = r'D:\multilabel_test\rgb-images'
multi_label_dir = r'D:\multilabel_test\labels'
try:
    os.mkdir(multi_rgb_dir)
except:
    print('already exist')

try:
    os.mkdir(multi_label_dir)
except:
    print('already exist')

# dataloader에서 참고하기 위한 txt파일 만들기
to_multi.mk_splitfiles(label_root_dir, multi_split_dir, True)

# img dataset 구성
print('==========================make multi test img dataset========================')
to_multi.mk_img_(img_root_dir, multi_rgb_dir)

# label dataset 구성
print('=====================make multi test label dataset======================')
to_multi.mk_label_(label_root_dir, multi_label_dir)


# singledataset의 test용  영상 frame 구성하기 ===============================================
single_split_dir = r'D:\singlelabel_test'
single_rgb_dir = r'D:\singlelabel_test\rgb-images'
single_label_dir = r'D:\singlelabel_test\labels'

try:
    os.mkdir(single_rgb_dir)
except:
    print('already exist')

try:
    os.mkdir(single_label_dir)
except:
    print('already exist')

# dataloader에서 참고하기 위한 txt파일 만들기
to_single.make_dataset(source_dir, src_img_dir, single_rgb_dir, single_label_dir)
to_single.mk_splitfiles(root_dir=single_rgb_dir, split_dir=single_split_dir, is_train=True)

to_single.rename_files_in_directory(single_rgb_dir)
to_single.check_in_sequence(single_rgb_dir)

to_single.rename_files_in_directory(single_label_dir)
to_single.check_in_sequence(single_label_dir, False)

# trafficdataset의 test용 영상 frame 구성하기 ===============================================
