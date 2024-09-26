# 기준이 되는 해상도 이외의 데이터는 제외하기 위함
import os
import json
from glob import glob

class dataset_filter():
    def __init__(self, json_path, resolution = (1280, 720), is_base = True):
        self.json_path = json_path
        self.resolution = resolution
        self.is_base = is_base

    def __count__(self):

        if self.is_base:

            count = 0

            # d1 ex) 정상 
            for d1 in os.listdir(self.json_path):
                json_d1 = os.path.join(self.json_path, d1)
                
                #d2 ex) 01 방향지시등 불이행 etc
                for d2 in os.listdir(json_d1):
                    json_d2 = os.path.join(json_d1, d2)
                    
                    # d3  ex) 'p01_22021103_072002_an1_036_03' 
                    for d3 in os.listdir(json_d2):
                        json_d3 = os.path.join(json_d2, d3)

                        #json_file ex) ~~.json
                        for json_file in os.listdir(json_d3):
                            mismatch_check = False
                            jdata_path = os.path.join(json_d3, json_file)

                            with open(jdata_path, 'rb') as jfile:
                                jdata = json.load(jfile)
                                w = jdata["imageInfo"]['width']
                                h = jdata["imageInfo"]['height']
                                wh = (w, h)
                                if self.resolution != wh:
                                    count += 1
                                    mismatch_check = True
                        
                        if mismatch_check:
                            print(d3)
                            continue
        else:
            self.json_path = 


        print(f'설정 해상도 이외 데이터 영상 개수: ', {count})

    
    def filtering(self):

        # d1 ex) 정상 
        for d1 in os.listdir(self.json_path):
            json_d1 = os.path.join(self.json_path, d1)
            
            #d2 ex) 01 방향지시등 불이행 etc
            for d2 in os.listdir(json_d1):
                json_d2 = os.path.join(json_d1, d2)
                
                # d3  ex) 'p01_22021103_072002_an1_036_03' 
                for d3 in os.listdir(json_d2):
                    json_d3 = os.path.join(json_d2, d3)

                    #json_file ex) ~~.json
                    for json_file in os.listdir(json_d3):
                        jdata_path = os.path.join(json_d3, json_file)

                        with open(jdata_path, 'rb') as jfile:
                            jdata = json.load(jfile)
                            w = jdata["imageInfo"]['width']
                            h = jdata["imageInfo"]['height']
                            wh = (w, h)
                            if self.resolution != wh:



                    
                    
















        



         
