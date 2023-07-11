import cv2
import numpy as np
import random
import os

class GenerateSyntheticData:
    def __init__(self, label_gt_txt_path, dataset_path, rec_gt_txt_path, new_dataset_path, repeat):
        self.generateted_string_list = []
        self.name_cnt = 0
        self.repeat = repeat

        self.dataset_path = dataset_path

        ##Create new synthetic dataset folder and crop_img folder
        self.new_dataset_path = new_dataset_path
        os.makedirs(new_dataset_path, exist_ok=True)
        os.makedirs(new_dataset_path+"crop_img", exist_ok=True)

        ##Create synthetic Label.txt and rec_gt.txt
        self.synthetic_label_txt = open(new_dataset_path + "Label.txt", "a+")
        self.synthetic_rec_gt_txt = open(new_dataset_path + "rec_gt.txt", "a+")

        ##Label.txt
        self.label_gt_txt = open(label_gt_txt_path, "r")
        self.label_gt_txt_lines = self.label_gt_txt.readlines()

        ##rec_gt.txt
        self.rec_gt_txt = open(rec_gt_txt_path, "r")
        self.rec_gt_txt_lines = self.rec_gt_txt.readlines()

        self.order_list = []

    def __call__(self):
        self.parse_rec_gt_txt()
        self.synthetic_label_txt.close()
        self.synthetic_rec_gt_txt.close()


    def parse_rec_gt_txt(self):
        for rec_line in self.rec_gt_txt_lines:
            tab_idx = rec_line.find("\t")
            gt_img_path = rec_line[0:tab_idx]
            gt_img_name = gt_img_path.split("/")[-1]

            gt_crop_image = cv2.imread(self.dataset_path + gt_img_path)
            gt_string = rec_line[tab_idx+1:][:-1]
            
            self.order_list = []
            for i in range(len(gt_string)-1):
                self.order_list.append(i)

            h,w,d = gt_crop_image.shape

            step = int(np.ceil(w/len(gt_string)))
            parse_coord_list = []
            for idx, step_num in enumerate(range(0, w, step)):
                if idx == len(gt_string)-1:
                    parse_coord_list.append([step_num,0,w-1,h-1])
                else:
                    parse_coord_list.append([step_num,0,step_num+step,h-1])

                # gt_crop_image = cv2.rectangle(gt_crop_image, (step_num,0), (step_num+step,h-1), (255,0,0),1)
                # cv2.imshow("image", gt_crop_image)
                # cv2.waitKey(0)
            
            label_txt_line, main_image = self.parse_label_txt(gt_img_name)

            self.generateted_string_list = []
            for idx in range(self.repeat):
                synthetic_crop_img, synthetic_string = self.generate_new_image(parse_coord_list, gt_crop_image, main_image, gt_string)
                ##Dont save if the string already generated
                if synthetic_string in self.generateted_string_list:
                    continue
                else:
                    self.generateted_string_list.append(synthetic_string)

                self.auto_label(synthetic_crop_img, main_image, rec_line, label_txt_line, gt_string, synthetic_string, gt_img_name)

            #self.visualize_type1(parse_coord_list, gt_crop_image)

    def auto_label(self, synthetic_crop_img, main_image, rec_txt_line, label_txt_line, gt_string, synthetic_string, gt_img_name):
        ##Save the org image and synthetic crop image
        synthetic_image_name = "synthetic_v1_" + str(self.name_cnt) + "_" + synthetic_string
        
        cv2.imwrite(self.new_dataset_path+synthetic_image_name+".jpg", main_image)
        cv2.imwrite(self.new_dataset_path+"crop_img/"+synthetic_image_name+"_crop_0.jpg", synthetic_crop_img)

        ##Write Label.txt
        tab_idx = label_txt_line.find("\t")
        self.gt_img_path = label_txt_line[0:tab_idx]

        label_txt_line = label_txt_line.replace(gt_string, synthetic_string)
        label_txt_line = label_txt_line.replace(self.gt_img_path, new_dataset_path.split("/")[-2] + "/" + synthetic_image_name)
        self.synthetic_label_txt.writelines(label_txt_line)
        

        ##Write rec_gt.txt
        rec_txt_line = rec_txt_line.replace(gt_string, synthetic_string)
        rec_txt_line = rec_txt_line.replace(gt_img_name, synthetic_image_name + "_crop_0.jpg")
        self.synthetic_rec_gt_txt.writelines(rec_txt_line)

        print(synthetic_image_name)
        self.name_cnt+=1

    def generate_new_image(self, parse_coord_list, gt_crop_image, main_image, gt_string):
        shuffled_order_list = self.order_list.copy()
        synthetic_img = gt_crop_image.copy()

        flag=True
        while flag:
            random.shuffle(shuffled_order_list)
            for i in range(len(gt_string)-1):
                if shuffled_order_list[i] == i:
                    flag=True
                    break
                else:
                    flag=False

        for i in range(len(gt_string)-1):
            j = shuffled_order_list[i]
            synthetic_img[parse_coord_list[i][1]:parse_coord_list[i][3], parse_coord_list[i][0]:parse_coord_list[i][2]] = gt_crop_image[parse_coord_list[j][1]:parse_coord_list[j][3], parse_coord_list[j][0]:parse_coord_list[j][2]]

        synthetic_string = ""
        for idx in shuffled_order_list:
            synthetic_string += gt_string[idx]
        synthetic_string += gt_string[-1]

        # cv2.imshow("org_img", gt_crop_image)
        # cv2.imshow("synthetic_img", synthetic_img)
        # cv2.imshow("synthetic_img_blured_kernel_3", cv2.medianBlur(synthetic_img, 3))
        # cv2.imshow("synthetic_img_blured_kernel_5", cv2.medianBlur(synthetic_img, 5))
        # cv2.waitKey(0)
        return synthetic_img, synthetic_string
                
    def parse_label_txt(self, image_name):
        image_name = image_name[:-11]+".jpg"
        for gt_line in self.label_gt_txt_lines:
            ###Info in ground truth txt
            tab_idx = gt_line.find("\t")
            self.gt_img_path = gt_line[0:tab_idx]
            self.gt_img_name = self.gt_img_path.split("/")[-1]

            if image_name != self.gt_img_name:
                continue
            else:
                gt_image = cv2.imread(self.dataset_path + self.gt_img_name)
                return gt_line, gt_image

            #self.visualize_type0(gt_image, gt_string, gt_points)

    def visualize_type0(self, gt_image, gt_string, gt_points):
        bbox = np.array([gt_points],np.int32)
        tmp_bbox = bbox.reshape((-1, 1, 2))
        self.imagecv_out = cv2.polylines(gt_image, [tmp_bbox], True, (0,255,0), 1)
        cv2.imshow("image", self.imagecv_out)
        cv2.waitKey(0)
    
    def visualize_type1(self, parse_coord_list, gt_crop_image):
        for coords in parse_coord_list:
            gt_crop_image = cv2.rectangle(gt_crop_image, (coords[0], coords[1]), (coords[2], coords[3]), (255,0,0),1)

        print(len(parse_coord_list))
        cv2.imshow("image", gt_crop_image)
        cv2.waitKey(0)
        
if __name__ == "__main__":
    repeat = 100

    dataset_path = "stop_sign/"
    new_dataset_path = "result/"   # The path of the newly created dataset
    label_gt_txt_path = dataset_path + "/Label.txt"
    rec_gt_txt_path = dataset_path + "/rec_gt.txt"

    GSY = GenerateSyntheticData(label_gt_txt_path, dataset_path, rec_gt_txt_path, new_dataset_path, repeat)
    GSY()