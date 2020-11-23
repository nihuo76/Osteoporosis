import numpy as np
import os
from PIL import Image
import pandas as pd

def getimagexywh(personname):
    Osteo_dataset_folder = os.path.join(os.getcwd(), "Osteoporosis_Dataset")
    image_folder = os.path.join(Osteo_dataset_folder, "dataset")
    coordinate_8ROI = np.zeros((8, 2))
    person_folder = os.path.join(image_folder, personname)
    all_file_of_person = os.listdir(person_folder)
    for i in range(len(all_file_of_person)):
        if all_file_of_person[i][-4:] == ".txt":
            coordinate_file = open(os.path.join(person_folder, all_file_of_person[i]))
        else:
            DPR_image_loc = os.path.join(person_folder, all_file_of_person[i])
    DPR_image = Image.open(DPR_image_loc)
    lines_read = coordinate_file.readlines()
    for j in range(len(lines_read)):
        if lines_read[j][:8] == "Landmark":
            coordinate_8ROI[int(lines_read[j][9])] = lines_read[j][12:-1].split(',')
        if lines_read[j][0] in {"0", "1", "2", "3", "4", "5", "6", "7"}:
            coordinate_8ROI[int(lines_read[j][0])] = lines_read[j][2:-1].split(',')
    # person2coordinate[personname] = coordinate_8ROI
    return DPR_image, coordinate_8ROI
