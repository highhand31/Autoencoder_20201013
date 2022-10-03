
from Utility import ExtractSegDefect




if __name__ == "__main__":
    img_dir = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\defects_but_ok_20220922"

    # id2class_name = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\classnames.txt"
    id2class_name = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\classnames_one_defect_class.txt"
    a = ExtractSegDefect(id2class_name)
    a.defect_crop2png(img_dir,to_classify=True)