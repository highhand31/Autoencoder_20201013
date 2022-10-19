
from AE_Seg_Util import ExtractSegDefect




if __name__ == "__main__":
    # id2class_name = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\classnames.txt"
    id2class_name = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\classnames_with_unknown.txt"
    a = ExtractSegDefect(id2class_name)

    #----extract defects
    # img_dir = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\defect_parts\小紅點"
    # a.defect_crop2png(img_dir,to_classify=False)

    #defect analysis
    # defect_dir = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train\defectCrop2png\gold_particle"
    defect_dir = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\defect_parts\小紅點\defectCrop2png\test"
    a.defect_analysis(defect_dir)

    #----sort by area
    # sort_dir = r"D:\dataset\optotech\silicon_division\PDAP\破洞_金顆粒_particle\train\defectCrop2png\hole"
    # a.sort_by_area(sort_dir,area_threshold=25,sort_type='under',img_ext='png')

    #----sort by HW ratio
    # a.sort_by_HW_ratio(sort_dir,HW_ratio_threshold=40,sort_type='over',img_ext='png')
