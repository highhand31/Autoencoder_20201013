from tools import embedding_comparison,get_ave_embed_distance,classification_by_embed_comparison,show_embed_results



if __name__ == "__main__":
    #----calculate the average embedding distance
    img_source = [r"F:\dataset\optoTech\CMIT\014_2_dataset\vali_L1\L1_OK"]
    # OK_list = [
    #     r"F:\dataset\optoTech\CMIT\014_dataset\training\L1_OK",
    #     # r"D:\dataset\optotech\014\manual_selection\AI_selection_1\OK",
    #     # r"D:\dataset\optotech\014\manual_selection\AI_selection_2\OK",
    #     # r"D:\dataset\optotech\014\manual_selection\AI_selection_3\OK",
    # ]
    # NG_list = [
    #                 r"F:\dataset\optoTech\CMIT\014_dataset\training\L1_NG",
    #                 # r"D:\dataset\optotech\014\manual_selection\AI_selection_1\NG",
    #                 # r"D:\dataset\optotech\014\manual_selection\AI_selection_2\NG",
    #                 # r"D:\dataset\optotech\014\manual_selection\AI_selection_3\NG",
    #               # r"D:\dataset\optotech\014\manual_selection\OK"
    # ]
    # img_source.extend(a_list)
    # databse_source = img_source
    databse_source = [r"F:\dataset\optoTech\CMIT\014_2_dataset\train_L1\L1_NG"]
    pb_path = r"D:\code\model_saver\Opto_tech\CMIT_009_45classes_noPool\infer_acc(90.4).nst"
    compare_type = 'under'
    content, _ = embedding_comparison(img_source, databse_source, pb_path,
                                      GPU_ratio=None, compare_type=compare_type)
    ave_dis,std_dis = get_ave_embed_distance(content)
    print("ave_distance: {}, ave_std:{}".format(ave_dis,std_dis))

    #----show embedding comparison results
    # show_embed_results(content,show_qty=3)

    #----classification by embedding comparison
    img_dir = r"F:\dataset\optoTech\CMIT\014_dataset\validation\L1_NG"
    # img_dir = r"D:\dataset\optotech\014\test_img\t7"
    # databse_dir = img_source
    # databse_dir = r"D:\dataset\optotech\014\manual_selection\NG"
    #pb_path = r"D:/code/model_saver/Opto_tech/CLS_014_AI_selection/inference_2021426181728.nst"
    compare_type = 'under'
    # content,_ = embedding_comparison(img_dir, databse_source, pb_path,
    #                                   GPU_ratio=None,compare_type=compare_type)

    # ----show embedding comparison results
    # show_embed_results(content,show_qty=3)

    save_dict = {'under': True, 'over': False}
    # classification_by_embed_comparison(content,img_dir,save_dict=save_dict,threshold=ave_dis)