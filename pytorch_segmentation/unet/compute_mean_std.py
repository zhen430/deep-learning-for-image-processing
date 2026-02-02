import os
from PIL import Image
import numpy as np


def main():
    img_channels = 3
    img_dir = "./DRIVE/training/images"
    roi_dir = "./DRIVE/training/mask"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".tif")]
    cumulative_mean = np.zeros(img_channels)  #对每个通道计算平均值和方差
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".tif", "_mask.gif"))
        img = np.array(Image.open(img_path)) / 255.    #训练集每个图片，像素归一化
        roi_img = np.array(Image.open(ori_path).convert('L'))  #转为灰度图（3个通道加权平均，权重按照人眼感光原理的经验值来定）

        img = img[roi_img == 255]  #从训练集图像img中提取mask为白色（血管）的像素值，组成一个数组
        #img是多通道，这里的数组也是多通道的【 ，C】。下一句对每个通道单独求。
        cumulative_mean += img.mean(axis=0)  #【C】
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()
