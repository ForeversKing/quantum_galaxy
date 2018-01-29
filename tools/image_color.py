import cv2
from multiprocessing import Pool



data_write_path = "home/yulongwu/d/voice/voice_image/"
origin_image_path = "/home/yulongwu/d/voice/voice_image/total_image.lst"


def gay_color(path):
    im_gray = cv2.imread(path)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
    cv2.imwrite("%s.jpg" % path, im_color)


if __name__ == "__main__":
    total_lst = []
    data_input = open(origin_image_path)
    for i in data_input:
        total_lst.append(i.strip())
    pool = Pool(8)
    rl = pool.map(gay_color, total_lst)
    pool.close()
    pool.join()
    data_input.close()
