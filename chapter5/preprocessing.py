import tensorflow as tf
def flip(x):
    x = tf.image.random_flip_left_right(x)
    return x


def color(x):
    x = tf.image.random_hue(x, 0.08)  # 隨機調整影像色調
    x = tf.image.random_saturation(x, 0, 6, 1.6)  # 隨機影像飽和度
    x = tf.image.random_brightness(x, 0.05)  # 隨機影像亮度
    x = tf.image.random_contrast(x, 0.7, 1.3)  # 隨機影像對比度
    return x


def rotate(x):
    # 隨機旋轉n次
    x = tf.image.rot90(x, tf.random.uniform(
        shape=[], minval=1, maxval=4, dtype=tf.int32))
    return x


def zoom(x, scale_min=0.6, scale_max=1.4):
    h, w, c = x.shape
    scale = tf.random.uniform([], scale_min, scale_max)
    sh = h*scale
    sw = w*scale
    x = tf.image.resize(x, (sh, sw))  # 縮放
    x = tf.image.resize_with_crop_or_pad(x, h, w)  # 縮放後的填補
    return x

def parse_aug_fn(dataset):
        # 像素縮小255倍, 變成0-1之間
    x = tf.cast(dataset['image'], tf.float32) / 255
    # 觸發水平翻轉
    x = flip(x)
    # 50%觸發顏色轉換
    x = tf.cond(tf.random.uniform(shape=[], 0, 1) > 0.5, lambda: color(x), lambda: x)
    # 25%觸發旋轉
    x = tf.cond(tf.random.uniform(shape=[], 0, 1) > 0.75, lambda: rotate(x), lambda: x)
    # 50%觸發縮放
    x = tf.cond(tf.random.uniform(shape=[], 0, 1) > 0.5, lambda: zoom(x), lambda: x)
    y = tf.one_hot(dataset['label'], 10)
    return x, y


def parse_fn(dataset):
    # 像素縮小255倍, 變成0-1之間
    x = tf.cast(dataset['image'], tf.float32) / 255
    y = tf.one_hot(dataset['label'], 10)
    return x, y
