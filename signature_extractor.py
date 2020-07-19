from skimage.filters import threshold_triangle, threshold_yen, threshold_otsu, threshold_local
from skimage.util import img_as_ubyte
import numpy as np
import cv2


class SignatureExtractor:

    def extract(self, img):
        return self._extract(img)

    def extract_and_resize(self, img, size=(500, 100)):
        sig = self.extract(img)
        return self.resize(sig, size)

    def _extract(self, img):
        pass

    def add_blur(self, img):

        #cv2.bilateralFilter(img, 9, 90, 16)
        #img = cv2.GaussianBlur(img, (5, 5), 0)

        cv2.bilateralFilter(img, -1, 150, 16)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        return img

    def find_roi(self, sig, padding=20):

        bitmap = cv2.bitwise_not(sig) // 255

        l, r, t, b = -1, -1, -1, -1
        treshold = 2

        for i in range(bitmap.shape[0]):

            if sum(bitmap[i]) > treshold:
                t = i
                break

        for i in reversed(range(bitmap.shape[0])):

            if sum(bitmap[i]) > treshold:
                b = i
                break

        for i in range(bitmap.shape[1]):

            if sum(bitmap[:, i]) > treshold:
                l = i
                break

        for i in reversed(range(bitmap.shape[1])):

            if sum(bitmap[:, i]) > treshold:
                r = i
                break

        def nvl(val):
            return val if val >= 0 else 0

        return nvl(t-padding), nvl(b+padding), nvl(l-padding), nvl(r+padding)

    def resize(self, sig, size=(500, 100)):

        t, b, l, r = self.find_roi(sig)
        content = sig[t:b, l:r]

        # TODO: to 1x5cm
        resized = SignatureExtractor.resize_and_keep_ratio(content, size)

        return resized

    def find_content_with_canny(self, img):
        canny = cv2.Canny(img, 0, 255)
        t, b, l, r = self.find_roi(cv2.bitwise_not(canny))
        content = img[t:b, l:r]
        return content

    def validate(self, sig):
        """
        Signature passes validation if:
            - percentage of darker pixels is in between 0.5% and 10%
            - TODO: check if signature is well rotated
            - TODO: check if the pixels are scattered to much for a good signature

        :param sig: processed image
        :return: ok - False if validaton fails, otherwise True. err_code - error code. error_msg - error message
        """

        msg_dict = {0: "Ok",
                    1: "Signature contains under the limmit amount of dark pixels",
                    2: "Signature contains over the limmit amount of dark pixels"}
        ok = True
        error_code = 0

        top_limit = sig.shape[0] * sig.shape[1] * 0.1
        bottom_limit = sig.shape[0] * sig.shape[1] * 0.005

        #unique, counts = np.unique(sig, return_counts=True)
        #n_dark = dict(zip(unique, counts)).get(0, 0)
        n_dark = len(np.where(sig < 20)[0])

        if n_dark < bottom_limit:
            ok, error_code = False, 1
        elif n_dark > top_limit:
            ok, error_code = False, 2

        return ok, error_code, msg_dict[error_code]

    @staticmethod
    def resize_and_keep_ratio(img, size):
        #print("resize_and_keep_ratio:", img.shape, size)

        h, w = img.shape
        sh, sw = size
        ratio = sh / sw

        if h / w < ratio:
            padding = int((h * ratio - w) / 2)

            if padding < 0:
                padding = 0

            pad = np.repeat(255., padding*h).reshape((h, padding))

            img = np.concatenate((pad, img, pad), axis=1)

        elif h / w > ratio:
            padding = int((h - w * ratio) / 2)

            pad = np.repeat(255., padding*w).reshape((padding, w))

            img = np.concatenate((pad, img, pad), axis=0)

        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


class TresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #th, im_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        #return cv2.bitwise_not(im_th)

        img = self.find_content_with_canny(img)
        img = self.add_blur(img)

        th, im_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return im_th


class OtsuTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.find_content_with_canny(img)
        img = self.add_blur(img)

        th, im_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return im_th


class AdaptiveMeanTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.find_content_with_canny(img)
        img = self.add_blur(img)
        out_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 10)

        """
        import matplotlib.pyplot as plt
        binImg = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 10)
        plt.imshow(binImg, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.gcf().axes[0].set_axis_off()
        filename = "./{}.png".format(str(time.time()).replace(".", "_"))
        plt.savefig(filename)
        img = cv2.imread(filename)
        os.remove(filename)
        """

        return out_img


class AdaptiveGaussianTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.find_content_with_canny(img)
        img = self.add_blur(img)

        out_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 10)
        #out_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        #cv2.imshow("a", out_img)
        #cv2.waitKey(1)

        return out_img


class MorphologySignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        canny = cv2.Canny(img, 100, 255)
        se = np.ones((5, 5), dtype='uint8')
        image_close = cv2.morphologyEx(canny, cv2.MORPH_CROSS, se)

        image_close = cv2.subtract(255, image_close)

        return image_close


class MaskedSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        canny = cv2.Canny(img, 100, 255)
        se = np.ones((5, 5), dtype='uint8')
        image_close = cv2.morphologyEx(canny, cv2.MORPH_CROSS, se)

        image_close = cv2.subtract(255, image_close)

        avg_val = np.average(img.flatten())

        if avg_val > 127:
            avg_val = 127

        masked_img = np.array(image_close)
        for r in range(masked_img.shape[0]):
            for c in range(masked_img.shape[1]):
                if masked_img[r, c] == 0:
                    masked_img[r, c] = img[r, c]
                else:
                    masked_img[r, c] = avg_val

        #masked_img = self.add_blur(masked_img)
        #out_img = cv2.adaptiveThreshold(masked_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 10)
        _, out_img = cv2.threshold(masked_img, avg_val+10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return out_img

"""
class FocusedSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #canny = cv2.Canny(img, 0, 255)
        #t, b, l, r = self.find_roi(cv2.bitwise_not(canny))
        #content = img[t:b, l:r]
        content = self.find_content_with_canny(img)

        content = self.add_blur(content)
        out_img = cv2.adaptiveThreshold(content, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 10)

        return out_img
"""

class GlobalOtsuTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        content = self.find_content_with_canny(img)
        content = self.add_blur(content)

        img = img_as_ubyte(content)
        threshold_global_otsu = threshold_otsu(img)
        global_otsu = img >= threshold_global_otsu

        out_img = np.zeros(global_otsu.shape).astype(np.uint8)
        out_img[global_otsu] = 255

        return out_img


class TriangleTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        content = self.find_content_with_canny(img)
        content = self.add_blur(content)

        treshold = img > threshold_triangle(content)

        out_img = np.zeros(treshold.shape).astype(np.uint8)
        out_img[treshold] = 255

        return out_img


class YenTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        content = self.find_content_with_canny(img)
        content = self.add_blur(content)

        treshold = img > threshold_yen(content)

        out_img = np.zeros(treshold.shape).astype(np.uint8)
        out_img[treshold] = 255

        return out_img


class BinaryLocalTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        content = self.find_content_with_canny(img)
        content = self.add_blur(content)

        local_thresh = threshold_local(content, block_size=35, offset=10)
        binary_local = content > local_thresh

        out_img = np.zeros(binary_local.shape).astype(np.uint8)
        out_img[binary_local] = 255

        return out_img
