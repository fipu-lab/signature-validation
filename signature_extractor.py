from skimage.filters import threshold_triangle, threshold_yen, threshold_otsu, threshold_local
from skimage.util import img_as_ubyte
import numpy as np
import cv2
import os
from exception import SignatureException


class SignatureExtractor:

    def __init__(self, lang="cro", verbose=0, img_name=None):
        assert lang in ["cro", "eng"]
        self.lang = lang
        self.verbose = verbose
        self.img_name = img_name

        assert verbose > 0 and img_name is not None or verbose == 0, "Image name must be set when using verbose mode"

        if verbose > 0:
            self._verbose_counter = 0
            self._verbose_folder = "./__verbose__/" + img_name + "/"
            os.makedirs(self._verbose_folder, exist_ok=True)

    def _verbose(self, img, operation, info=""):
        out = img.copy()

        if self.verbose == 2:
            cv2.putText(out, info, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imwrite(self._verbose_folder + str(self._verbose_counter) + "_" + operation + ".png", out)
            self._verbose_counter += 1

    def _prepare_img(self, img):

        if img.shape[2] > 3: # 4 channels - deal with transparency

            # Crop-out as much as possible of the transparent part of the image
            y, x = img[:, :, 3].nonzero() # get the nonzero alpha coordinates
            minx = np.min(x)
            miny = np.min(y)
            maxx = np.max(x)
            maxy = np.max(y)
            img = img[miny:maxy, minx:maxx]

            # convert the rest of transparent pixels to white
            trans_mask = img[:, :, 3] == 0
            img[trans_mask] = [255, 255, 255, 255]
            img = img[:, :, :3]

        self._verbose(img, "prepared_image")

        return img

    def extract(self, img):
        self._verbose(img, "original")

        img = self._prepare_img(img)
        return self._extract(img)

    def run(self, img, size=(500, 100)):

        self.pre_validate(img)

        sig = self.extract(img)

        sig = self.prettify(sig)

        self.validate(sig)

        return self.resize(sig, size)

    def _extract(self, img):
        pass

    def add_blur(self, img):

        #cv2.bilateralFilter(img, 9, 90, 16)
        #img = cv2.GaussianBlur(img, (5, 5), 0)

        # img = cv2.bilateralFilter(img, -1, 150, 16)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        self._verbose(img, "blurred")

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

        size = (size[0], size[1]) # ensure tuple

        t, b, l, r = self.find_roi(sig)
        content = sig[t:b, l:r]

        resized = SignatureExtractor.resize_and_keep_ratio(content, size)

        self._verbose(resized, "resized")
        return resized

    def find_content(self, image):
        img = image.copy()

        """
        # Zero-parameter, automatic canny
        sigma = .33
        v = np.median(img)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        canny = cv2.Canny(img, lower, upper)
        """

        #canny = cv2.Canny(img, 0, 255)
        #canny = cv2.Canny(img, 0, 100)
        #canny = cv2.Canny(img, 40, 120)
        #canny = cv2.Canny(img, 100, 255)
        #t, b, l, r = self.find_roi(cv2.bitwise_not(canny))
        #self._verbose(canny[t:b, l:r], "find_content")

        img = cv2.GaussianBlur(img, (7, 7), 0)
        kernel = np.ones((7, 7), np.uint8)

        img = cv2.fastNlMeansDenoising(img.astype(np.uint8), dst=None, h=10, templateWindowSize=7, searchWindowSize=21)
        opening = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

        tresh = cv2.adaptiveThreshold(cv2.bitwise_not(opening), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 10)
        canny = cv2.Canny(tresh, 50, 255)

        t, b, l, r = self.find_roi(cv2.bitwise_not(canny))
        self._verbose(canny[t:b, l:r], "find_content")

        content = image[t:b, l:r]

        return content

    def pre_validate(self, img):
        """
        Image passes validation if:
            - is not blurry
        :param img:
        """
        # English messages
        msg_dict = {
            "ok": "Ok",
            "image_blurry": "Image is blurry"
        }

        if self.lang == "cro":
            # Croatian images
            msg_dict = {
                "ok": "Ok",
                "image_blurry": "Slika potpisa je mutna, izoštri fokus prije slikanja i drži mobitel/kameru mirno prije slikanja potpisa"
            }

        error_code = "ok"

        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val = cv2.Laplacian(img_bw, cv2.CV_64F).var()

        self._verbose(img_bw, "blur_validation", "Laplacian var: {0}".format(val))

        if val < 1:
            error_code = "image_blurry"

        if error_code != "ok":
            raise SignatureException(message=msg_dict[error_code], error_code=error_code)

    def validate(self, sig):
        """
        Signature passes validation if:
            - percentage of darker pixels is in between 0.1% and 20%
            - TODO: check if signature is well rotated
            - TODO: check if the pixels are scattered to much for a good signature

        :param sig: processed image
        :return:
        """
        # English messages
        msg_dict = {
            "ok": "Ok",
            "sig_not_visible": "Signature is not clearly visible",
            "sig_overemphasized": "Signature is overemphasized or an error occurred in image processing"
        }

        if self.lang == "cro":
            # Croatian messages
            msg_dict = {
                "ok": "Ok",
                "sig_not_visible": "Potpis nije dovoljno jasno vidljiv, slikaj potpis na bijelom papiru sa običnom plavom kemijskom iz prikladne blizine da je potpis jasno vidljiv",
                "sig_overemphasized": "Potpis je prenaglašen ili je došlo do pogreške prilikom obrade slike, slikaj potpis na bijelom papiru sa običnom plavom kemijskom iz prikladne blizine da je potpis jasno vidljiv"
            }

        error_code = "ok"

        sig_flat = sig.flatten()
        top_limit = len(sig_flat) * 0.20
        bottom_limit = len(sig_flat) * 0.001

        n_dark = len(np.where(sig_flat < 200)[0])

        if n_dark < bottom_limit: # Signature contains under the limmit amount of dark pixels
            error_code = "sig_not_visible"
        elif n_dark > top_limit: # Signature contains over the limmit amount of dark pixels
            error_code = "sig_overemphasized"

        if error_code != "ok":
            raise SignatureException(message=msg_dict[error_code], error_code=error_code, img=sig)

    def prettify(self, sig):
        """
        Post proccessing on signature:
            - remove noise
            - ...
        """
        sig = cv2.fastNlMeansDenoising(sig.astype(np.uint8), dst=None, h=10, templateWindowSize=7, searchWindowSize=21)
        return sig

    @staticmethod
    def resize_and_keep_ratio(img, size):
        #print("resize_and_keep_ratio:", img.shape, size)

        h, w = img.shape
        _w, _h = size
        ratio = _w / _h

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

        img = self.find_content(img)
        img = self.add_blur(img)

        th, im_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return im_th


class OtsuTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.find_content(img)
        img = self.add_blur(img)

        th, im_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return im_th


class AdaptiveMeanTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.find_content(img)
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

        img = self.find_content(img)
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
        content = self.find_content(img)

        content = self.add_blur(content)
        out_img = cv2.adaptiveThreshold(content, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 10)

        return out_img
"""

class GlobalOtsuTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        content = self.find_content(img)
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

        content = self.find_content(img)
        content = self.add_blur(content)

        treshold = img > threshold_triangle(content)

        out_img = np.zeros(treshold.shape).astype(np.uint8)
        out_img[treshold] = 255

        return out_img


class YenTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        content = self.find_content(img)
        content = self.add_blur(content)

        treshold = img > threshold_yen(content)

        out_img = np.zeros(treshold.shape).astype(np.uint8)
        out_img[treshold] = 255

        return out_img


class BinaryLocalTresholdSignatureExtractor(SignatureExtractor):

    def _extract(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        content = self.find_content(img)
        content = self.add_blur(content)

        local_thresh = threshold_local(content, block_size=35, offset=10)
        binary_local = content > local_thresh

        out_img = np.zeros(binary_local.shape).astype(np.uint8)
        out_img[binary_local] = 255

        return out_img
