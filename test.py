import os
import matplotlib.pyplot as plt
import cv2
from exception import SignatureException
from signature_extractor import *


tex = TresholdSignatureExtractor()
otse = OtsuTresholdSignatureExtractor()
amtex = AdaptiveMeanTresholdSignatureExtractor()
agtse = AdaptiveGaussianTresholdSignatureExtractor()
mse = MorphologySignatureExtractor()
bltse = BinaryLocalTresholdSignatureExtractor()
gotse = GlobalOtsuTresholdSignatureExtractor()
ttse = TriangleTresholdSignatureExtractor()
ytse = YenTresholdSignatureExtractor()
maskse = MaskedSignatureExtractor()


class Signatures:

    def __init__(self, dataset_folder="./images/original/"):
        self.images = os.listdir(dataset_folder)
        self.images.sort()
        self.n = 0

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.images):
            result = self.images[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration


def plot_sig(_se, _ax, _img, _title):

    try:
        _se.pre_validate(_img)
    except SignatureException as e:
        plt.setp(_ax.spines.values(), color="red")
        _ax.text(0, 10, e.error_code, color="purple") # e.message

    _sig = _se.extract_and_resize(_img)

    _sig = _se.prettify(_sig)

    _ax.imshow(_sig, cmap="gray")
    _ax.title.set_text(_title)

    try:
        _se.validate(_sig)
    except SignatureException as e:
        plt.setp(_ax.spines.values(), color="red")
        _ax.text(0, 25, e.error_code, color="red") # e.message


def run_for_all():

    for i, fn in enumerate(Signatures()):

        img = cv2.imread("./images/original/{0}".format(fn), cv2.IMREAD_UNCHANGED)

        assert img is not None, "Image cannot be none!"

        cv2.imwrite("./images/out/{0}_0.png".format(i), img)

        fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7), (ax8, ax9)) = plt.subplots(5, 2)

        plot_sig(tex, ax0, img, "Treshold")
        plot_sig(otse, ax1, img, "Otsu Treshold")
        plot_sig(amtex, ax2, img, "Adaptive Mean Treshold")
        plot_sig(agtse, ax3, img, "Adaptive Gaussian Treshold")
        plot_sig(mse, ax4, img, "Morphology")
        plot_sig(maskse, ax5, img, "Masked")
        plot_sig(bltse, ax6, img, "Binary Local Treshold")
        plot_sig(gotse, ax7, img, "Global Otsu Treshold")
        plot_sig(ttse, ax8, img, "Triangle Treshold")
        plot_sig(ytse, ax9, img, "Yen Treshold")

        #fig.set_size_inches((14.5, 7.3))
        fig.set_size_inches((14.5, 9))

        for ax in fig.axes:
            ax.set_yticks(())
            ax.set_xticks(())

        plt.savefig("./images/out/{0}_1.png".format(i))
        plt.close(plt.gcf())


def run_for_file(src="./images/nt.png"):

    #src = "./images/nt.png"
    ext = src.split(".")[-1]

    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    agtse.pre_validate(img)

    sig = agtse.extract_and_resize(img, size=(500, 50))
    sig = agtse.prettify(sig)

    cv2.imwrite(src.replace("."+ext, "_1.png"), sig)


def debug(se_class, dataset_folder="./images/original/"):

    verbose_folder = "./__verbose__/final/"
    os.makedirs(verbose_folder, exist_ok=True)

    for i, fn in enumerate(Signatures(dataset_folder)):
        se = se_class(verbose=2, img_name=str(i))

        src = dataset_folder + fn
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

        try:
            se.pre_validate(img)
        except SignatureException as e:
            print(i, e.error_code)

        sig = se.extract_and_resize(img, size=(500, 50))
        sig = se.prettify(sig)

        try:
            se.validate(sig)
        except SignatureException as e:
            print(i, e.error_code)

        se._verbose(sig, "final")
        cv2.imwrite(verbose_folder + str(i) + "_0.png", img)
        cv2.imwrite(verbose_folder + str(i) + "_1.png", sig)


if __name__ == "__main__":

    #run_for_all()
    #debug(AdaptiveGaussianTresholdSignatureExtractor)
    import time
    start_time = time.time()
    debug(AdaptiveGaussianTresholdSignatureExtractor)
    print("--- %s seconds ---" % (time.time() - start_time))
    pass
