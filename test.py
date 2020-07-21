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


def run_for_dataset(dataset_index=0):

    datasets = [[15, "jpeg"], [35, "jpg"], [10, "png"]]

    size, ext = datasets[dataset_index]

    for i in range(size+1):

        img = cv2.imread("./images/original/eg{0}.{1}".format(i, ext), cv2.IMREAD_UNCHANGED)

        assert img is not None, "Image cannot be none!"

        cv2.imwrite("./images/out/eg_{0}_{1}_0.png".format(dataset_index, i), img)

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

        plt.savefig("./images/out/eg_{0}_{1}_1.png".format(dataset_index, i))
        plt.close(plt.gcf())


def run_all():
    run_for_dataset(0)
    run_for_dataset(1)
    run_for_dataset(2)


def run_for_file(src="./images/nt.png"):

    #src = "./images/nt.png"
    ext = src.split(".")[-1]

    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    sig = tex.extract_and_resize(img)
    sig = tex.prettify(sig)
    cv2.imwrite(src.replace("."+ext, "_1."+ext), sig)


if __name__ == "__main__":

    run_all()
    pass
