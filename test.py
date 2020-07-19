import matplotlib.pyplot as plt
import cv2
from signature_extractor import TresholdSignatureExtractor, OtsuTresholdSignatureExtractor,\
                                AdaptiveMeanTresholdSignatureExtractor, AdaptiveGaussianTresholdSignatureExtractor,\
                                MorphologySignatureExtractor, MaskedSignatureExtractor, \
                                GlobalOtsuTresholdSignatureExtractor, TriangleTresholdSignatureExtractor, \
                                YenTresholdSignatureExtractor, BinaryLocalTresholdSignatureExtractor


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


def run_for_dataset(dataset_index=0):

    def plot_sig(_se, _ax, _sig, _title):
        _ax.imshow(_sig, cmap="gray")
        _ax.title.set_text(_title)
        ok, code, msg = _se.validate(_sig)
        if not ok:
            plt.setp(_ax.spines.values(), color="red")
            _ax.text(0, 10, msg, color="red")

    datasets = [[15, "jpeg"], [25, "jpg"], [4, "png"]]

    size, ext = datasets[dataset_index]

    for i in range(size+1):

        img = cv2.imread("./images/original/eg{0}.{1}".format(i, ext))

        assert img is not None, "Image cannot be none!"

        sig1 = tex.extract_and_resize(img=img)
        sig2 = otse.extract_and_resize(img=img)
        sig3 = amtex.extract_and_resize(img=img)
        sig4 = agtse.extract_and_resize(img=img)
        sig5 = mse.extract_and_resize(img=img)
        sig6 = maskse.extract_and_resize(img=img)
        sig7 = bltse.extract_and_resize(img=img)
        sig8 = gotse.extract_and_resize(img=img)
        sig9 = ttse.extract_and_resize(img=img)
        sig10 = ytse.extract_and_resize(img=img)

        cv2.imwrite("./images/out/eg_{0}_{1}_0.png".format(dataset_index, i), img)

        fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7), (ax8, ax9)) = plt.subplots(5, 2)

        plot_sig(tex, ax0, sig1, "Treshold")
        plot_sig(otse, ax1, sig2, "Otsu Treshold")
        plot_sig(amtex, ax2, sig3, "Adaptive Mean Treshold")
        plot_sig(agtse, ax3, sig4, "Adaptive Gaussian Treshold")
        plot_sig(mse, ax4, sig5, "Morphology")
        plot_sig(maskse, ax5, sig6, "Masked")
        plot_sig(bltse, ax6, sig7, "Binary Local Treshold")
        plot_sig(gotse, ax7, sig8, "Global Otsu Treshold")
        plot_sig(ttse, ax8, sig9, "Triangle Treshold")
        plot_sig(ytse, ax9, sig10, "Yen Treshold")

        #fig.set_size_inches((14.5, 7.3))
        fig.set_size_inches((14.5, 9))

        for ax in fig.axes:
            ax.set_yticks(())
            ax.set_xticks(())

        plt.savefig("./images/out/eg_{0}_{1}_1.png".format(dataset_index, i))
        plt.close(plt.gcf())


if __name__ == "__main__":
    run_for_dataset(0)
    run_for_dataset(1)
    run_for_dataset(2)
