import matplotlib.pyplot as plt
import cv2
from signature_extractor import TresholdSignatureExtractor, OtsuTresholdSignatureExtractor,\
                                AdaptiveMeanTresholdSignatureExtractor, AdaptiveGaussianTresholdSignatureExtractor,\
                                MorphologySignatureExtractor, FocusedSignatureExtractor


tex = TresholdSignatureExtractor()
otse = OtsuTresholdSignatureExtractor()
amtex = AdaptiveMeanTresholdSignatureExtractor()
agtse = AdaptiveGaussianTresholdSignatureExtractor()
mse = MorphologySignatureExtractor()
fse = FocusedSignatureExtractor()


def run_for_dataset(dataset_index=0):

    datasets = [[15, "jpeg"], [5, "jpg"], [4, "png"]]

    size, ext = datasets[dataset_index]

    for i in range(size+1):

        img = cv2.imread("./images/original/eg{0}.{1}".format(i, ext))

        sig1 = tex.extract_and_resize(img=img)
        sig2 = otse.extract_and_resize(img=img)
        sig3 = amtex.extract_and_resize(img=img)
        sig4 = agtse.extract_and_resize(img=img)
        sig5 = mse.extract_and_resize(img=img)
        sig6 = fse.extract_and_resize(img=img)

        cv2.imwrite("./images/out/eg_{0}_{1}_0.png".format(dataset_index, i), img)

        fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2)

        ax0.imshow(sig1, cmap="gray")
        ax0.title.set_text('Treshold')

        ax1.imshow(sig2, cmap="gray")
        ax1.title.set_text('Otsu Treshold')

        ax2.imshow(sig3, cmap="gray")
        ax2.title.set_text('Adaptive Mean Treshold')

        ax3.imshow(sig4, cmap="gray")
        ax3.title.set_text('Adaptive Gaussian Treshold')

        ax4.imshow(sig5, cmap="gray")
        ax4.title.set_text('Morphology')

        ax5.imshow(sig6, cmap="gray")
        ax5.title.set_text('Focused AGT')

        fig.set_size_inches((14.5, 7.3))

        for ax in fig.axes:
            ax.set_yticks(())
            ax.set_xticks(())

        plt.savefig("./images/out/eg_{0}_{1}_1.png".format(dataset_index, i))
        plt.close(plt.gcf())


if __name__ == "__main__":
    run_for_dataset(0)
    run_for_dataset(1)
    run_for_dataset(2)


