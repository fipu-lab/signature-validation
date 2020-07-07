# SignatureExtractor
**A service for extracting and validating handwritten signatures from photos**

#### Service supports uploading a file on the route `extract-signature` with following options:
1. Uploading a file through a form (param name **_img_**)
2. Uploading a file in base64 encoding (param name **_img64_**)
3. Uploading a file encoded as bytes (param name **_img_bytes_**)
4. Specifiying output encoding (param name **_resp_enc_**)





## There are a couple of methods for extracting signature (check `test.py` for more info)

This service uses `FocusedSignatureExtractor` as it performs the best

```python
import cv2
from signature_extractor import FocusedSignatureExtractor

img = cv2.imread("./images/original/eg0.jpeg")

se = FocusedSignatureExtractor()
sig = tex.extract_and_resize(img=img)

cv2.imshow("extracted_signature", sig)
cv2.waitKey(1)
```



## Examples

**Original image**
![](images/out/eg_0_0_0.png)


**Extracted signatures by different methods**
![](images/out/eg_0_0_1.png)
