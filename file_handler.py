import base64
import cv2
import numpy as np

from exception import SignatureException
from signature_extractor import FocusedSignatureExtractor

JPEG_QUALITY = 80
ENCODING_BASE64 = 'base64'
ENCODING_BYTES = 'bytes'


def cv2_read_img(stream):
    return cv2.imdecode(np.frombuffer(stream, np.uint8), cv2.IMREAD_UNCHANGED)


def get_from_file(file, encoding=ENCODING_BASE64):
    return extract_signature(cv2_read_img(file.read()), encoding)


def get_from_base64(uri, encoding=ENCODING_BASE64):
    encoded_data = uri.split('base64,')[-1]
    try:
        img = cv2_read_img(base64.b64decode(encoded_data))
    except:
        raise SignatureException("Invalid base64 encoded image")

    return extract_signature(img, encoding)


def get_from_bytes(img_bytes, encoding=ENCODING_BASE64):
    try:
        return extract_signature(img_bytes, encoding)
    except:
        raise SignatureException("Invalid image bytes")


def extract_signature(img, encoding=ENCODING_BASE64):
    cv2.imwrite('./images/test0.jpg', img)

    se = FocusedSignatureExtractor()
    img = se.extract_and_resize(img)

    cv2.imwrite('./images/test1.jpg', img)
    return convert_img(img, encoding)


def convert_img(img, method):
    if img is None:
        return None
    success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    if not success:
        return None

    if method == ENCODING_BASE64:
        return base64.b64encode(buffer).decode('utf-8')
    elif method == ENCODING_BYTES:
        return buffer.tobytes()

    raise SignatureException("Encoding method {} not implemented. Use {}".format(method, (ENCODING_BASE64, ENCODING_BYTES)))