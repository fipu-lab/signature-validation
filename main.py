from flask import Flask, request, jsonify
from exception import SignatureException
import file_handler as fh
import os
from dotenv import load_dotenv
import bugsnag
from bugsnag.flask import handle_exceptions


app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max image size

IMG_FILE = 'img'
IMG_FILE64 = 'img64'
IMG_FILE_BYTES = 'img_bytes'
IMG_RESPONSE_ENCODING = 'resp_enc'
IMG_SIZES = 'img_sizes'
ALLOWED_KEYS = (IMG_FILE, IMG_FILE64, IMG_FILE_BYTES)
ALLOWED_EXTENSIONS = ('png', 'jpg', 'jpeg')

load_dotenv()
STAGE = os.environ.get("STAGE", os.environ.get("FLASK_ENV", "development"))
if STAGE != "development" and "BUGSNAG_API_KEY" in os.environ and os.environ.get("BUGSNAG_API_KEY"):
    bugsnag.configure(
        api_key=os.environ.get("BUGSNAG_API_KEY"),
        project_root="",
        app_version="0.0.1",
        release_stage=STAGE,
    )
    handle_exceptions(app)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_value(k):
    if request.form is not None and k in request.form:
        return request.form[k]
    if request.json is not None and k in request.json:
        return request.json[k]
    return None


def is_file():
    return IMG_FILE in request.files


def is_file64():
    return get_value(IMG_FILE64) is not None


def is_file_bytes():
    return get_value(IMG_FILE_BYTES) is not None


def get_response_encoding():
    return get_value(IMG_RESPONSE_ENCODING) or fh.ENCODING_BASE64


def get_img_sizes():
    return get_value(IMG_SIZES) or {"img": (500, 50)}


def create_response(imgs):
    #resp = jsonify({'img': img})
    resp = jsonify(imgs)
    resp.status_code = 201
    return resp


def handle_file():
    file = request.files[IMG_FILE]
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        imgs = fh.get_from_file(request.files[IMG_FILE], get_response_encoding(), get_img_sizes())
        return create_response(imgs)
    else:
        resp = jsonify({'message': 'Allowed file types are {0}'.format(ALLOWED_EXTENSIONS)})
        resp.status_code = 400
        return resp


@app.route('/extract-signature', methods=['POST'])
def extract_signature():
    if is_file():
        return handle_file()

    elif is_file64():
        file64 = get_value(IMG_FILE64)
        imgs = fh.get_from_base64(file64, get_response_encoding(), get_img_sizes())
        return create_response(imgs)

    elif is_file_bytes():
        file_bytes = get_value(IMG_FILE_BYTES)
        imgs = fh.get_from_bytes(file_bytes, get_response_encoding(), get_img_sizes())
        return create_response(imgs)

    resp = jsonify({'message': 'No image in the request. Use {} in either form or json request'.format(ALLOWED_KEYS)})
    resp.status_code = 400
    return resp


@app.errorhandler(SignatureException)
def handle_image_exception(error):
    out = error.to_dict(get_response_encoding())
    response = jsonify(out)
    response.status_code = error.status_code
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    bugsnag.notify(e)
    resp = jsonify({"errors": [{"error_code": "unexpected_error", "message": "An error has occurred while processing the image. Please try again in a few moments."}]})
    resp.status_code = 500
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="7000")
