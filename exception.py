class SignatureException(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None, error_code=None, img=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
        self.error_code = error_code
        self.img = img

    def to_dict(self, encoding):
        from file_handler import convert_img
        rv = dict(self.payload or ())
        rv["errors"] = [{"error_code": self.error_code, "message": self.message}] # possibility for future extension for multiple errors
        rv["img"] = convert_img(self.img, encoding)
        return rv
