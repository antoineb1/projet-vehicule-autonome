import cv2

class Utils:

    @staticmethod
    def serialize(kp) -> dict:
        if isinstance(kp, list):
            return [Utils.serialize(k) for k in kp]

        return {"pt": kp.pt, "size": kp.size, "angle": kp.angle, "response": kp.response, "octave": kp.octave, "class_id": kp.class_id}

    @staticmethod
    def deserialize(kp) -> cv2.KeyPoint:
        if isinstance(kp, list):
            return [Utils.deserialize(k) for k in kp]

        return cv2.KeyPoint(x=kp["pt"][0], y=kp["pt"][1], size=kp["size"], angle=kp["angle"], response=kp["response"], octave=kp["octave"], class_id=kp["class_id"])