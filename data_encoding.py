import numpy as np

def encode_data(data, angle_encodings=None, **kwargs):

    new_data = []
    for i, d in data:
        if i in angle_encodings:
            a = encode_angle_deg(d)
            new_data.extend(a)
    return np.array(new_data)


def encode_angle_deg(ang):
    if isinstance(ang, list) or \
            isinstance(ang, np.array) :
        to_rad = np.pi / 180. * np.array(ang)
        return np.concatenate((np.sin(to_rad), np.cos(to_rad)))
    else:
        to_rad = np.pi / 180. * ang
        return np.array([np.sin(to_rad), np.cos(to_rad)])


def decode_angle_deg():
    pass