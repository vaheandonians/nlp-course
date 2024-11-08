import time
import os
import binascii
import struct


def generate_mongodb_id():
    timestamp = struct.pack(">I", int(time.time()))
    random_value = os.urandom(5)
    counter = os.urandom(3)
    mongo_id = timestamp + random_value + counter
    return binascii.hexlify(mongo_id).decode()


def is_valid_mongodb_id(mongo_id):
    try:
        binascii.unhexlify(mongo_id)
        return True
    except binascii.Error:
        return False
