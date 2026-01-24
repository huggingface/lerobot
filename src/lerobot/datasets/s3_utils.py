import json
import boto3
import os
from dotenv import load_dotenv
import builtins


def monkey_patch_open(key_id: str, secret: str, endpoint_url: str):
    """
    Monkey patch the open function to use the s3 client.
    """

    load_dotenv()

    s3_client = boto3.client(
        's3',
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        endpoint_url=endpoint_url
    )
    transport_params = {
        'client': s3_client
    }

    # save original open function
    _original_open = builtins.open

    # this import changes default builtins.open to smart_open.open
    from smart_open import open as s3_open

    def patched_open(file, mode='r', *args, **kwargs):
        file_str = str(file)

        if file_str.startswith('s3://'):
            return s3_open(file_str, mode, transport_params=transport_params, *args, **kwargs)
        else:
            return _original_open(file_str, mode, *args, **kwargs)

    builtins.open = patched_open
