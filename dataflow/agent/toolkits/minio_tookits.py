from minio import Minio
import os

def local_tool_for_init_client(endpoint, access_key, secret_key, secure=False):
    """初始化 MinIO 客户端"""
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

def local_tool_for_list_buckets(client):
    """列出所有 bucket"""
    return [bucket.name for bucket in client.list_buckets()]

def local_tool_for_list_objects(client, bucket_name, prefix="", recursive=True):
    """列出 bucket 下所有对象"""
    return [obj.object_name for obj in client.list_objects(bucket_name, prefix=prefix, recursive=recursive)]

def local_tool_for_download_file(client, bucket_name, object_name, local_path):
    """下载对象到本地文件"""
    client.fget_object(bucket_name, object_name, local_path)
    return os.path.abspath(local_path)

def local_tool_for_upload_file(client, bucket_name, object_name, local_path):
    """上传本地文件到 MinIO"""
    client.fput_object(bucket_name, object_name, local_path)
    return f"{bucket_name}/{object_name}"

def local_tool_for_read_bytes(client, bucket_name, object_name):
    """读取对象内容为 bytes"""
    response = client.get_object(bucket_name, object_name)
    data = response.read()
    response.close()
    return data

def local_tool_for_delete_object(client, bucket_name, object_name):
    """删除对象"""
    client.remove_object(bucket_name, object_name)
    return True

def local_tool_for_make_bucket(client, bucket_name):
    """新建 bucket（已存在则跳过）"""
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        return True
    return False

def local_tool_for_bucket_exists(client, bucket_name):
    """判断 bucket 是否存在"""
    return client.bucket_exists(bucket_name)
