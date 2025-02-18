import requests
import paramiko
from flask import Flask, request, jsonify

app = Flask(__name__)

def download_rknn_file(url, save_path, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(url, username=username, password=password)
    
    sftp = ssh.open_sftp()
    sftp.get(url, save_path)
    sftp.close()
    ssh.close()


   

# 示例用法
# download_rknn_file('http://example.com/path/to/your/file.rknn', '/local/path/to/save/file.rknn')