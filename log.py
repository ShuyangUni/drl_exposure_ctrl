'''
Author: Shuyang Zhang
Date: 2024-07-02 23:40:52
LastEditors: ShuyangUni shuyang.zhang1995@gmail.com
LastEditTime: 2024-08-25 20:00:31
Description: 

Copyright (c) 2024 by Shuyang Zhang, All Rights Reserved. 
'''
import os


class Log():
    def __init__(self, savepath):
        print(f"create log file: {savepath}...")
        self.f = open(savepath, 'a')
        if os.path.exists(savepath):
            success = True
        else:
            success = False
        assert success, f"create file ({savepath}) failed..."
        self.log_buffer = []

    def __del__(self):
        self.f.close()

    def add_log(self, msg):
        self.log_buffer.append(msg)

    def save_buffer_to_file(self):
        for msg in self.log_buffer:
            self.f.write(msg+"\n")
        self.log_buffer = []
