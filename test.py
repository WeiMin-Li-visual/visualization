# encoding:utf-8
# !/usr/bin/env python
import psutil
import time
import random
from threading import Lock
from flask import Flask, render_template
from flask_socketio import SocketIO

async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()


# 后台线程 产生数据，即刻推送至前端
def background_thread():
    count=0
    while count<10:
        count+=1
        socketio.sleep(1)
        t = random.randint(1, 100)
        socketio.emit('server_response',
                      {'data': t}, namespace='/test')


@app.route('/')
def index():
    return render_template('test.html')


@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(target=background_thread)


if __name__ == '__main__':
    socketio.run(app, debug=True)