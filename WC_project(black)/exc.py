from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
import threading, time
import winsound as sd

from threading import Thread


# def init():
#     t1 = Thread(target=ex())
#     t1.daemon = True
#     t1.start()
#
#
# def ex():
#     print("되냐?")
#     fr = 2000  # range : 37 ~ 32767
#     du = 500  # 1000 ms ==1second
#     sd.Beep(fr, du)  # winsound.Beep(frequency, duration)
#     exec(open("Thdataapi.py", encoding='utf-8').read())
#
#
# init()

def ex():
    print("되냐?")
    fr = 2000  # range : 37 ~ 32767
    du = 500  # 1000 ms ==1second
    sd.Beep(fr, du)  # winsound.Beep(frequency, duration)
    #exec(open("Thdataapi.py", encoding='utf-8').read())

    threading.Timer(10, ex).start()



abc=list()
cvb=list()
cvb.append('1')
abc.append(cvb)
print(cvb)
print(abc[0])
