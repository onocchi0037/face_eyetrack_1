# -*- coding: utf-8 -*-
from flask import Flask
import json
import socket

HOST = '127.0.0.1'  # The remote host
PORT = 50007  # The same port as used by the server

app = Flask(__name__)

@app.route("/mcs", methods=['GET'])
def mcs():
  s   = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect((HOST,PORT))
  s.sendall(b"get pose")
  data = s.recv(2048)
  s.close()
  return data.decode()

if __name__ == "__main__":
  app.debug = True
  app.run()
