#!flask/bin/python

from flask import Flask, request, redirect,render_template
import os
from werkzeug.utils import secure_filename
from document_classification_model import document_prediction as pf 
import json

application = Flask(__name__)
@application.route("/", methods=["GET", "POST"])
def mainPage():
    return render_template("text_classifier.html")


@application.route("/upload-files", methods=["GET", "POST"])
def upload_file():
	if request.method == "GET":
         words=request.args.get('word')
         return (pf.prediction(str(words)))


if __name__ == '__main__':
    application.run(debug=True, use_reloader=True)


