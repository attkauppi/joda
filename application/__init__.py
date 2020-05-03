from flask import Flask

app = Flask(__name__)


from application import plotting
from application import views
