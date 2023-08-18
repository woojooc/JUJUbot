from flask import Flask, render_template, url_for
from .views import main_views, stt_views


app = Flask(__name__)

app.register_blueprint(main_views.bp)
app.register_blueprint(stt_views.bp)

if __name__ == "__main__":
    app.run(debug=True)

