from flask import Flask, render_template, url_for

from views import main_views, calling_views, voicecall_views


app = Flask(__name__)

app.register_blueprint(main_views.bp)
app.register_blueprint(voicecall_views.bp)
app.register_blueprint(calling_views.bp)

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')
    app.run(host='0.0.0.0', port=5000, debug=True)

