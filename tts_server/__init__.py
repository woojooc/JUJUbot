from flask import Flask

def create_app():
    app = Flask(__name__)

    from .Controller import tts_controller
    app.register_blueprint(tts_controller.bp)

    return app


if __name__ == '__main__':
    app = create_app()
    # main으로 실행하지 않을 시 flask run --host=0.0.0.0 --port=5002
    app.run(port=5002)


