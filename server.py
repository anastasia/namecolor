from flask import Flask, request, render_template
from trained import get_color
from flask import jsonify

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("main.html")
    elif request.method == 'POST':
        color_string = request.form.get('color', None)

        if not color_string:
            return render_template("main.html")

        hex_color = get_color(color_string)
        return render_template("main.html", color=hex_color)


@app.route("/color", methods=['GET'])
def color():
    color_string = request.args.get("name", None)
    if not color_string:
        return jsonify({"color": "", "status": 204, "value": color_string})

    hex_color = get_color(color_string)
    return jsonify({"color": hex_color, "status": 200, "value": color_string})


if __name__ == '__main__':
    app.run()
