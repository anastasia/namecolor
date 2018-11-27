from flask import Flask, request, render_template
app = Flask(__name__)

from trained import get_color

@app.route("/", methods=['GET', 'POST'])
def color():
    if request.method == 'GET':
        return render_template("main.html")
    elif request.method == 'POST':
        color_string = request.form.get('color', None)

        if not color_string:
            return render_template("main.html")

        hex_color = get_color(color_string)
        return render_template("main.html", color=hex_color)


if __name__ == '__main__':
    app.run()