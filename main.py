from flask import Flask, request, jsonify, render_template
from gaffke import gaffke_CI

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def gaffke_endpoint():

    result = None
    if request.method == "POST":
        try:
            sample = request.form["sample"]
            sample = [float(x) for x in sample.split(",")]
            alpha = float(request.form.get("alpha"))
            iterations = int(request.form["iterations"])
            side = request.form.get("side")

            bound = gaffke_CI(
                sample, alpha=alpha, B=iterations, side=side, bounds=(0, 1)
            )

            result = f"{side.capitalize()} Gaffke bound (1 - {alpha}): {bound:.4f}"

        except Exception as e:
            result = f"Error: {str(e)}"
        
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)