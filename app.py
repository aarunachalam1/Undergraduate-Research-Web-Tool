from flask import Flask, request, render_template
from gaffke import gaffke_CI

app = Flask(__name__)
application = app  # For compatibility with some deployment setups

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", css_file="styles.css")

@app.route("/boundswithsample", methods=["GET", "POST"])
def gaffke_endpoint():

    result = None
    result = None
    sample_val = ""
    confidence_val = 0.95
    iterations_val = 1000
    side_val = "lower"

    if request.method == "POST":
        try:
            sample_val = request.form["sample"]
            confidence_val = request.form.get("confidence")
            iterations_val = request.form["iterations"]
            side_val = request.form.get("side")

            sample = [float(x) for x in sample_val.split(",")]
            confidence = float(confidence_val)
            iterations = int(iterations_val)
            side = side_val

            bound = gaffke_CI(
                sample, alpha=1-confidence, B=iterations, side=side, bounds=(0, 1)
            )

            result = f"{side.capitalize()} Gaffke bound ({confidence * 100}% Confidence): {bound:.4f}"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template(
        "boundswithsample.html",
        result=result,
        css_file="boundswithsample.css",
        sample_val=sample_val,
        confidence_val=confidence_val,
        iterations_val=iterations_val,
        side_val=side_val)

@app.route("/code_snippets", methods=["GET"])
def code_snippets():
    return render_template("code_snippets.html", css_file="styles.css")

@app.route("/tools", methods=["GET"])
def tools():
    return render_template("tools.html", css_file="styles.css")

@app.route("/publications", methods=["GET"])
def publications():
    return render_template("publications.html", css_file="styles.css")

if __name__ == "__main__":
    app.run(debug=True)