from flask import Flask, request, render_template
from gaffke import gaffke_CI

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", css_file="styles.css")

@app.route("/boundswithsample", methods=["GET", "POST"])
def gaffke_endpoint():
    
    result = None
    
    if request.method == "POST":
        try:
            sample = request.form["sample"]
            sample = [float(x) for x in sample.split(",")]
            confidence = float(request.form.get("confidence"))
            iterations = int(request.form["iterations"])
            side = request.form.get("side")

            bound = gaffke_CI(
                sample, alpha=1-confidence, B=iterations, side=side, bounds=(0, 1)
            )

            result = f"{side.capitalize()} Gaffke bound ({confidence * 100}% Confidence): {bound:.4f}"

        except Exception as e:
            result = f"Error: {str(e)}"
    
    return render_template(
        "boundswithsample.html", result=result, css_file="boundswithsample.css")

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