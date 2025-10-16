from flask import Flask, request, render_template
from gaffke import gaffke_CI

from flask import Flask, request, render_template, Response, stream_with_context
import json
import numpy as np

app = Flask(__name__)
application = app  # For compatibility with some deployment setups

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", css_file="styles.css")


# your existing endpoint that renders the page
@app.route("/boundswithsample", methods=["GET"])
def boundswithsample():
    # defaults for the form
    return render_template(
        "boundswithsample.html",
        result=None,
        css_file="boundswithsample.css",
        sample_val="",
        confidence_val=0.95,
        iterations_val=1000,
        side_val="lower"
    )

def sse_format(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

@app.route("/boundswithsample/stream")
def boundswithsample_stream():
    """
    Streams progressive results as Server-Sent Events.
    Query params:
      sample: "0.1,0.7,0.2"
      confidence: "0.95"
      iterations: "10000"
      side: "lower" or "upper"
    """
    # parse inputs
    try:
        sample_val = request.args.get("sample", "")
        confidence_val = float(request.args.get("confidence", "0.95"))
        iterations_val = int(request.args.get("iterations", "1000"))
        side_val = request.args.get("side", "lower").strip().lower()
        if side_val not in {"lower", "upper"}:
            side_val = "lower"

        sample = [float(x) for x in sample_val.split(",") if x.strip() != ""]
        if len(sample) == 0:
            raise ValueError("Sample is empty.")
        if any((x < 0) for x in sample):
            raise ValueError("Sample values must be in [0, 1].")

        alpha = 1.0 - confidence_val
        max_B = max(10, iterations_val)
        # 10 points from 10 to iterations (clamp if iterations < 10)
        start_B = 10 if iterations_val >= 10 else iterations_val
        Bs = np.unique([int(b) for b in np.linspace(start_B, max_B, num=20)])
    except Exception as e:
        msg = str(e)  # capture before defining the generator

        def err_stream(msg=msg):
            yield sse_format({"type": "error", "message": msg})
            yield "event: end\ndata: {}\n\n"

        return Response(stream_with_context(err_stream()),
                        mimetype="text/event-stream")
    @stream_with_context
    def generate():
        # notify client we are starting
        yield sse_format({"type": "start", "points": len(Bs)})

        for idx, B in enumerate(Bs, start=1):
            cur_vals = []
            for _ in range(20):
                # call your function
                bound = gaffke_CI(sample, alpha=alpha, B=int(B), side=side_val, bounds=(0, 1))
                cur_vals.append(float(bound))

            mean_val = float(np.mean(cur_vals))
            std_val = float(np.std(cur_vals, ddof=1)) if len(cur_vals) > 1 else 0.0

            payload = {
                "type": "update",
                "idx": idx,
                "B": int(B),
                "mean": mean_val,
                "std": std_val,
                "samples": cur_vals  # include raw if you want a table later
            }
            yield sse_format(payload)

        # done
        yield "event: end\ndata: {}\n\n"

    return Response(generate(), mimetype="text/event-stream")


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
    app.run(debug=True, threaded=True)