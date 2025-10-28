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
        sample_val="",
        confidence_val=0.95,
        iterations_val=1000,
        steps_val=1,
        iterations_per_step_val=1,
        min_max_val=0,
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
        print(request.args)
        sample_val = request.args.get("sample", "")
        confidence_val = float(request.args.get("confidence", "0.95"))
        iterations_val = int(request.args.get("iterations", "1000"))
        steps_val = int(request.args.get("steps", "1"))
        iterations_per_step_val = int(request.args.get("iterations_per_step", "1"))
        min_max_val = float(request.args.get("min_max", "0"))

        side_val = request.args.get("side", "lower").strip().lower()

        if side_val not in ["lower", "upper"]:
            side_val = "lower"

        sample = [float(x) for x in sample_val.split(",") if x.strip() != ""]
        if len(sample) == 0:
            raise ValueError("Sample is empty.")
        if any((x < 0) for x in sample):
            raise ValueError("Sample values must be in [0, inf].")

        max_B = max(10, iterations_val)

        start_B = 10 if steps_val >= 1 else iterations_val
        if steps_val == 1:
            Bs = [iterations_val]
        else:
            # Bs = np.unique([int(b) for b in np.linspace(start_B, max_B, num=steps_val)])
            # log_B = np.log(iterations_val)
            # frac = log_B/steps_val
            # Bs = np.rint(np.exp(np.arange(1, log_B, frac)))
            # Bs[-1] = iterations_val  # ensure last is exactly iterations_val

            start_B = max(1, int(start_B))
            max_B = int(max_B)

            # Grid in log space, inclusive of both endpoints
            grid = np.exp(np.linspace(np.log(start_B), np.log(max_B), steps_val))

            # Round up to avoid duplicates at the low end, then enforce monotonicity
            Bs = np.unique(np.round(grid).astype(int))

            Bs[-1] = max_B

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
            for _ in range(iterations_per_step_val):
                bound = gaffke_CI(sample, conf=confidence_val, B=int(B), side=side_val, extrema=min_max_val)
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