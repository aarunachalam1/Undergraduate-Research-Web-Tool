import inspect
from flask import Flask, request, render_template
from gaffke import gaffke_CI
from other_bounds import hoeffding_bound, student_t, anderson_bound
from flask import Flask, request, render_template, Response, stream_with_context
import json
import numpy as np

BOUND_FUNCTIONS = {
    "gaffke": gaffke_CI,
    "student_t": student_t,
    "anderson": anderson_bound,
    "hoeffding": hoeffding_bound
}
app = Flask(__name__)
application = app

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

@app.route("/boundswithsample/simple")
def boundswithsample_simple():
    """
    Simple endpoint for bounds with no Monte Carlo simulation.
    """
    try:
        bound_type = request.args.get("bound").strip().lower()
        sample_val = request.args.get("sample", "")
        confidence_val = float(request.args.get("confidence", "0.95"))
        side_val = request.args.get("side", "lower").strip().lower()
        min_val = float(request.args.get("min_hoeffding", "0"))
        max_val = float(request.args.get("max_hoeffding", "1"))
        
        if bound_type == "hoeffding":
            bounds = (min_val, max_val)
        else:
            bounds = None

        sample = [float(x) for x in sample_val.split(",") if x.strip() != ""]
        
        if side_val not in ["lower", "upper"]:
            raise ValueError("side must be 'lower' or 'upper'.")
    
        if len(sample) == 0:
            raise ValueError("Sample is empty.")
        
        bound_func = BOUND_FUNCTIONS.get(bound_type)

        if bound_func is None:
            raise ValueError(f"Unknown bound type: {bound_type}")

        sig = inspect.signature(bound_func)
        args = {
            "x": sample,
            "alpha": 1 - confidence_val,
            "side": side_val,
            "bounds": bounds
        }
        valid_args = {
            name: val for name, val in args.items() if name in sig.parameters
        }

        result = bound_func(**valid_args)

        return {"value": float(result)}
    
    except Exception as e:
        return {"error": str(e)}

    

@app.route("/boundswithsample/stream")
def boundswithsample_stream():
    """
    Streams progressive Monte Carlo results for Gaffke bound.
    Query params:
      sample: "0.1,0.7,0.2"
      confidence: "0.95"
      iterations: "1000"
      steps: "10"
      iterations_per_step: "5"
      min_max: "0"
      side: "lower" or "upper"
    """
    try:
        sample = [float(x) for x in request.args.get("sample", "").split(",") if x.strip()]
        if not sample:
            raise ValueError("Sample is empty.")

        confidence = float(request.args.get("confidence", 0.95))
        iterations = int(request.args.get("iterations", 1000))
        steps = int(request.args.get("steps", 10))
        iterations_per_step = int(request.args.get("iterations_per_step", 1))
        min_max = float(request.args.get("min_max", 0))
        side = request.args.get("side", "lower").lower()
        if side not in ["lower", "upper"]:
            side = "lower"

        Bs = np.unique(
            np.round(np.exp(np.linspace(np.log(1), np.log(max(10, iterations)), steps))).astype(int)
        )
        Bs[-1] = iterations
    except Exception as e:
        def err_stream():
            yield sse_format({"type": "error", "message": str(e)})
            yield sse_format({"type": "end"})
        return Response(stream_with_context(err_stream()), mimetype="text/event-stream")

    @stream_with_context
    def generate():
        yield sse_format({"type": "start", "points": len(Bs)})

        for idx, B in enumerate(Bs, start=1):
            cur_vals = []
            for _ in range(iterations_per_step):
                try:
                    val = gaffke_CI(x=sample, conf=confidence, B=B, side=side, extrema=min_max)
                except Exception as e:
                    yield sse_format({"type": "error", "message": str(e)})
                    continue
                cur_vals.append(float(val))

            mean_val = float(np.mean(cur_vals))
            std_val = float(np.std(cur_vals, ddof=1)) if len(cur_vals) > 1 else 0.0

            yield sse_format({
                "type": "update",
                "idx": idx,
                "B": int(B),
                "mean": mean_val,
                "std": std_val
            })
        # Send a final data message with type 'end' so the client onmessage handler
        # receives it as JSON (consistent with other messages) instead of relying
        # on a named SSE event which the client doesn't listen for.
        yield sse_format({"type": "end"})

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