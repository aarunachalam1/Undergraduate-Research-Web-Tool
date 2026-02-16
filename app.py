import inspect
import io
from flask import Flask, abort, request, render_template, jsonify, send_file, Response, stream_with_context
import requests
from gaffke import gaffke_CI
from other_bounds import hoeffding_bound, student_t, anderson_bound
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

#
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

        last_mean = None
        last_std = None
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
            last_mean = mean_val
            last_std = std_val

            yield sse_format({
                "type": "update",
                "idx": idx,
                "B": int(B),
                "mean": mean_val,
                "std": std_val
            })

        yield sse_format({"type": "end", "mean": last_mean, "std": last_std})

    return Response(generate(), mimetype="text/event-stream")

@app.route("/sample_dist")
def sample_dist():
    dist = request.args.get("dist", "uniform")
    num_samples = int(request.args.get("num_samples", 20))
    num_experiments = int(request.args.get("num_experiments", 1))
    
    if dist == "uniform":
        a = float(request.args.get("a", 0))
        b = float(request.args.get("b", 1))
        experiments = []
        for _ in range(num_experiments):
            vals = list(map(float, np.round(np.random.uniform(a, b, num_samples), 6)))
            experiments.append(vals)
    elif dist == "binomial":
        n = int(request.args.get("n", 10))
        p = float(request.args.get("p", 0.5))
        experiments = []
        for _ in range(num_experiments):
            vals = list(map(float, np.random.binomial(n, p, num_samples)))
            experiments.append(vals)
    elif dist == "beta":
        alpha_param = float(request.args.get("alpha", 1))
        beta_param = float(request.args.get("beta", 1))
        experiments = []
        for _ in range(num_experiments):
            vals = list(map(float, np.round(np.random.beta(alpha_param, beta_param, num_samples), 6)))
            experiments.append(vals)
    else:
        return jsonify({"error": "Unknown distribution"}), 400
    
    if num_experiments == 1:
        return jsonify({"values": experiments[0]})
    else:
        return jsonify({"experiments": experiments})

# Renders the sample from distribution page
@app.route("/samplefromdist", methods=["GET"])
def samplefromdist():
    return render_template(
        "samplefromdist.html",
        css_file="samplefromdist.css",
        side_val="lower",
        confidence_val=0.95,
        iterations_val=1000,
        steps_val=10,
        iterations_per_step_val=1,
        min_max_val=0,
        min_hoeffding_val=0,
        max_hoeffding_val=1
    )

@app.route("/samplefromdist/stream")
def samplefromdist_stream():

    try:
        # New parameters: sample_sizes (comma-separated ints) and num_repeats (repeats per size)
        sample_sizes_str = request.args.get("sample_sizes", "").strip()
        num_repeats = int(request.args.get("num_repeats", request.args.get("num-samples", 1)))
        dist = request.args.get("dist", "uniform").strip().lower()
        # distribution params
        a = float(request.args.get("a", 0))
        b = float(request.args.get("b", 1))
        n = int(request.args.get("n", 10))
        p = float(request.args.get("p", 0.5))

        bound_type = request.args.get("bound", "gaffke").strip().lower()
        confidence = float(request.args.get("confidence", 0.95))
        iterations = int(request.args.get("iterations", 1000))
        min_max = float(request.args.get("min_max", 0))
        side = request.args.get("side", "lower").lower()
        min_hoeffding = float(request.args.get("min_hoeffding", 0))
        max_hoeffding = float(request.args.get("max_hoeffding", 1))

        if side not in ["lower", "upper"]:
            side = "lower"

        # Parse sample sizes
        if sample_sizes_str:
            sample_sizes = [int(x) for x in sample_sizes_str.split(",") if x.strip()]
        else:
            # Fallback to single sample size param
            single_size = request.args.get("sample_size") or request.args.get("sample-size")
            sample_sizes = [int(single_size)] if single_size else []

        if not sample_sizes:
            raise ValueError("No sample sizes provided.")

        if bound_type not in BOUND_FUNCTIONS:
            raise ValueError(f"Unknown bound type: {bound_type}")

    except Exception as e:
        def err_stream():
            yield sse_format({"type": "error", "message": str(e)})
            yield sse_format({"type": "end"})
        return Response(stream_with_context(err_stream()), mimetype="text/event-stream")

    @stream_with_context
    def generate():
        # Calculate the true mean of the distribution
        true_mean = None
        if dist == "uniform":
            true_mean = (a + b) / 2
        elif dist == "binomial":
            true_mean = n * p
        elif dist == "beta":
            alpha_param = float(request.args.get("alpha", 1))
            beta_param = float(request.args.get("beta", 1))
            true_mean = alpha_param / (alpha_param + beta_param)
        
        yield sse_format({"type": "start", "points": len(sample_sizes), "true_mean": true_mean})

        last_mean = None
        last_std = None
        bound_func = BOUND_FUNCTIONS[bound_type]
        sig = inspect.signature(bound_func)

        for idx, s in enumerate(sample_sizes, start=1):
            bound_vals_for_size = []
            for rep in range(num_repeats):
                if dist == "uniform":
                    sample_vals = list(map(float, np.round(np.random.uniform(a, b, s), 6)))
                elif dist == "binomial":
                    sample_vals = list(map(float, np.random.binomial(n, p, s)))
                elif dist == "beta":
                    alpha_param = float(request.args.get("alpha", 1))
                    beta_param = float(request.args.get("beta", 1))
                    sample_vals = list(map(float, np.round(np.random.beta(alpha_param, beta_param, s), 6)))
                else:
                    yield sse_format({"type": "error", "message": f"Unknown distribution: {dist}"})
                    continue

                try:
                    if bound_type == "gaffke":
                        val = gaffke_CI(x=sample_vals, conf=confidence, B=iterations, side=side, extrema=min_max)
                    else:
                        bounds = (min_hoeffding, max_hoeffding) if bound_type == "hoeffding" else None
                        args = {
                            "x": sample_vals,
                            "alpha": 1 - confidence,
                            "side": side,
                            "bounds": bounds
                        }
                        valid_args = { name: v for name, v in args.items() if name in sig.parameters }
                        val = bound_func(**valid_args)
                    bound_vals_for_size.append(float(val))
                except Exception as e:
                    yield sse_format({"type": "error", "message": str(e)})
                    continue

            if not bound_vals_for_size:
                mean_val = 0.0
                std_val = 0.0
            else:
                mean_val = float(np.mean(bound_vals_for_size))
                std_val = float(np.std(bound_vals_for_size, ddof=1)) if len(bound_vals_for_size) > 1 else 0.0
                last_mean = mean_val
                last_std = std_val

            yield sse_format({
                "type": "update",
                "idx": idx,
                "experiment_num": idx,
                "sample_size": int(s),
                "mean": mean_val,
                "std": std_val
            })

        yield sse_format({"type": "end", "mean": last_mean, "std": last_std})

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

@app.route("/download_proxy")
def download_proxy():
    url = request.args.get("url", "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        abort(400, "Invalid URL")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        abort(502, f"Upstream fetch failed: {e}")

    filename = url.split("/")[-1] or "download.txt"
    return send_file(
        io.BytesIO(r.content),
        as_attachment=True,
        download_name=filename,
        mimetype="application/octet-stream",
        max_age=0
    )

if __name__ == "__main__":
    app.run(debug=True, threaded=True)