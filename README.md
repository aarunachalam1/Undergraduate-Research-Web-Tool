### Undergraduate Research Web Tool

The purpose of this repo is to provide a simple web interface to test out various bounds on unknown, discrete distributions, with known or unknown support. This is to be used as a resource for an ongoing research project under Professor Erik Learned-Miller regarding a novel method for finding confidence intervals on such distributions. Currently, only the Monte Carlo variant of the Gaffke bound exists but more will be added shortly.

### Usage

1. Clone the repository with:

```bash
git clone https://github.com/aarunachalam1/undergraduate-research-web-tool.git
cd undergraduate-research-web-tool
```
2. Install dependencies
```bash
pip install -r requiremenets.txt
```
3. Run the Flask development server (while in the repo's folder on your local machine)

```bash
flask --app main run
```

4. Open http://127.0.0.1:5000 in your browser to access the tool.
