"""CPU-only health endpoint for the HF Jobs runtime image; never loads models."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class Health(BaseHTTPRequestHandler):
    """Expose build provenance without loading weights or touching CUDA."""

    def do_GET(self):
        """Serve the immutable build receipt."""
        payload = json.loads(Path("/opt/hand-worker/image-version.json").read_text())
        payload.update(
            status="ready",
            inference_tested=False,
            models_embedded=payload.get("build", {}).get("model_weights_embedded", False),
        )
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        """Keep health polling out of the build log."""
        pass


HTTPServer(("0.0.0.0", 7860), Health).serve_forever()  # nosec B104 - HF Space health endpoint
