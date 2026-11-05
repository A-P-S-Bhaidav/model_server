# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise the real sample clients against local KServe protocol fixtures."""
import argparse
import json
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import grpc
import numpy as np
from tritonclient.grpc import service_pb2, service_pb2_grpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--mode", choices=["dtype", "async"], required=True)
    args = parser.parse_args()
    seen = []

    def check_input(tensor):
        assert tensor["datatype"] == "FP32", tensor
        seen.append(tensor["shape"][0])
        return tensor["shape"][0]

    class HttpHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            header_size = int(self.headers.get("Inference-Header-Content-Length", len(body)))
            request = json.loads(body[:header_size])
            batch = check_input(request["inputs"][0])
            response = json.dumps({
                "model_name": "resnet",
                "outputs": [{"name": "output", "datatype": "FP32", "shape": [batch, 2],
                             "data": [1.0, 0.0] * batch}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_):
            pass

    class GrpcHandler(service_pb2_grpc.GRPCInferenceServiceServicer):
        def ModelInfer(self, request, context):
            tensor = request.inputs[0]
            batch = check_input({"datatype": tensor.datatype, "shape": list(tensor.shape)})
            response = service_pb2.ModelInferResponse(model_name=request.model_name)
            output = response.outputs.add(name="output", datatype="FP32")
            output.shape.extend([batch, 2])
            response.raw_output_contents.append(np.tile([[1.0, 0.0]], (batch, 1)).astype(np.float32).tobytes())
            return response

    http = ThreadingHTTPServer(("127.0.0.1", 0), HttpHandler)
    http_thread = threading.Thread(target=http.serve_forever, daemon=True)
    http_thread.start()
    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(GrpcHandler(), server)
    grpc_port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    samples = args.repository.resolve() / "client" / "python" / "kserve-api" / "samples"
    cases = (
        [(protocol, asynchronous, dtype, 3, 1, 0, 3)
         for protocol in ["grpc", "http"] for asynchronous in [False, True]
         for dtype in ["uint8", "float64", "float32"]]
        if args.mode == "dtype" else
        [("http", True, "float32", images, batch, iterations, expected)
         for images, batch, iterations, expected in [(4, 1, 1, 1), (6, 2, 0, 3), (4, 1, 6, 6), (4, 1, 0, 4)]]
    )
    failures = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            image_file = Path(directory) / "images.npy"
            for protocol, asynchronous, dtype, images, batch, iterations, expected in cases:
                np.save(image_file, np.arange(images * 12, dtype=dtype).reshape(images, 2, 2, 3))
                sample = f"{protocol}_{'async_' if asynchronous else ''}infer_resnet.py"
                command = [sys.executable, str(samples / sample), "--images_numpy_path", str(image_file),
                           "--transpose_input", "False", "--iterations", str(iterations),
                           "--batchsize", str(batch), "--output_name", "output",
                           f"--{protocol}_address", "127.0.0.1", f"--{protocol}_port",
                           str(grpc_port if protocol == "grpc" else http.server_port)]
                before = len(seen)
                result = subprocess.run(command, cwd=samples, capture_output=True, text=True, timeout=30)
                case = f"{sample} dtype={dtype} images={images} batch={batch} iterations={iterations}"
                if result.returncode or len(seen) - before != expected:
                    failures.append(case)
                    print(f"FAIL: {case}\n{result.stdout}\n{result.stderr}")
                else:
                    assert seen[before:] == [batch] * expected
                    print(f"PASS: {case}; {expected} requests")
    finally:
        http.shutdown()
        http.server_close()
        http_thread.join(timeout=5)
        server.stop(0).wait(timeout=5)
    assert not failures, failures
    print(f"{len(cases)} real-client cases passed against local HTTP/gRPC fixtures; no OVMS inference server or model was used.")


if __name__ == "__main__":
    main()
