import argparse
import sys

import numpy as np
import pandas as pd
import tritonclient.http as httpclient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    FLAGS = parser.parse_args()

    # For the HTTP client, need to specify large enough concurrency to
    # issue all the inference requests to the server in parallel. For
    # this example we want to be able to send 2 requests concurrently.
    try:
        concurrent_request_count = 2
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, concurrency=concurrent_request_count, ssl=True
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    print("\n=========")
    async_requests = []

    input0_data = pd.read_csv(
        "event000000000-spacepoint-converted.csv", header=None
    ).to_numpy(dtype=np.float32)
    print("Sending request to batching model: input = {}".format(input0_data))
    inputs = [httpclient.InferInput("FEATURES", input0_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input0_data)
    async_requests.append(triton_client.async_infer("ActsExaTrkX", inputs))

    for async_request in async_requests:
        # Get the result from the initiated asynchronous inference
        # request. This call will block till the server responds.
        result = async_request.get_result()
        print("Response: {}".format(result.get_response()))
        print("OUTPUT = {}".format(result.as_numpy("LABELS")))

    np.save("event000000000-backend-output-labels.npy", result.as_numpy("LABELS"))
