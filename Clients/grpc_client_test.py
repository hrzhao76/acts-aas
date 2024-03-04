import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(
    "acts-triton.nrp-nautilus.io:443", ssl=True, verbose=True
)
client.is_server_live()
