# ActsExaTrkXTritonBackend

This custom backend is based on the Triton backend example:
https://github.com/triton-inference-server/backend/tree/main/examples/backends/recommended

## Compile the backend

### Prerequisites
The first step is to compile the ACTS library with the ExaTrkX plugin enabled. This will provide the necessary headers and libraries for the backend to link against.

Then compile the triton backend with the compiled ACTS library as CMAKE_PREFIX_PATH.

## Dev Note
### Worklog:
1. rename "recommended" to "ActsExaTrkX"
2. add ExaTrkXPipeline. 01/17/2024
3. fix the error of "GPU instances not supported". 01/18/2024
4. minimum working example confirmed. 01/22/2024
5. multi-gpu support done with acts version v32.0.0. 01/22/2024

### Todo:
- [x] add instructions for the compilation of the backend.
- [x] test the inference of the backend - Minimum working example!
    - [x] Standalone version passed the test.
    - [x] Triton version.
- [x] multi-gpu support(probably).
- [ ] clean the backend code.
- [ ] model path in the model_repo/1 folder for people without perlmutter.
- [ ] more customized of the backend, e.g. pass config from `config.pbtxt` to the backend.
