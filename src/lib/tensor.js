// Minimal tensor container shared by the main thread and the inference
// worker. The main thread never loads onnxruntime-web: it builds these, the
// worker turns them into ONNX Runtime tensors and sends results back.
export class Tensor {
  /**
   * @param {'float32'} type
   * @param {Float32Array} data
   * @param {number[]} dims
   */
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }

  dispose() {
    this.disposed = true;
  }
}
