import { createRpc } from './workerRpc';
import { Tensor } from './tensor';

/**
 * Starts an inference worker, loads the given ONNX models inside it and
 * returns session-like objects whose run() has the same shape as
 * onnxruntime, so detectPlastic / classifyResin work unchanged.
 *
 * @param {{ models: Record<string, { url: string, providers?: string[], reportProgress?: boolean }>, onProgress?: (fraction: number) => void, workerFactory: () => any }} options
 */
export async function loadInferenceSessions({ models, onProgress, workerFactory }) {
  const rpc = createRpc(workerFactory(), { onProgress });

  let init;
  try {
    init = await rpc.request({ type: 'init', models });
  } catch (err) {
    rpc.dispose();
    throw err;
  }

  const sessions = {};
  for (const [key, described] of Object.entries(init.models)) {
    sessions[key] = {
      inputNames: described.inputNames,
      outputNames: described.outputNames,
      async run(feeds) {
        const [inputName] = Object.keys(feeds);
        const input = feeds[inputName];
        const outputName = described.outputNames[0];
        // The input buffer is transferred to the worker, not copied.
        const reply = await rpc.request(
          { type: 'run', model: key, inputName, outputName, data: input.data, dims: input.dims },
          [input.data.buffer]
        );
        return { [outputName]: new Tensor('float32', reply.data, reply.dims) };
      },
    };
  }

  return { sessions, threads: init.threads, dispose: () => rpc.dispose() };
}
