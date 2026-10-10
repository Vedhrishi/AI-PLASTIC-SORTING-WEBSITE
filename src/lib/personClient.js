import { createRpc } from './workerRpc';

/**
 * Starts the TF.js worker that hosts COCO-SSD (person / bottle detection) and
 * returns a detector whose detect(canvas) has the same shape as coco-ssd's.
 * The canvas pixels are copied once and transferred, so no inference work
 * (and no WebGL shader compilation) ever runs on the UI thread.
 *
 * @param {{ modelUrl: string, preferredBackends?: string[], workerFactory: () => any }} options
 */
export async function loadPersonDetector({ modelUrl, preferredBackends, workerFactory }) {
  const rpc = createRpc(workerFactory());

  let init;
  try {
    init = await rpc.request({ type: 'init', modelUrl, preferredBackends });
  } catch (err) {
    rpc.dispose();
    throw err;
  }

  return {
    backend: /** @type {string} */ (init.backend),
    /**
     * @param {{ width: number, height: number, getContext: (type: '2d') => any }} canvas
     * @returns {Promise<{ bbox: [number, number, number, number], class: string, score: number }[]>}
     */
    async detect(canvas) {
      const { width, height } = canvas;
      const pixels = canvas.getContext('2d').getImageData(0, 0, width, height);
      const reply = await rpc.request({ type: 'detect', data: pixels.data, width, height }, [pixels.data.buffer]);
      return reply.detections;
    },
    dispose: () => rpc.dispose(),
  };
}
