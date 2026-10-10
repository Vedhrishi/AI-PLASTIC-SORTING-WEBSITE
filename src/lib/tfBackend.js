/**
 * Picks the first TensorFlow.js backend that actually initialises. WebGL is
 * preferred (GPU); when it is missing or fails (blocklisted GPU, disabled
 * hardware acceleration, headless browsers) the CPU backend keeps the app
 * working, just slower.
 * @param {{ setBackend: (name: string) => Promise<boolean>, ready: () => Promise<void>, getBackend: () => string }} tf
 * @param {string[]} [preferred]
 * @returns {Promise<string>} the backend that is now active
 */
export async function selectTfBackend(tf, preferred = ['webgl', 'cpu']) {
  const failures = [];
  for (const name of preferred) {
    try {
      const ok = await tf.setBackend(name);
      if (ok === false) {
        failures.push(`${name}: not available`);
        continue;
      }
      await tf.ready();
      return tf.getBackend();
    } catch (err) {
      failures.push(`${name}: ${err?.message ?? err}`);
    }
  }
  throw new Error(`No TensorFlow.js backend could be initialised (${failures.join('; ')})`);
}
