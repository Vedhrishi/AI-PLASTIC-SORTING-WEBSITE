async function sha256Hex(text) {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export async function buildInspectionCertificate(result, { simulated = false } = {}) {
  const scanId = crypto.randomUUID ? crypto.randomUUID() : `scan-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const timestamp = new Date().toISOString();

  const payload = {
    scanId,
    timestamp,
    simulated,
    detectedPolymer: {
      resin: result.resin.label,
      resinCode: result.resin.code,
      confidence: Number((result.resin.confidence * 100).toFixed(1)),
    },
    contamination: {
      level: result.contamination.label,
      index: result.contamination.level,
      confidence: Number((result.contamination.confidence * 100).toFixed(1)),
    },
    directive: {
      action: result.rule.action,
      recyclingRoute: result.rule.route,
      reuseSuggestion: result.rule.reuse,
    },
  };

  const complianceHash = await sha256Hex(JSON.stringify(payload));

  return { ...payload, complianceHash };
}

export function downloadCertificate(certificate) {
  const blob = new Blob([JSON.stringify(certificate, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `inspection-${certificate.scanId}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
